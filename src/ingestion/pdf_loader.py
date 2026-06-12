"""
pdf_body_extractor.py (Updated)
================================
Original: Extracts main body text from structured PDFs using hairline rules.
Added:    Chunker-compatible methods that produce enriched chunks with
          structural metadata, contextual prefixes, and page tracking.

New methods
-----------
    extract_with_page_map()  → full text + character-position-to-page mapping
    extract_sections()       → structural sections (Articles, Chapters, Recitals)
    prepare_chunks()         → production-ready chunks for vector store

Quick start (chunking pipeline)
-------------------------------
    from pdf_body_extractor import PDFBodyExtractor

    with PDFBodyExtractor("eu_ai_act.pdf") as extractor:
        chunks = extractor.prepare_chunks(chunk_size=512, chunk_overlap=50)

        for chunk in chunks[:3]:
            print(chunk["text_for_embedding"][:200])
            print(chunk["metadata"])
            print()
"""

from __future__ import annotations

import re
import fitz  # PyMuPDF
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterator
from pathlib import Path


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class PageBounds:
    """Y-coordinates of the three structural separator lines on a page."""
    header_y:   float | None = None
    footer_y:   float | None = None
    footnote_y: float | None = None

    def __repr__(self) -> str:
        return (
            f"PageBounds(header_y={self.header_y}, "
            f"footer_y={self.footer_y}, "
            f"footnote_y={self.footnote_y})"
        )


@dataclass
class TextBlock:
    """A single text block within the body region of a page."""
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    block_no: int

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0


@dataclass
class PageResult:
    """Extraction result for a single page."""
    page_num: int
    bounds: PageBounds
    blocks: list[TextBlock] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(b.text.strip() for b in self.blocks)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class PDFBodyExtractor:
    """
    Extracts the main body text from structured PDFs using hairline rules
    to identify zone boundaries, with added support for structure-aware
    chunking for RAG pipelines.

    Parameters
    ----------
    pdf_path : str
        Path to the PDF file.
    start_page, end_page : int | None
        Page range (1-based, inclusive). Defaults to full document.
    hairline_max_height : float
        Max rect height (px) to qualify as a separator line. Default: 2.0
    full_width_threshold : float
        Min merged width (px) for header/footer classification. Default: 400.0
    footnote_max_width : float
        Max merged width (px) for footnote classification. Default: 100.0
    y_snap_tolerance : float
        Y-axis tolerance for merging collinear segments. Default: 1.5
    body_margin : float
        Extra px of slack at zone boundaries. Default: 2.0
    text_separator : str
        Separator when joining body blocks. Default: "\\n"
    """

    def __init__(
        self,
        pdf_path: str,
        *,
        start_page:           int | None = None,
        end_page:             int | None = None,
        hairline_max_height:  float = 2.0,
        hairline_min_width:   float = 10.0,
        full_width_threshold: float = 400.0,
        footnote_max_width:   float = 100.0,
        y_snap_tolerance:     float = 1.5,
        body_margin:          float = 2.0,
        text_separator:       str   = "\n",
    ) -> None:
        self.pdf_path             = pdf_path
        self.hairline_max_height  = hairline_max_height
        self.hairline_min_width   = hairline_min_width
        self.full_width_threshold = full_width_threshold
        self.footnote_max_width   = footnote_max_width
        self.y_snap_tolerance     = y_snap_tolerance
        self.body_margin          = body_margin
        self.text_separator       = text_separator

        self._doc: fitz.Document = fitz.open(pdf_path)
        self._total_pages: int   = len(self._doc)

        self.start_page = start_page
        self.end_page   = end_page

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def start_page(self) -> int:
        return self._start_page_1based

    @start_page.setter
    def start_page(self, value: int | None) -> None:
        resolved = 1 if value is None else int(value)
        if not (1 <= resolved <= self._total_pages):
            raise ValueError(f"start_page={resolved} out of range [1, {self._total_pages}]")
        self._start_page_1based = resolved

    @property
    def end_page(self) -> int:
        return self._end_page_1based

    @end_page.setter
    def end_page(self, value: int | None) -> None:
        resolved = self._total_pages if value is None else int(value)
        if not (1 <= resolved <= self._total_pages):
            raise ValueError(f"end_page={resolved} out of range [1, {self._total_pages}]")
        self._end_page_1based = resolved

    @property
    def total_pages(self) -> int:
        return self._total_pages

    @property
    def page_range(self) -> range:
        return range(self._start_page_1based, self._end_page_1based + 1)

    # ------------------------------------------------------------------
    # Core internal helpers (original)
    # ------------------------------------------------------------------

    def _collect_hairlines(self, drawings: list) -> list:
        lines = []
        for d in drawings:
            r = d["rect"]
            if d.get("type") != "f":
                continue
            if r.height > self.hairline_max_height and r.width > self.hairline_max_height:
                continue
            if max(r.width, r.height) < self.hairline_min_width:
                continue
            lines.append(r)
        return lines

    def _merge_collinear(self, segments: list) -> list[tuple[float, float, float]]:
        if not segments:
            return []
        groups: dict[float, list] = defaultdict(list)
        for r in segments:
            key = round(r.y0 / self.y_snap_tolerance) * self.y_snap_tolerance
            groups[key].append(r)
        merged = []
        for _, rects in groups.items():
            x0 = min(r.x0 for r in rects)
            x1 = max(r.x1 for r in rects)
            y0 = sum(r.y0 for r in rects) / len(rects)
            merged.append((y0, x0, x1))
        return sorted(merged)

    def _classify_lines(self, page: fitz.Page) -> PageBounds:
        drawings = page.get_drawings()
        segments = self._collect_hairlines(drawings)
        lines    = self._merge_collinear(segments)
        page_mid = page.rect.height / 2.0
        bounds   = PageBounds()
        for y0, x0, x1 in lines:
            width = x1 - x0
            if width >= self.full_width_threshold:
                if y0 < page_mid:
                    bounds.header_y = y0
                else:
                    bounds.footer_y = y0
            elif width <= self.footnote_max_width:
                bounds.footnote_y = y0
        return bounds

    def _extract_blocks(self, page: fitz.Page, bounds: PageBounds) -> list[TextBlock]:
        top = (bounds.header_y or 0.0) + self.body_margin
        if bounds.footnote_y is not None:
            bottom = bounds.footnote_y - self.body_margin
        elif bounds.footer_y is not None:
            bottom = bounds.footer_y - self.body_margin
        else:
            bottom = page.rect.height
        body_blocks: list[TextBlock] = []
        for b in page.get_text("blocks"):
            bx0, by0, bx1, by1, text, block_no, block_type = b
            if block_type != 0:
                continue
            if not text.strip():
                continue
            if by0 >= top and by1 <= bottom:
                body_blocks.append(
                    TextBlock(x0=bx0, y0=by0, x1=bx1, y1=by1, text=text, block_no=block_no)
                )
        body_blocks.sort(key=lambda b: (b.y0, b.x0))
        return body_blocks

    # ------------------------------------------------------------------
    # Original public API
    # ------------------------------------------------------------------

    def extract_page(self, page_num: int) -> PageResult:
        page   = self._doc[page_num - 1]
        bounds = self._classify_lines(page)
        blocks = self._extract_blocks(page, bounds)
        return PageResult(page_num=page_num, bounds=bounds, blocks=blocks)

    def iter_pages(self) -> Iterator[tuple[int, list[TextBlock]]]:
        for page_num in self.page_range:
            page   = self._doc[page_num - 1]
            bounds = self._classify_lines(page)
            blocks = self._extract_blocks(page, bounds)
            yield page_num, blocks

    def extract_all(self) -> list[PageResult]:
        results = []
        for page_num in self.page_range:
            page   = self._doc[page_num - 1]
            bounds = self._classify_lines(page)
            blocks = self._extract_blocks(page, bounds)
            results.append(PageResult(page_num=page_num, bounds=bounds, blocks=blocks))
        return results

    def extract_text(self) -> str:
        page_texts = []
        for _, blocks in self.iter_pages():
            page_text = self.text_separator.join(b.text.strip() for b in blocks)
            if page_text:
                page_texts.append(page_text)
        return self.text_separator.join(page_texts)

    # ==================================================================
    # NEW: Chunker-compatible methods
    # ==================================================================

    def extract_with_page_map(self) -> tuple[str, list[dict]]:
        """
        Extract full body text AND a page map that tracks which page
        each character range came from.

        Returns
        -------
        (full_text, page_map)

        full_text : str
            Complete body text across all pages in the active range.

        page_map : list[dict]
            Each entry: {"page_num": int, "start": int, "end": int}
            where start/end are character positions in full_text.

        Usage
        -----
            text, page_map = extractor.extract_with_page_map()

            # Find which page character position 5000 belongs to:
            for pm in page_map:
                if pm["start"] <= 5000 < pm["end"]:
                    print(f"Character 5000 is on page {pm['page_num']}")
        """
        full_text = ""
        page_map = []

        for page_num, blocks in self.iter_pages():
            page_text = self.text_separator.join(b.text.strip() for b in blocks)
            if not page_text:
                continue

            start = len(full_text)
            full_text += page_text + "\n"
            end = len(full_text)

            page_map.append({
                "page_num": page_num,
                "start": start,
                "end": end,
            })

        return full_text, page_map

    def _find_page_for_position(self, position: int, page_map: list[dict]) -> int:
        """Given a character position in full_text, return the page number."""
        for pm in page_map:
            if pm["start"] <= position < pm["end"]:
                return pm["page_num"]
        # Fallback: return last page
        return page_map[-1]["page_num"] if page_map else 1

    def extract_sections(self) -> list[dict]:
        """
        Detect structural sections (Chapters, Articles, Recitals, Annexes)
        from the extracted body text.

        Returns a list of sections, each containing:
        {
            "type":       "article" | "recital" | "annex" | "preamble",
            "label":      "Article 5" | "Recital 42" | "ANNEX III",
            "title":      "Prohibited AI practices" | "",
            "chapter":    "CHAPTER II" | "",
            "chapter_title": "PROHIBITED AI PRACTICES" | "",
            "content":    "Full text of this section...",
            "start_page": 51,
            "char_count": 11427,
        }

        The sections are returned in document order.
        """
        full_text, page_map = self.extract_with_page_map()
        sections = []

        # --- Detect chapter markers (for metadata enrichment) ---
        chapter_markers = []
        for m in re.finditer(
            r'(CHAPTER\s+[IVXLC]+)\s*\n\s*(.+?)(?=\n)',
            full_text
        ):
            chapter_markers.append({
                "label": m.group(1).strip(),
                "title": m.group(2).strip(),
                "position": m.start(),
            })

        def _find_chapter(position: int) -> tuple[str, str]:
            """Find which chapter a position belongs to."""
            current_chapter = ""
            current_title = ""
            for cm in chapter_markers:
                if cm["position"] <= position:
                    current_chapter = cm["label"]
                    current_title = cm["title"]
                else:
                    break
            return current_chapter, current_title

        # --- Split into articles ---
        # Pattern: "Article N\nTitle text\n" followed by content until next Article
        article_pattern = re.compile(
            r'\nArticle\s+(\d+)\s*\n\s*(.+?)(?=\n)',
        )
        article_matches = list(article_pattern.finditer(full_text))

        if article_matches:
            # Everything BEFORE the first article is the preamble (recitals)
            preamble_end = article_matches[0].start()
            preamble_text = full_text[:preamble_end].strip()

            if preamble_text:
                # Split preamble into individual recitals
                recital_pattern = re.compile(r'\((\d+)\)\s*\n', re.MULTILINE)
                recital_matches = list(recital_pattern.finditer(preamble_text))

                if recital_matches:
                    for i, rm in enumerate(recital_matches):
                        r_start = rm.start()
                        r_end = recital_matches[i + 1].start() if i + 1 < len(recital_matches) else len(preamble_text)
                        r_content = preamble_text[r_start:r_end].strip()
                        r_num = rm.group(1)

                        sections.append({
                            "type": "recital",
                            "label": f"Recital {r_num}",
                            "title": "",
                            "chapter": "",
                            "chapter_title": "",
                            "content": r_content,
                            "start_page": self._find_page_for_position(r_start, page_map),
                            "char_count": len(r_content),
                        })
                else:
                    # No recital pattern — treat entire preamble as one section
                    sections.append({
                        "type": "preamble",
                        "label": "Preamble",
                        "title": "",
                        "chapter": "",
                        "chapter_title": "",
                        "content": preamble_text,
                        "start_page": self._find_page_for_position(0, page_map),
                        "char_count": len(preamble_text),
                    })

            # Process each article
            for i, am in enumerate(article_matches):
                art_num = am.group(1)
                art_title = am.group(2).strip()

                # Content runs from end of this match to start of next article (or end of articles zone)
                content_start = am.end()

                # Find end: next article or start of annexes
                if i + 1 < len(article_matches):
                    content_end = article_matches[i + 1].start()
                else:
                    # Last article — find where annexes start, or use end of text
                    annex_start = re.search(r'\nANNEX\s+[IVXLC]+\s*\n', full_text[content_start:])
                    if annex_start:
                        content_end = content_start + annex_start.start()
                    else:
                        content_end = len(full_text)

                content = full_text[content_start:content_end].strip()
                chapter, chapter_title = _find_chapter(am.start())
                page = self._find_page_for_position(am.start(), page_map)

                sections.append({
                    "type": "article",
                    "label": f"Article {art_num}",
                    "title": art_title,
                    "chapter": chapter,
                    "chapter_title": chapter_title,
                    "content": content,
                    "start_page": page,
                    "char_count": len(content),
                })

        # --- Detect annexes ---
        annex_pattern = re.compile(r'\n(ANNEX\s+[IVXLC]+[A-Za-z\s]*)\s*\n')
        annex_matches = list(annex_pattern.finditer(full_text))

        for i, am in enumerate(annex_matches):
            content_start = am.end()
            content_end = annex_matches[i + 1].start() if i + 1 < len(annex_matches) else len(full_text)
            content = full_text[content_start:content_end].strip()

            sections.append({
                "type": "annex",
                "label": am.group(1).strip(),
                "title": "",
                "chapter": "",
                "chapter_title": "",
                "content": content,
                "start_page": self._find_page_for_position(am.start(), page_map),
                "char_count": len(content),
            })

        return sections

    def prepare_chunks(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        include_recitals: bool = True,
        include_annexes: bool = True,
    ) -> list[dict]:
        """
        The main chunking method. Produces production-ready chunks for
        a vector store with structural metadata and contextual prefixes.

        Strategy: Recursive character splitting (512 tokens, 50 overlap)
        with structural metadata enrichment and article-title prefixing.

        Parameters
        ----------
        chunk_size : int
            Target chunk size in characters. Default: 512
        chunk_overlap : int
            Overlap between consecutive chunks. Default: 50
        include_recitals : bool
            Whether to include preamble recitals. Default: True
        include_annexes : bool
            Whether to include annexes. Default: True

        Returns
        -------
        list[dict], each containing:
        {
            "text_for_embedding": str,  # contextual prefix + chunk text (use for embedding)
            "text_original":      str,  # chunk text without prefix (use for display)
            "metadata": {
                "source":         str,  # filename
                "section_type":   str,  # "article" | "recital" | "annex" | "preamble"
                "label":          str,  # "Article 5"
                "title":          str,  # "Prohibited AI practices"
                "chapter":        str,  # "CHAPTER II"
                "chapter_title":  str,  # "PROHIBITED AI PRACTICES"
                "start_page":     int,  # page where this section starts
                "chunk_index":    int,  # global chunk index (0-based)
                "section_part":   int,  # which part within this section (1-based)
                "section_parts":  int,  # total parts this section was split into
            }
        }
        """
        sections = self.extract_sections()
        source = Path(self.pdf_path).name
        all_chunks = []
        global_index = 0

        for section in sections:
            # Filter by type
            if section["type"] == "recital" and not include_recitals:
                continue
            if section["type"] == "annex" and not include_annexes:
                continue

            content = section["content"]

            # Build contextual prefix for embedding enrichment
            prefix = ""
            if section["type"] == "article" and section["title"]:
                prefix = f"{section['label']}: {section['title']}. "
            elif section["type"] == "recital":
                prefix = f"{section['label']}. "
            elif section["type"] == "annex":
                prefix = f"{section['label']}. "

            # Split the section content into chunks
            sub_chunks = self._recursive_split(content, chunk_size, chunk_overlap)

            for part_idx, chunk_text in enumerate(sub_chunks):
                # Clean the chunk text
                cleaned = self._clean_chunk(chunk_text)

                if not cleaned or len(cleaned.strip()) < 20:
                    continue  # skip tiny/empty chunks

                all_chunks.append({
                    "text_for_embedding": prefix + cleaned,
                    "text_original": cleaned,
                    "metadata": {
                        "source": source,
                        "section_type": section["type"],
                        "label": section["label"],
                        "title": section.get("title", ""),
                        "chapter": section.get("chapter", ""),
                        "chapter_title": section.get("chapter_title", ""),
                        "start_page": section["start_page"],
                        "chunk_index": global_index,
                        "section_part": part_idx + 1,
                        "section_parts": len(sub_chunks),
                    },
                })
                global_index += 1

        return all_chunks

    def _recursive_split(
        self,
        text: str,
        chunk_size: int,
        overlap: int,
        separators: list[str] | None = None,
    ) -> list[str]:
        """
        Recursive character text splitter.

        Tries to split on the best separator first (paragraph breaks),
        falling back to less ideal separators (newlines, sentences, words,
        hard character split) only when a piece exceeds chunk_size.

        This is equivalent to LangChain's RecursiveCharacterTextSplitter
        but implemented without the dependency.
        """
        if separators is None:
            separators = ["\n\n", "\n", ". ", ", ", " ", ""]

        if not text or len(text) <= chunk_size:
            return [text] if text.strip() else []

        # Find the best separator that exists in the text
        separator = ""
        for sep in separators:
            if sep == "":
                separator = sep
                break
            if sep in text:
                separator = sep
                break

        # Split text by the chosen separator
        if separator:
            pieces = text.split(separator)
        else:
            # Hard character split (last resort)
            pieces = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

        # Merge pieces into chunks that respect chunk_size
        chunks = []
        current_chunk = ""

        for piece in pieces:
            # If adding this piece would exceed chunk_size
            candidate = (current_chunk + separator + piece) if current_chunk else piece

            if len(candidate) <= chunk_size:
                current_chunk = candidate
            else:
                # Save current chunk if it has content
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())

                # If this single piece is still too large, recurse with next separator
                if len(piece) > chunk_size:
                    remaining_separators = separators[separators.index(separator) + 1:] if separator in separators else [""]
                    if remaining_separators:
                        sub_chunks = self._recursive_split(piece, chunk_size, overlap, remaining_separators)
                        chunks.extend(sub_chunks)
                        current_chunk = ""
                    else:
                        # Absolute last resort: hard split
                        chunks.append(piece[:chunk_size].strip())
                        current_chunk = piece[chunk_size:]
                else:
                    current_chunk = piece

        # Don't forget the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # Add overlap between consecutive chunks
        if overlap > 0 and len(chunks) > 1:
            overlapped = [chunks[0]]
            for i in range(1, len(chunks)):
                prev_tail = chunks[i - 1][-overlap:]
                overlapped.append(prev_tail + " " + chunks[i])
            chunks = overlapped

        return chunks

    def _clean_chunk(self, text: str) -> str:
        """
        Clean a single chunk. Called AFTER structure-aware chunking.
        Normalizes whitespace and fixes encoding — does NOT remove
        structural markers (those were already used for section detection).
        """
        # Normalize multiple newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Normalize multiple spaces
        text = re.sub(r' {2,}', ' ', text)
        # Fix common encoding artifacts
        replacements = {
            'â€™': "'", 'â€˜': "'", 'â€œ': '"', 'â€\x9d': '"',
            'â€"': "—", 'â€"': "–", 'â€¦': "…", 'Â ': " ",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        # Remove isolated page numbers (lines that are just digits)
        text = re.sub(r'^\d{1,3}\s*$', '', text, flags=re.MULTILINE)
        # Strip
        text = text.strip()
        return text

    # ------------------------------------------------------------------
    # Chunking diagnostics
    # ------------------------------------------------------------------

    def chunk_report(self, chunks: list[dict] | None = None) -> str:
        """
        Generate a human-readable report about chunk quality.
        If chunks is None, generates fresh chunks with default settings.

        Returns a formatted string you can print or save.
        """
        if chunks is None:
            chunks = self.prepare_chunks()

        lines = []
        lines.append(f"CHUNK QUALITY REPORT: {Path(self.pdf_path).name}")
        lines.append("=" * 60)

        # Basic stats
        total = len(chunks)
        sizes = [len(c["text_original"]) for c in chunks]
        embed_sizes = [len(c["text_for_embedding"]) for c in chunks]

        lines.append(f"Total chunks: {total}")
        lines.append(f"")
        lines.append(f"Original text sizes:")
        lines.append(f"  Min: {min(sizes)} chars")
        lines.append(f"  Max: {max(sizes)} chars")
        lines.append(f"  Avg: {sum(sizes) // total} chars")
        lines.append(f"  Median: {sorted(sizes)[total // 2]} chars")
        lines.append(f"")
        lines.append(f"Embedding text sizes (with prefix):")
        lines.append(f"  Min: {min(embed_sizes)} chars")
        lines.append(f"  Max: {max(embed_sizes)} chars")
        lines.append(f"  Avg: {sum(embed_sizes) // total} chars")

        # Size distribution
        lines.append(f"")
        lines.append(f"Size distribution (original text):")
        buckets = [
            ("< 100 chars", lambda s: s < 100),
            ("100–256", lambda s: 100 <= s < 256),
            ("256–512", lambda s: 256 <= s < 512),
            ("512–768", lambda s: 512 <= s < 768),
            ("768–1024", lambda s: 768 <= s < 1024),
            ("> 1024 chars", lambda s: s >= 1024),
        ]
        for label, fn in buckets:
            count = sum(1 for s in sizes if fn(s))
            bar = "█" * min(count, 40)
            lines.append(f"  {label:>14}: {count:>4}  {bar}")

        # By section type
        lines.append(f"")
        lines.append(f"Chunks by section type:")
        type_counts = defaultdict(int)
        for c in chunks:
            type_counts[c["metadata"]["section_type"]] += 1
        for t, count in sorted(type_counts.items()):
            lines.append(f"  {t:>12}: {count:>4} chunks")

        # Sample chunks
        lines.append(f"")
        lines.append(f"Sample chunks (first, middle, last):")
        for label, idx in [("First", 0), ("Middle", total // 2), ("Last", -1)]:
            c = chunks[idx]
            lines.append(f"")
            lines.append(f"  --- {label} (chunk {c['metadata']['chunk_index']}) ---")
            lines.append(f"  Type: {c['metadata']['section_type']}")
            lines.append(f"  Label: {c['metadata']['label']}")
            lines.append(f"  Chapter: {c['metadata']['chapter']}")
            lines.append(f"  Part: {c['metadata']['section_part']}/{c['metadata']['section_parts']}")
            lines.append(f"  Size: {len(c['text_original'])} chars")
            lines.append(f"  Embedding text: {c['text_for_embedding'][:200]}...")

        # Quality checks
        lines.append(f"")
        lines.append(f"Quality checks:")
        tiny = sum(1 for s in sizes if s < 50)
        huge = sum(1 for s in sizes if s > 1024)
        empty_meta = sum(1 for c in chunks if not c["metadata"]["label"])
        lines.append(f"  Tiny chunks (< 50 chars): {tiny} {'✓' if tiny < total * 0.05 else '⚠ review these'}")
        lines.append(f"  Oversized chunks (> 1024): {huge} {'✓' if huge < total * 0.1 else '⚠ review these'}")
        lines.append(f"  Missing label metadata: {empty_meta} {'✓' if empty_meta == 0 else '⚠ fix this'}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Context manager & cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._doc.close()

    def __enter__(self) -> "PDFBodyExtractor":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"PDFBodyExtractor("
            f"pdf_path={self.pdf_path!r}, "
            f"pages={self.start_page}-{self.end_page})"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "OJ_L_202401689_EN_TXT.pdf"

    with PDFBodyExtractor(pdf_path) as extractor:
        print(f"Extracting from: {pdf_path}")
        print(f"Pages: {extractor.total_pages}\n")

        # Generate chunks
        chunks = extractor.prepare_chunks(chunk_size=512, chunk_overlap=50)

        # Print report
        print(extractor.chunk_report(chunks))

        # Print a few full samples
        print("\n" + "=" * 60)
        print("FULL CHUNK SAMPLES")
        print("=" * 60)
        for i in [0, 5, len(chunks) // 2, -1]:
            c = chunks[i]
            print(f"\n--- Chunk {c['metadata']['chunk_index']} ---")
            print(f"Metadata: {c['metadata']}")
            print(f"For embedding:\n  {c['text_for_embedding'][:300]}")
            print()