"""
pdf_body_extractor.py
=====================
A reusable class for extracting the main body text from structured PDFs
that use thin hairline rules to delimit header, footer, and footnote zones.

Layout assumed
--------------
  ┌─────────────────────────────┐
  │  Header area                │  ← above header separator line
  ├─────────────────────────────┤  ← header line  (static y, full-width)
  │                             │
  │  MAIN TEXT BODY             │  ← target extraction zone
  │                             │
  ├─────────────────────────────┤  ← footnote line (dynamic y, short width)
  │  Footnotes                  │
  ├─────────────────────────────┤  ← footer line  (static y, full-width)
  │  Footer area                │
  └─────────────────────────────┘

Quick start
-----------
    from pdf_body_extractor import PDFBodyExtractor

    # Basic usage — entire document, EU OJ defaults
    extractor = PDFBodyExtractor("my_document.pdf")
    for page_num, blocks in extractor.iter_pages():
        for block in blocks:
            print(block["text"])

    # Specific page range
    extractor = PDFBodyExtractor("my_document.pdf", start_page=3, end_page=10)

    # Different PDF structure — override thresholds
    extractor = PDFBodyExtractor(
        "other_doc.pdf",
        hairline_max_height=3.0,
        full_width_threshold=300.0,
        footnote_max_width=150.0,
        body_margin=4.0,
    )

    # Get all pages as flat text
    text = extractor.extract_text()

    # Get per-page structured results
    results = extractor.extract_all()
    # results[0] → {"page_num": 1, "bounds": {...}, "blocks": [...]}
"""

from __future__ import annotations

import fitz  # PyMuPDF
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterator


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class PageBounds:
    """Y-coordinates of the three structural separator lines on a page.
    Any value may be None if that line was not detected."""
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
    x0:       float
    y0:       float
    x1:       float
    y1:       float
    text:     str
    block_no: int

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        return f"TextBlock(y0={self.y0:.1f}, text={preview!r})"


@dataclass
class PageResult:
    """Extraction result for a single page."""
    page_num: int          # 1-based
    bounds:   PageBounds
    blocks:   list[TextBlock] = field(default_factory=list)

    @property
    def text(self) -> str:
        """Plain text of all body blocks joined by newlines."""
        return "\n".join(b.text.strip() for b in self.blocks)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class PDFBodyExtractor:
    """
    Extracts the main body text from structured PDFs using hairline rules
    drawn via PyMuPDF's get_drawings() to identify zone boundaries.

    Parameters
    ----------
    pdf_path : str
        Path to the PDF file.

    start_page : int | None
        First page to process (1-based, inclusive).
        Defaults to 1 (beginning of document).

    end_page : int | None
        Last page to process (1-based, inclusive).
        Defaults to None (end of document).

    hairline_max_height : float
        Maximum rect height (px) to qualify as a separator line.
        Increase if your PDF uses slightly thicker rules.
        Default: 2.0

    hairline_min_width : float
        Minimum rect width (px) to be considered as a separator line.
        Filters out tiny decorative marks or stray path artefacts.
        Default: 10.0

    full_width_threshold : float
        Minimum total merged width (px) for a line to be classified as a
        header or footer separator. Set to roughly 70 % of your page's
        text-column width.
        Default: 400.0

    footnote_max_width : float
        Maximum total merged width (px) for a line to be classified as a
        footnote separator. The footnote rule is intentionally short.
        Default: 100.0

    y_snap_tolerance : float
        Two segments whose y0 values differ by less than this amount (px)
        are considered collinear and merged into one logical line.
        Default: 1.5

    body_margin : float
        Extra pixels of slack added/subtracted at zone boundaries when
        deciding whether a text block falls inside the body region.
        Handles sub-pixel rounding differences between text and drawings.
        Default: 2.0

    text_separator : str
        String used to join body blocks when returning plain text.
        Default: "\\n"
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

        # Open document once; keep it open for the lifetime of the instance
        self._doc: fitz.Document = fitz.open(pdf_path)
        self._total_pages: int   = len(self._doc)

        # Resolve and validate page range (convert to 0-based internally)
        self.start_page = start_page  # public, 1-based
        self.end_page   = end_page    # public, 1-based

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def start_page(self) -> int:
        """First page to process (1-based, inclusive)."""
        return self._start_page_1based

    @start_page.setter
    def start_page(self, value: int | None) -> None:
        resolved = 1 if value is None else int(value)
        if not (1 <= resolved <= self._total_pages):
            raise ValueError(
                f"start_page={resolved} is out of range "
                f"[1, {self._total_pages}]"
            )
        self._start_page_1based = resolved

    @property
    def end_page(self) -> int:
        """Last page to process (1-based, inclusive)."""
        return self._end_page_1based

    @end_page.setter
    def end_page(self, value: int | None) -> None:
        resolved = self._total_pages if value is None else int(value)
        if not (1 <= resolved <= self._total_pages):
            raise ValueError(
                f"end_page={resolved} is out of range "
                f"[1, {self._total_pages}]"
            )
        if resolved < self._start_page_1based:
            raise ValueError(
                f"end_page={resolved} must be >= start_page={self._start_page_1based}"
            )
        self._end_page_1based = resolved

    @property
    def total_pages(self) -> int:
        """Total number of pages in the PDF (read-only)."""
        return self._total_pages

    @property
    def page_range(self) -> range:
        """The active page range as a 1-based range object."""
        return range(self._start_page_1based, self._end_page_1based + 1)

    # ------------------------------------------------------------------
    # Core internal helpers
    # ------------------------------------------------------------------

    def _collect_hairlines(self, drawings: list) -> list:
        """
        Filter get_drawings() output down to thin hairline segments only.
        Returns a list of fitz.Rect objects.
        """
        lines = []
        for d in drawings:
            r = d["rect"]
            # Must be a filled path (type == 'f')
            if d.get("type") != "f":
                continue
            # Must be thin in at least one dimension
            if r.height > self.hairline_max_height and r.width > self.hairline_max_height:
                continue
            # Must be wide enough to matter
            if max(r.width, r.height) < self.hairline_min_width:
                continue
            lines.append(r)
        return lines

    def _merge_collinear(self, segments: list) -> list[tuple[float, float, float]]:
        """
        Merge collinear segments (same y0 within y_snap_tolerance, abutting
        x-ranges) into a sorted list of (y0, x0, x1) tuples.

        Note on ordering
        ----------------
        get_drawings() returns segments in PDF content-stream order, NOT
        visual top-to-bottom order. For example, a footnote rule drawn later
        in the stream may appear after a footer rule even though it sits
        visually above it. This method sorts the merged output by y0 so that
        callers always see lines in top-to-bottom visual order.
        """
        if not segments:
            return []

        groups: dict[float, list] = defaultdict(list)
        for r in segments:
            # Snap y0 to a bucket so near-identical y values collapse
            key = round(r.y0 / self.y_snap_tolerance) * self.y_snap_tolerance
            groups[key].append(r)

        merged = []
        for _, rects in groups.items():
            x0 = min(r.x0 for r in rects)
            x1 = max(r.x1 for r in rects)
            y0 = sum(r.y0 for r in rects) / len(rects)   # average y
            merged.append((y0, x0, x1))

        return sorted(merged)   # top-to-bottom visual order

    def _classify_lines(self, page: fitz.Page) -> PageBounds:
        """
        Analyse a single page's drawings and classify separator lines into
        header, footer, and footnote boundaries.

        Classification rules
        --------------------
        - merged_width >= full_width_threshold AND y0 < page_mid  → header
        - merged_width >= full_width_threshold AND y0 >= page_mid → footer
        - merged_width <= footnote_max_width                       → footnote
        """
        drawings = page.get_drawings()
        segments = self._collect_hairlines(drawings)
        lines    = self._merge_collinear(segments)

        page_mid   = page.rect.height / 2.0
        bounds     = PageBounds()

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
        """
        Extract text blocks that lie within the body zone of the page.

        Body zone
        ---------
          top    = (header_y  or 0)            + body_margin
          bottom = (footnote_y or footer_y or page_height) - body_margin

        Blocks are returned in visual reading order (top-to-bottom,
        left-to-right).
        """
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
            if block_type != 0:       # 0 = text, 1 = image
                continue
            if not text.strip():
                continue
            if by0 >= top and by1 <= bottom:
                body_blocks.append(
                    TextBlock(
                        x0=bx0, y0=by0, x1=bx1, y1=by1,
                        text=text,
                        block_no=block_no,
                    )
                )

        body_blocks.sort(key=lambda b: (b.y0, b.x0))
        return body_blocks

    # ------------------------------------------------------------------
    # Public extraction API
    # ------------------------------------------------------------------

    def classify_page(self, page_num: int) -> PageBounds:
        """
        Return the detected boundary lines for a single page.

        Parameters
        ----------
        page_num : int
            1-based page number.
        """
        self._validate_page_num(page_num)
        return self._classify_lines(self._doc[page_num - 1])

    def extract_page(self, page_num: int) -> PageResult:
        """
        Extract body text blocks from a single page.

        Parameters
        ----------
        page_num : int
            1-based page number.

        Returns
        -------
        PageResult with .page_num, .bounds, .blocks, and .text
        """
        self._validate_page_num(page_num)
        page   = self._doc[page_num - 1]
        bounds = self._classify_lines(page)
        blocks = self._extract_blocks(page, bounds)
        return PageResult(page_num=page_num, bounds=bounds, blocks=blocks)

    def iter_pages(self) -> Iterator[tuple[int, list[TextBlock]]]:
        """
        Lazily iterate over the active page range, yielding
        (page_num, blocks) tuples. Memory-efficient for large PDFs.

        Example
        -------
            for page_num, blocks in extractor.iter_pages():
                for block in blocks:
                    print(block.text)
        """
        for page_num in self.page_range:
            page   = self._doc[page_num - 1]
            bounds = self._classify_lines(page)
            blocks = self._extract_blocks(page, bounds)
            yield page_num, blocks

    def extract_all(self) -> list[PageResult]:
        """
        Extract body text from every page in the active range.

        Returns
        -------
        list[PageResult] — one entry per page, in page order.
        """
        results = []
        for page_num in self.page_range:
            page   = self._doc[page_num - 1]
            bounds = self._classify_lines(page)
            blocks = self._extract_blocks(page, bounds)
            results.append(PageResult(page_num=page_num, bounds=bounds, blocks=blocks))
        return results

    def extract_text(self) -> str:
        """
        Extract all body text across the active page range and return it
        as a single string (pages joined by the text_separator attribute).
        """
        page_texts = []
        for _, blocks in self.iter_pages():
            page_text = self.text_separator.join(b.text.strip() for b in blocks)
            if page_text:
                page_texts.append(page_text)
        return self.text_separator.join(page_texts)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def describe(self) -> str:
        """
        Return a human-readable summary of the extractor's configuration
        and the active page range.
        """
        lines = [
            "PDFBodyExtractor",
            f"  pdf_path             : {self.pdf_path}",
            f"  total_pages          : {self.total_pages}",
            f"  active range         : pages {self.start_page}–{self.end_page} "
            f"({len(self.page_range)} pages)",
            "  ── Thresholds ──",
            f"  hairline_max_height  : {self.hairline_max_height}",
            f"  hairline_min_width   : {self.hairline_min_width}",
            f"  full_width_threshold : {self.full_width_threshold}",
            f"  footnote_max_width   : {self.footnote_max_width}",
            f"  y_snap_tolerance     : {self.y_snap_tolerance}",
            f"  body_margin          : {self.body_margin}",
        ]
        return "\n".join(lines)

    def scan_boundaries(self) -> list[dict]:
        """
        Scan every page in the active range and return a list of dicts
        with per-page boundary information. Useful for auditing whether the
        classifier is behaving as expected across the document.

        Returns
        -------
        list of dicts:
            {
              "page_num"  : int,
              "header_y"  : float | None,
              "footer_y"  : float | None,
              "footnote_y": float | None,
              "n_blocks"  : int,
            }
        """
        rows = []
        for page_num in self.page_range:
            page   = self._doc[page_num - 1]
            bounds = self._classify_lines(page)
            blocks = self._extract_blocks(page, bounds)
            rows.append({
                "page_num":   page_num,
                "header_y":   bounds.header_y,
                "footer_y":   bounds.footer_y,
                "footnote_y": bounds.footnote_y,
                "n_blocks":   len(blocks),
            })
        return rows

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying PDF document."""
        self._doc.close()

    def __enter__(self) -> "PDFBodyExtractor":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"PDFBodyExtractor("
            f"pdf_path={self.pdf_path!r}, "
            f"start_page={self.start_page}, "
            f"end_page={self.end_page})"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_page_num(self, page_num: int) -> None:
        if not (1 <= page_num <= self._total_pages):
            raise ValueError(
                f"page_num={page_num} is out of range [1, {self._total_pages}]"
            )


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    pdf_path   = sys.argv[1] if len(sys.argv) > 1 else "OJ_L_202401689_EN_TXT.pdf"
    start      = int(sys.argv[2]) if len(sys.argv) > 2 else None
    end        = int(sys.argv[3]) if len(sys.argv) > 3 else None
    max_pages  = 3  # cap preview pages for readability

    with PDFBodyExtractor(pdf_path, start_page=start, end_page=end) as extractor:
        print(extractor.describe())
        text = extractor.extract_text()
        print(text[:10000])  # print first 1000 chars of extracted text

        # preview_end = min(
        #     extractor.start_page + max_pages - 1,
        #     extractor.end_page,
        # )
        # preview_extractor = PDFBodyExtractor(
        #     pdf_path,
        #     start_page=extractor.start_page,
        #     end_page=preview_end,
        # )

        # print(f"\n{'='*70}")
        # print(f"Boundary scan (pages {extractor.start_page}–{extractor.end_page})")
        # print(f"{'─'*70}")
        # print(f"{'Page':>5}  {'header_y':>10}  {'footer_y':>10}  {'footnote_y':>10}  {'blocks':>6}")
        # print(f"{'─'*70}")
        # for row in extractor.scan_boundaries():
        #     print(
        #         f"{row['page_num']:>5}  "
        #         f"{str(row['header_y'] or '—'):>10}  "
        #         f"{str(row['footer_y'] or '—'):>10}  "
        #         f"{str(row['footnote_y'] or '—'):>10}  "
        #         f"{row['n_blocks']:>6}"
        #     )

        # print(f"\n{'='*70}")
        # print(f"Text preview — first {max_pages} pages")
        # print(f"{'─'*70}")
        # for result in preview_extractor.extract_all():
        #     print(f"\n── Page {result.page_num} ({len(result.blocks)} blocks) ──")
        #     for blk in result.blocks:
        #         preview = blk.text[:100].replace("\n", " ")
        #         print(f"  [y={blk.y0:.1f}] {preview}")

        # preview_extractor.close()