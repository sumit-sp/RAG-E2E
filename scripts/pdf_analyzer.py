"""
Universal PDF Analyzer for RAG Projects
========================================
Run on ANY PDF to understand its structure before building your chunking strategy.

Usage:
    python pdf_analyzer.py <path_to_pdf>
    python pdf_analyzer.py <path_to_pdf> --output report.txt

What it tells you:
    1. Basic stats (pages, characters, words)
    2. Font hierarchy (what fonts/sizes are used — reveals headings vs body)
    3. Table of Contents (if the PDF has one built-in)
    4. Structural patterns detected (chapters, sections, articles, numbered items)
    5. Content distribution across pages
    6. Section size analysis (how big are the natural sections)
    7. Chunking recommendation based on the analysis

Requirements:
    pip install pymupdf
"""

import fitz  # PyMuPDF
import re
import sys
import json
from collections import Counter, defaultdict
from pathlib import Path


class PDFAnalyzer:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)
        self.filename = Path(pdf_path).name
        self.full_text = ""
        self.pages = []
        self.font_data = []
        self.sections = []

        # Extract everything upfront
        self._extract_pages()
        self._extract_font_data()

    def _extract_pages(self):
        """Extract text from all pages."""
        for i, page in enumerate(self.doc):
            text = page.get_text()
            self.pages.append({
                "page_number": i + 1,
                "text": text,
                "char_count": len(text),
                "word_count": len(text.split()),
            })
            self.full_text += text + "\n"

    def _extract_font_data(self):
        """Extract font size and style information from all pages."""
        for page_num, page in enumerate(self.doc):
            try:
                page_dict = page.get_text("dict")
                for block in page_dict.get("blocks", []):
                    if block.get("type", 0) != 0:
                        continue
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text = span.get("text", "").strip()
                            if text:
                                self.font_data.append({
                                    "text": text,
                                    "font_size": round(span.get("size", 0), 1),
                                    "font_name": span.get("font", ""),
                                    "is_bold": "Bold" in span.get("font", "") or "bold" in span.get("font", ""),
                                    "is_italic": "Italic" in span.get("font", "") or "italic" in span.get("font", ""),
                                    "page": page_num + 1,
                                })
            except Exception:
                pass  # Some pages may not support dict extraction

    # ==================================================
    # ANALYSIS 1: Basic Document Stats
    # ==================================================
    def analyze_basics(self):
        """Basic document statistics."""
        total_chars = sum(p["char_count"] for p in self.pages)
        total_words = sum(p["word_count"] for p in self.pages)
        page_chars = [p["char_count"] for p in self.pages]

        return {
            "filename": self.filename,
            "pages": len(self.pages),
            "total_characters": total_chars,
            "total_words": total_words,
            "avg_chars_per_page": total_chars // len(self.pages) if self.pages else 0,
            "min_page_chars": min(page_chars) if page_chars else 0,
            "max_page_chars": max(page_chars) if page_chars else 0,
            "blank_pages": sum(1 for p in self.pages if p["char_count"] < 50),
        }

    # ==================================================
    # ANALYSIS 2: Font Hierarchy
    # ==================================================
    def analyze_fonts(self):
        """Analyze font sizes to detect heading hierarchy."""
        if not self.font_data:
            return {"error": "No font data extracted"}

        # Count font sizes
        size_counter = Counter()
        size_samples = defaultdict(list)
        bold_counter = Counter()

        for fd in self.font_data:
            size = fd["font_size"]
            size_counter[size] += 1
            if len(size_samples[size]) < 5:
                size_samples[size].append(fd["text"][:80])
            if fd["is_bold"]:
                bold_counter[size] += 1

        # Find body text size (most common)
        body_size = size_counter.most_common(1)[0][0] if size_counter else 0

        # Classify each size
        font_hierarchy = []
        for size in sorted(size_counter.keys(), reverse=True):
            count = size_counter[size]
            bold_pct = (bold_counter.get(size, 0) / count * 100) if count > 0 else 0
            role = "body text"
            if size > body_size * 1.3:
                role = "major heading"
            elif size > body_size * 1.1:
                role = "minor heading"
            elif size < body_size * 0.85:
                role = "footnote/small text"

            font_hierarchy.append({
                "size_pt": size,
                "count": count,
                "bold_pct": round(bold_pct, 1),
                "likely_role": role,
                "samples": size_samples[size],
            })

        return {
            "body_font_size": body_size,
            "total_font_sizes": len(size_counter),
            "hierarchy": font_hierarchy,
        }

    # ==================================================
    # ANALYSIS 3: Table of Contents
    # ==================================================
    def analyze_toc(self):
        """Extract built-in table of contents."""
        toc = self.doc.get_toc()
        if not toc:
            return {"has_toc": False, "entries": 0, "items": []}

        items = []
        for level, title, page in toc:
            items.append({
                "level": level,
                "title": title[:100],
                "page": page,
            })

        max_level = max(entry[0] for entry in toc) if toc else 0

        return {
            "has_toc": True,
            "entries": len(toc),
            "max_depth": max_level,
            "items": items,
        }

    # ==================================================
    # ANALYSIS 4: Structural Patterns
    # ==================================================
    def analyze_structure(self):
        """Detect structural patterns using regex."""
        patterns = {
            "Chapter/Part": r'(?:CHAPTER|PART|Chapter|Part)\s+[IVXLC\d]+',
            "Section": r'(?:SECTION|Section)\s+\d+',
            "Article": r'Article\s+\d+',
            "Numbered paragraph": r'^\(\d+\)',
            "Lettered list": r'^\([a-z]\)',
            "Numbered heading (1., 2.)": r'^\d+\.\s+[A-Z]',
            "Heading (ALL CAPS line)": r'^[A-Z][A-Z\s]{10,}$',
            "Annex/Appendix": r'(?:ANNEX|APPENDIX|Annex|Appendix)\s+[IVXLC\d]+',
        }

        results = {}
        for name, pattern in patterns.items():
            matches = re.findall(pattern, self.full_text, re.MULTILINE)
            if matches:
                unique_matches = sorted(set(m.strip() for m in matches))
                results[name] = {
                    "count": len(matches),
                    "unique": len(unique_matches),
                    "samples": unique_matches[:15],
                }

        return results

    # ==================================================
    # ANALYSIS 5: Section Size Analysis
    # ==================================================
    def analyze_section_sizes(self):
        """Split by the most granular structural pattern and analyze sizes."""
        # Try to find the best splitting pattern
        # Priority: Article > Section > Chapter > Numbered heading > Paragraph breaks

        split_patterns = [
            ("Article", r'(?=\nArticle\s+\d+[\n\s])'),
            ("Section", r'(?=\nSECTION\s+\d+[\n\s])'),
            ("Chapter", r'(?=\nCHAPTER\s+[IVXLC\d]+[\n\s])'),
            ("Numbered heading", r'(?=\n\d+\.\s+[A-Z])'),
            ("Double newline", r'\n\n+'),
        ]

        best_split = None
        for name, pattern in split_patterns:
            parts = re.split(pattern, self.full_text)
            # Filter out tiny fragments
            meaningful = [p for p in parts if len(p.strip()) > 50]
            if len(meaningful) >= 5:
                best_split = {
                    "pattern_name": name,
                    "pattern": pattern,
                    "sections": meaningful,
                }
                break

        if not best_split:
            return {"error": "No clear structural pattern found"}

        sections = best_split["sections"]
        sizes = [len(s) for s in sections]

        # Size distribution
        buckets = {
            "< 256 chars": sum(1 for s in sizes if s < 256),
            "256–512": sum(1 for s in sizes if 256 <= s < 512),
            "512–1024": sum(1 for s in sizes if 512 <= s < 1024),
            "1024–2048": sum(1 for s in sizes if 1024 <= s < 2048),
            "2048–4096": sum(1 for s in sizes if 2048 <= s < 4096),
            "> 4096 chars": sum(1 for s in sizes if s >= 4096),
        }

        # Find section identifiers
        section_details = []
        for section in sections:
            # Try to extract a title/identifier from the first line
            first_line = section.strip().split("\n")[0][:100]
            section_details.append({
                "identifier": first_line,
                "char_count": len(section),
            })

        # Sort by size for extremes
        sorted_sections = sorted(section_details, key=lambda x: x["char_count"])

        return {
            "split_by": best_split["pattern_name"],
            "total_sections": len(sections),
            "size_stats": {
                "min": min(sizes),
                "max": max(sizes),
                "avg": sum(sizes) // len(sizes),
                "median": sorted(sizes)[len(sizes) // 2],
            },
            "size_distribution": buckets,
            "smallest_5": sorted_sections[:5],
            "largest_5": sorted_sections[-5:][::-1],
        }

    # ==================================================
    # ANALYSIS 6: Chunking Recommendation
    # ==================================================
    def recommend_chunking(self):
        """Based on all analysis, recommend a chunking strategy."""
        structure = self.analyze_structure()
        section_analysis = self.analyze_section_sizes()
        font_analysis = self.analyze_fonts()
        toc = self.analyze_toc()

        recommendations = []

        # Check if document has clear article/section structure
        has_articles = "Article" in structure
        has_sections = "Section" in structure
        has_chapters = "Chapter/Part" in structure
        has_rich_toc = toc.get("entries", 0) > 20

        # Check font variety (indicates heading hierarchy)
        font_sizes_count = font_analysis.get("total_font_sizes", 0) if isinstance(font_analysis, dict) else 0
        has_font_hierarchy = font_sizes_count >= 3

        # Section size stats
        median_section = section_analysis.get("size_stats", {}).get("median", 0) if isinstance(section_analysis, dict) else 0
        max_section = section_analysis.get("size_stats", {}).get("max", 0) if isinstance(section_analysis, dict) else 0

        # Generate recommendations
        recommendations.append("=" * 60)
        recommendations.append("CHUNKING RECOMMENDATION")
        recommendations.append("=" * 60)

        if has_articles:
            article_count = structure["Article"]["count"]
            recommendations.append(f"\n✓ STRONG STRUCTURE DETECTED: {article_count} Articles found")
            recommendations.append("  → Primary strategy: Recursive chunking (512 tokens, 50 overlap)")
            recommendations.append("  → Enhancement: Detect Article boundaries via regex, attach as metadata")
            recommendations.append("  → Enhancement: Prepend 'Article N: Title' to each chunk before embedding")
            recommendations.append("  → Small articles (< 512 chars): keep whole as single chunks")
            recommendations.append("  → Large articles (> 1024 chars): recursive split within article boundary")
        elif has_rich_toc:
            recommendations.append(f"\n✓ RICH TABLE OF CONTENTS: {toc['entries']} entries at {toc['max_depth']} levels")
            recommendations.append("  → Primary strategy: ToC-based chunking using PDF's built-in structure")
            recommendations.append("  → Extract text between ToC entries as natural sections")
            recommendations.append("  → Split oversized sections with recursive character splitter")
        elif has_font_hierarchy:
            recommendations.append(f"\n✓ FONT HIERARCHY DETECTED: {font_sizes_count} distinct font sizes")
            recommendations.append("  → Primary strategy: Font-based section detection")
            recommendations.append("  → Use font size changes to identify heading boundaries")
            recommendations.append("  → Chunk at heading boundaries, split large sections recursively")
        else:
            recommendations.append("\n⚠ NO CLEAR STRUCTURE DETECTED")
            recommendations.append("  → Primary strategy: Recursive chunking (512 tokens, 50 overlap)")
            recommendations.append("  → This is the benchmark-proven default for unstructured documents")
            recommendations.append("  → Consider semantic chunking if you need better boundary detection")

        # Universal recommendations
        recommendations.append("\nUNIVERSAL RECOMMENDATIONS:")
        recommendations.append(f"  → Chunk size: 512 tokens (covers {sum(1 for s in section_analysis.get('size_distribution', {}).values() if True)} sections)")

        if max_section > 10000:
            recommendations.append(f"  → WARNING: Largest section is {max_section:,} chars — will need sub-splitting")

        recommendations.append("  → Always add overlap (50 tokens) to avoid mid-concept cuts")
        recommendations.append("  → Attach metadata: source file, page number, section identifier")
        recommendations.append("  → Prepend section title to chunk text before embedding")
        recommendations.append("  → Don't over-optimize chunking — invest time in hybrid retrieval + re-ranking instead")

        return "\n".join(recommendations)

    # ==================================================
    # FULL REPORT
    # ==================================================
    def generate_report(self) -> str:
        """Generate the complete analysis report."""
        lines = []

        def section(title):
            lines.append("")
            lines.append("=" * 60)
            lines.append(title)
            lines.append("=" * 60)

        lines.append(f"PDF ANALYSIS REPORT: {self.filename}")
        lines.append(f"{'=' * 60}")

        # 1. Basics
        section("1. BASIC DOCUMENT STATS")
        basics = self.analyze_basics()
        for key, val in basics.items():
            lines.append(f"  {key}: {val:,}" if isinstance(val, int) else f"  {key}: {val}")

        # 2. Font Hierarchy
        section("2. FONT HIERARCHY")
        fonts = self.analyze_fonts()
        if "error" not in fonts:
            lines.append(f"  Body text font size: {fonts['body_font_size']}pt")
            lines.append(f"  Distinct font sizes: {fonts['total_font_sizes']}")
            lines.append("")
            lines.append(f"  {'Size':>8} {'Count':>7} {'Bold%':>7} {'Role':<20} Samples")
            lines.append(f"  {'-'*8} {'-'*7} {'-'*7} {'-'*20} {'-'*30}")
            for f in fonts["hierarchy"]:
                samples = f["samples"][0][:40] if f["samples"] else ""
                lines.append(f"  {f['size_pt']:>7.1f}pt {f['count']:>7,} {f['bold_pct']:>6.1f}% {f['likely_role']:<20} {samples}")

        # 3. Table of Contents
        section("3. TABLE OF CONTENTS")
        toc = self.analyze_toc()
        if toc["has_toc"]:
            lines.append(f"  Entries: {toc['entries']}")
            lines.append(f"  Max depth: {toc['max_depth']}")
            lines.append("")
            for item in toc["items"][:25]:
                indent = "  " * item["level"]
                lines.append(f"  {indent}[L{item['level']}] {item['title']} (p.{item['page']})")
            if toc["entries"] > 25:
                lines.append(f"  ... and {toc['entries'] - 25} more")
        else:
            lines.append("  No built-in Table of Contents found.")

        # 4. Structural Patterns
        section("4. STRUCTURAL PATTERNS DETECTED")
        structure = self.analyze_structure()
        if structure:
            for name, data in structure.items():
                lines.append(f"\n  {name}: {data['count']} occurrences ({data['unique']} unique)")
                for s in data["samples"][:8]:
                    lines.append(f"    → {s}")
                if len(data["samples"]) > 8:
                    lines.append(f"    ... and {data['unique'] - 8} more")
        else:
            lines.append("  No clear structural patterns found.")

        # 5. Section Size Analysis
        section("5. SECTION SIZE ANALYSIS")
        sections = self.analyze_section_sizes()
        if "error" not in sections:
            lines.append(f"  Split by: {sections['split_by']}")
            lines.append(f"  Total sections: {sections['total_sections']}")
            lines.append("")
            stats = sections["size_stats"]
            lines.append(f"  Min: {stats['min']:,} chars")
            lines.append(f"  Max: {stats['max']:,} chars")
            lines.append(f"  Avg: {stats['avg']:,} chars")
            lines.append(f"  Median: {stats['median']:,} chars")
            lines.append("")
            lines.append("  Size distribution:")
            for bucket, count in sections["size_distribution"].items():
                bar = "█" * min(count, 40)
                lines.append(f"    {bucket:>16}: {count:>4} {bar}")
            lines.append("")
            lines.append("  Smallest sections:")
            for s in sections["smallest_5"]:
                lines.append(f"    {s['char_count']:>6} chars — {s['identifier'][:70]}")
            lines.append("")
            lines.append("  Largest sections:")
            for s in sections["largest_5"]:
                lines.append(f"    {s['char_count']:>6} chars — {s['identifier'][:70]}")

        # 6. Chunking Recommendation
        lines.append("")
        lines.append(self.recommend_chunking())

        return "\n".join(lines)


# ==================================================
# CLI ENTRY POINT
# ==================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pdf_analyzer.py <path_to_pdf> [--output report.txt]")
        sys.exit(1)

    pdf_path = sys.argv[1]

    if not Path(pdf_path).exists():
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)

    print(f"Analyzing: {pdf_path}")
    print("This may take a moment for large PDFs...\n")

    analyzer = PDFAnalyzer(pdf_path)
    report = analyzer.generate_report()

    # Print to terminal
    print(report)

    # Save to file if --output specified
    if "--output" in sys.argv:
        output_idx = sys.argv.index("--output")
        if output_idx + 1 < len(sys.argv):
            output_path = sys.argv[output_idx + 1]
            with open(output_path, "w") as f:
                f.write(report)
            print(f"\nReport saved to: {output_path}")