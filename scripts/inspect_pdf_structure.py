"""
PDF Page Structure Inspector
Usage:  python inspect_pdf_structure.py input.pdf [page_number]
        page_number is 0-indexed, defaults to 0
Outputs a readable tree of every block → line → span on the page.
"""

import sys
import json
import fitz  # pip install PyMuPDF


def fmt_bbox(bbox):
    """Compact bbox string."""
    return f"[{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]"


def fmt_color(c):
    """Convert sRGB int to hex."""
    return f"#{c:06X}" if c else "#000000"


def flag_str(flags):
    """Decode the span flags bitmask."""
    parts = []
    if flags & (1 << 0): parts.append("superscript")
    if flags & (1 << 1): parts.append("italic")
    if flags & (1 << 2): parts.append("serif")
    if flags & (1 << 3): parts.append("mono")
    if flags & (1 << 4): parts.append("bold")
    return "+".join(parts) if parts else "regular"


def inspect_page(doc, page_num=0, max_text=80):
    page = doc[page_num]
    data = page.get_text("dict", sort=True)

    print(f"\n{'='*70}")
    print(f"  PAGE {page_num}  |  {page.rect.width:.0f} × {page.rect.height:.0f} pt  |  rotation: {page.rotation}°")
    print(f"{'='*70}\n")

    for bi, block in enumerate(data["blocks"]):
        btype = "IMAGE" if block["type"] == 1 else "TEXT"
        print(f"  BLOCK {bi}  [{btype}]  bbox={fmt_bbox(block['bbox'])}")
        print(f"  {'─'*60}")

        if block["type"] == 1:
            # Image block
            print(f"    image: {block.get('width','?')}×{block.get('height','?')}  "
                  f"bpc={block.get('bpc','?')}  cs={block.get('colorspace','?')}")
            print()
            continue

        for li, line in enumerate(block["lines"]):
            direction = "→" if line["dir"] == (1.0, 0.0) else f"dir{line['dir']}"
            print(f"    LINE {li}  bbox={fmt_bbox(line['bbox'])}  {direction}")

            for si, span in enumerate(line["spans"]):
                text = span["text"]
                preview = (text[:max_text] + "…") if len(text) > max_text else text
                preview = preview.replace("\n", "\\n")

                print(f"      SPAN {si}  \"{preview}\"")
                print(f"             font={span['font']}  size={span['size']:.1f}  "
                      f"color={fmt_color(span['color'])}  flags={flag_str(span['flags'])}")
                print(f"             bbox={fmt_bbox(span['bbox'])}")

            print()
        print()


def save_raw_json(doc, page_num, out_path):
    """Dump the raw dict to JSON for programmatic inspection."""
    page = doc[page_num]
    data = page.get_text("dict", sort=True)
    # Make bbox tuples serializable
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Raw JSON saved to: {out_path}")


# ── main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_pdf_structure.py input.pdf [page_num]")
        sys.exit(1)

    pdf_path = sys.argv[1]
    page_num = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    doc = fitz.open(pdf_path)
    if page_num >= doc.page_count:
        print(f"Error: page {page_num} doesn't exist (doc has {doc.page_count} pages)")
        sys.exit(1)

    inspect_page(doc, page_num)
    save_raw_json(doc, page_num, pdf_path.replace(".pdf", f"_page{page_num}_structure.json"))
    doc.close()