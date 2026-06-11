import re
import fitz  # PyMuPDF
from dataclasses import dataclass


@dataclass
class PDFPage:
    text: str
    page_number: int
    source: str

def load_pdf(pdf_path: str) -> list[PDFPage]:
    """
    Load a PDF and return a list of PDFPage objects.
    Pages with fewer than 20 characters are skipped.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of PDFPage objects.
    """
    pages = []

    with fitz.open(pdf_path) as doc:
        for page_index in range(len(doc)):
            page = doc[page_index]

            try:
                text = page.get_text("text")
            except Exception:
                try:
                    text = page.get_text("rawtext")
                except Exception:
                    continue

            # Handle encoding errors by encoding/decoding with replacement
            try:
                text = text.encode("utf-8", errors="replace").decode("utf-8")
            except Exception:
                continue

            if len(text.strip()) < 20:
                continue

            pages.append(
                PDFPage(
                    text=text,
                    page_number=page_index + 1,
                    source=pdf_path,
                )
            )
    print(f"Loaded {len(pages)} pages from {pdf_path}")
    return pages