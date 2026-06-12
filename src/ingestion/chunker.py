from pdf_loader import PDFBodyExtractor
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re

def detect_article_context(text, position):
    """Find which article a text position belongs to."""
    # Search backwards from position for nearest Article marker
    preceding = text[:position]
    articles = list(re.finditer(r'Article\s+(\d+)\s*\n\s*(.+?)(?=\n)', preceding))
    chapters = list(re.finditer(r'(CHAPTER\s+[IVXLC]+)\s*\n?\s*(.+?)(?=\n)', preceding))
    
    article = articles[-1] if articles else None
    chapter = chapters[-1] if chapters else None
    
    return {
        "article": f"Article {article.group(1)}" if article else "Preamble",
        "article_title": article.group(2).strip() if article else "",
        "chapter": chapter.group(1).strip() if chapter else "",
        "chapter_title": chapter.group(2).strip() if chapter else "",
    }

# ---- Step 3: Chunk with recursive splitter (vanilla, proven) ----
splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len,
)

raw_chunks = splitter.split_text(full_document_text)

# ---- Enrich each chunk with structural context ----
enriched_chunks = []
for chunk_text in raw_chunks:
    # Find where this chunk appears in the original text
    pos = full_document_text.find(chunk_text[:100])
    context = detect_article_context(full_document_text, pos)
    
    # Prepend article context to the chunk text (for better embeddings)
    prefix = ""
    if context["article"] != "Preamble":
        prefix = f"{context['article']}: {context['article_title']}. "
    
    enriched_chunks.append({
        "text": prefix + chunk_text,          # for embedding
        "original_text": chunk_text,           # for display
        "metadata": {
            **context,
            "source": "eu_ai_act.pdf",
            "chunk_index": len(enriched_chunks),
        }
    })