"""
chunker.py
==========
Plugs downstream of PDFBodyExtractor.
Handles: text → structural detection → recursive chunking → embedding → vector store ready.

Pipeline:
    PDFBodyExtractor (pdf_loader.py)
        ↓ extract_text() / extract_all()
    Chunker (this file)
        ↓ chunk() → enriched chunks with metadata
    Embedder (this file)
        ↓ embed() → vectors
    VectorStore (this file)
        ↓ store() → ChromaDB

Usage:
    from pdf_loader import PDFBodyExtractor
    from chunker import ChunkingPipeline

    extractor = PDFBodyExtractor("eu_ai_act.pdf")
    pipeline = ChunkingPipeline(extractor)

    # One-liner: extract → chunk → embed → store
    pipeline.run()

    # Or step by step:
    chunks = pipeline.chunk()
    chunks_with_embeddings = pipeline.embed(chunks)
    pipeline.store(chunks_with_embeddings)

    # Query
    results = pipeline.query("What are the prohibited AI practices?", top_k=5)
"""

import re
import json
import time
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from pdf_loader import PDFBodyExtractor

# ---------------------------------------------------------------------------
# Configuration — all tunable values in one place
# ---------------------------------------------------------------------------

@dataclass
class ChunkingConfig:
    """All parameters that control chunking, embedding, and storage."""

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 50
    min_chunk_length: int = 30          # skip chunks shorter than this
    separators: tuple = ("\n\n", "\n", ". ", ", ", " ", "")

    # Embedding
    embedding_model: str = "BAAI/bge-small-en-v1.5"  # 384 dims, fast, free
    embedding_batch_size: int = 64

    # Vector store
    chroma_persist_dir: str = "./data/processed/chroma_db"
    collection_name: str = "eu_ai_act"

    # Section detection patterns (customize per document type)
    article_pattern: str = r'\nArticle\s+(\d+)\s*\n\s*(.+?)(?=\n)'
    chapter_pattern: str = r'(CHAPTER\s+[IVXLC]+)\s*\n?\s*(.+?)(?=\n)'
    recital_pattern: str = r'\((\d+)\)\s*\n'
    annex_pattern: str = r'\n(ANNEX\s+[IVXLC]+[A-Za-z\s]*)\s*\n'


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structural detection
# ---------------------------------------------------------------------------

class StructureDetector:
    """
    Detects structural elements (Chapters, Articles, Recitals, Annexes)
    in raw text extracted from a PDF.

    Separated from the chunker so you can swap detection strategies
    without touching the splitting logic.
    """

    def __init__(self, config: ChunkingConfig):
        self.config = config

    def detect_chapters(self, text: str) -> list[dict]:
        """Find all chapter markers with their positions."""
        chapters = []
        for m in re.finditer(self.config.chapter_pattern, text):
            chapters.append({
                "label": m.group(1).strip(),
                "title": m.group(2).strip(),
                "position": m.start(),
            })
        return chapters

    def find_chapter_at(self, position: int, chapters: list[dict]) -> tuple[str, str]:
        """Find which chapter a given text position belongs to."""
        current_label, current_title = "", ""
        for ch in chapters:
            if ch["position"] <= position:
                current_label = ch["label"]
                current_title = ch["title"]
            else:
                break
        return current_label, current_title

    def detect_article_context(
        self, text: str, position: int, chapters: list[dict]
    ) -> dict:
        """
        For a given character position, find the nearest Article and Chapter.
        Used to enrich each chunk with structural metadata.
        """
        preceding = text[:position]
        articles = list(re.finditer(self.config.article_pattern, preceding))
        article = articles[-1] if articles else None
        chapter_label, chapter_title = self.find_chapter_at(position, chapters)

        return {
            "article": f"Article {article.group(1)}" if article else "Preamble",
            "article_title": article.group(2).strip() if article else "",
            "chapter": chapter_label,
            "chapter_title": chapter_title,
        }


# ---------------------------------------------------------------------------
# Recursive text splitter (no LangChain dependency)
# ---------------------------------------------------------------------------

class RecursiveSplitter:
    """
    Recursive character text splitter.

    How it works:
    1. Start with the FULL text
    2. Try splitting on the BEST separator first (paragraph breaks \\n\\n)
    3. If any piece is still too big, recursively try the NEXT separator
    4. Separator priority: \\n\\n → \\n → ". " → ", " → " " → hard character split
    5. After splitting, add overlap between consecutive chunks

    This is equivalent to LangChain's RecursiveCharacterTextSplitter
    but implemented without the dependency.
    """

    def __init__(self, config: ChunkingConfig):
        self.chunk_size = config.chunk_size
        self.chunk_overlap = config.chunk_overlap
        self.min_chunk_length = config.min_chunk_length
        self.separators = list(config.separators)

    def split(self, text: str) -> list[str]:
        """Split text into chunks respecting natural boundaries."""
        if not text or not text.strip():
            return []

        raw_chunks = self._split_recursive(text, self.separators)

        # Add overlap
        if self.chunk_overlap > 0 and len(raw_chunks) > 1:
            raw_chunks = self._add_overlap(raw_chunks)

        # Filter out tiny/empty chunks
        return [c for c in raw_chunks if len(c.strip()) >= self.min_chunk_length]

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        """Core recursive splitting logic."""
        if len(text) <= self.chunk_size:
            return [text.strip()] if text.strip() else []

        # Find the best separator that exists in the text
        separator = ""
        for sep in separators:
            if sep == "":
                separator = sep
                break
            if sep in text:
                separator = sep
                break

        # Split by chosen separator
        if separator:
            pieces = text.split(separator)
        else:
            # Hard character split (absolute last resort)
            return [text[i:i + self.chunk_size]
                    for i in range(0, len(text), self.chunk_size)]

        # Merge pieces into chunks that fit within chunk_size
        chunks = []
        current = ""

        for piece in pieces:
            candidate = (current + separator + piece) if current else piece

            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                # Save the current chunk
                if current.strip():
                    chunks.append(current.strip())

                # If this piece alone exceeds chunk_size, recurse deeper
                if len(piece) > self.chunk_size:
                    remaining_seps = separators[separators.index(separator) + 1:] \
                        if separator in separators else [""]
                    sub_chunks = self._split_recursive(piece, remaining_seps)
                    chunks.extend(sub_chunks)
                    current = ""
                else:
                    current = piece

        if current.strip():
            chunks.append(current.strip())

        return chunks

    def _add_overlap(self, chunks: list[str]) -> list[str]:
        """Prepend the tail of the previous chunk to create overlap."""
        result = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = chunks[i - 1][-self.chunk_overlap:]
            result.append(prev_tail + " " + chunks[i])
        return result


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

class Embedder:
    """
    Generates vector embeddings for text chunks using sentence-transformers.

    Uses BAAI/bge-small-en-v1.5 by default:
    - 384 dimensions (small, fast retrieval)
    - ~130MB download (runs on CPU, free forever)
    - Strong MTEB benchmark performance for retrieval tasks

    The model loads once and stays in memory for the lifetime of the instance.
    """

    def __init__(self, config: ChunkingConfig):
        self.model_name = config.embedding_model
        self.batch_size = config.embedding_batch_size
        self._model = None  # lazy load

    @property
    def model(self):
        """Lazy-load the embedding model on first use."""
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            logger.info(f"Model loaded. Dimension: {self._model.get_sentence_embedding_dimension()}")
        return self._model

    @property
    def dimension(self) -> int:
        """Embedding vector dimension."""
        return self.model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of texts into vectors.

        Parameters
        ----------
        texts : list[str]
            Texts to embed. For best retrieval, these should already
            include the contextual prefix (e.g., "Article 5: ...").

        Returns
        -------
        list[list[float]] — one vector per text, each of length self.dimension
        """
        logger.info(f"Embedding {len(texts)} texts in batches of {self.batch_size}")
        start = time.time()

        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=True,  # L2 normalize for cosine similarity
        )

        elapsed = time.time() - start
        logger.info(f"Embedded {len(texts)} texts in {elapsed:.1f}s "
                     f"({len(texts)/elapsed:.0f} texts/sec)")

        return vectors.tolist()

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a single query.

        Note: BGE models perform better when queries are prefixed with
        "Represent this sentence: " but this depends on the model.
        For bge-small-en-v1.5, the prefix is not strictly required
        but can marginally improve retrieval.
        """
        vector = self.model.encode(
            query,
            normalize_embeddings=True,
        )
        return vector.tolist()


# ---------------------------------------------------------------------------
# Vector Store wrapper
# ---------------------------------------------------------------------------

class VectorStore:
    """
    ChromaDB wrapper for storing and querying embedded chunks.

    Handles:
    - Creating/loading a persistent collection
    - Adding chunks with embeddings and metadata
    - Querying by vector similarity
    - Metadata filtering (e.g., only search within a specific chapter)
    """

    def __init__(self, config: ChunkingConfig):
        self.persist_dir = config.chroma_persist_dir
        self.collection_name = config.collection_name
        self._client = None
        self._collection = None

    @property
    def collection(self):
        """Lazy-load ChromaDB collection."""
        if self._collection is None:
            import chromadb
            logger.info(f"Initializing ChromaDB at: {self.persist_dir}")
            self._client = chromadb.PersistentClient(path=self.persist_dir)
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(f"Collection '{self.collection_name}': "
                         f"{self._collection.count()} existing documents")
        return self._collection

    def add(self, chunks: list[dict]) -> None:
        """
        Add embedded chunks to the vector store.

        Each chunk dict must have:
        - "text_for_embedding": str
        - "text_original": str
        - "embedding": list[float]
        - "metadata": dict
        """
        if not chunks:
            return

        ids = [f"chunk_{c['metadata']['chunk_index']}" for c in chunks]
        documents = [c["text_original"] for c in chunks]
        embeddings = [c["embedding"] for c in chunks]

        # ChromaDB metadata must be flat (str, int, float, bool)
        metadatas = []
        for c in chunks:
            meta = {}
            for k, v in c["metadata"].items():
                if isinstance(v, (str, int, float, bool)):
                    meta[k] = v
                else:
                    meta[k] = str(v)
            meta["text_for_embedding"] = c["text_for_embedding"]
            metadatas.append(meta)

        # ChromaDB has a batch limit — add in chunks of 5000
        batch = 5000
        for i in range(0, len(ids), batch):
            self.collection.add(
                ids=ids[i:i + batch],
                documents=documents[i:i + batch],
                embeddings=embeddings[i:i + batch],
                metadatas=metadatas[i:i + batch],
            )

        logger.info(f"Added {len(chunks)} chunks to collection "
                     f"'{self.collection_name}' (total: {self.collection.count()})")

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """
        Query the vector store.

        Parameters
        ----------
        query_embedding : list[float]
            The query vector.
        top_k : int
            Number of results to return.
        where : dict | None
            Optional metadata filter.
            Example: {"chapter": "CHAPTER II"} to search only Chapter II.

        Returns
        -------
        list[dict] with keys: text, metadata, score, text_for_embedding
        """
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)

        output = []
        for i in range(len(results["ids"][0])):
            meta = results["metadatas"][0][i]
            output.append({
                "text": results["documents"][0][i],
                "text_for_embedding": meta.pop("text_for_embedding", ""),
                "metadata": meta,
                "score": 1 - results["distances"][0][i],  # cosine distance → similarity
            })

        return output

    def clear(self) -> None:
        """Delete all documents from the collection."""
        if self._client and self._collection:
            self._client.delete_collection(self.collection_name)
            self._collection = None
            logger.info(f"Cleared collection '{self.collection_name}'")


# ---------------------------------------------------------------------------
# Text cleaner
# ---------------------------------------------------------------------------

def clean_chunk(text: str) -> str:
    """
    Clean a single chunk after splitting.
    Normalizes whitespace and fixes encoding artifacts.
    """
    # Normalize excessive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Normalize multiple spaces
    text = re.sub(r' {2,}', ' ', text)
    # Fix common encoding artifacts
    for old, new in {
        'â€™': "'", 'â€˜': "'", 'â€œ': '"', 'â€\x9d': '"',
        'â€"': "—", 'â€"': "–", 'â€¦': "…", 'Â ': " ",
    }.items():
        text = text.replace(old, new)
    # Remove isolated page numbers
    text = re.sub(r'^\d{1,3}\s*$', '', text, flags=re.MULTILINE)
    return text.strip()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

class ChunkingPipeline:
    """
    Complete pipeline: PDF extraction → chunking → embedding → vector store.

    Usage:
        extractor = PDFBodyExtractor("eu_ai_act.pdf")
        pipeline = ChunkingPipeline(extractor)

        # Full pipeline
        pipeline.run()

        # Query
        results = pipeline.query("What are the prohibited AI practices?")
        for r in results:
            print(f"[{r['score']:.3f}] {r['metadata']['article']}")
            print(f"  {r['text'][:200]}")
    """

    def __init__(
        self,
        extractor: PDFBodyExtractor,
        config: Optional[ChunkingConfig] = None,
    ):
        self.extractor = extractor
        self.config = config or ChunkingConfig()
        self.source = Path(extractor.pdf_path).name

        # Initialize components
        self.detector = StructureDetector(self.config)
        self.splitter = RecursiveSplitter(self.config)
        self.embedder = Embedder(self.config)
        self.store = VectorStore(self.config)

        # Cache
        self._full_text: Optional[str] = None
        self._chapters: Optional[list] = None

    # ------------------------------------------------------------------
    # Step 1: Extract full text from PDF (via extractor)
    # ------------------------------------------------------------------

    @property
    def full_text(self) -> str:
        """Cached full document text from the extractor."""
        if self._full_text is None:
            logger.info("Extracting text from PDF...")
            pages = self.extractor.extract_all()
            self._full_text = "\n".join(p.text for p in pages if p.text.strip())
            logger.info(f"Extracted {len(self._full_text):,} chars from "
                         f"{len(pages)} pages")
        return self._full_text

    @property
    def chapters(self) -> list[dict]:
        """Cached chapter markers."""
        if self._chapters is None:
            self._chapters = self.detector.detect_chapters(self.full_text)
            logger.info(f"Detected {len(self._chapters)} chapters")
        return self._chapters

    # ------------------------------------------------------------------
    # Step 2: Chunk with structural enrichment
    # ------------------------------------------------------------------

    def chunk(self) -> list[dict]:
        """
        Split the document into enriched chunks with structural metadata.

        Returns
        -------
        list[dict], each containing:
            text_for_embedding : str  — contextual prefix + chunk text
            text_original      : str  — clean chunk text (for display)
            metadata           : dict — article, chapter, page, etc.
        """
        text = self.full_text

        logger.info(f"Splitting {len(text):,} chars with recursive splitter "
                     f"(size={self.config.chunk_size}, overlap={self.config.chunk_overlap})")

        raw_chunks = self.splitter.split(text)
        logger.info(f"Produced {len(raw_chunks)} raw chunks")

        # Enrich each chunk with structural context
        enriched = []
        for chunk_text in raw_chunks:
            # Find position in original text
            search_key = chunk_text[:100]
            pos = text.find(search_key)
            if pos == -1:
                # Overlap prefix may have shifted the text — try without first 50 chars
                pos = text.find(chunk_text[self.config.chunk_overlap + 1:self.config.chunk_overlap + 101])

            # Detect structural context
            context = self.detector.detect_article_context(text, max(pos, 0), self.chapters)

            # Clean the chunk
            cleaned = clean_chunk(chunk_text)
            if len(cleaned) < self.config.min_chunk_length:
                continue

            # Build contextual prefix for embedding
            prefix = ""
            if context["article"] != "Preamble" and context["article_title"]:
                prefix = f"{context['article']}: {context['article_title']}. "
            elif context["article"] == "Preamble":
                prefix = "Preamble. "

            enriched.append({
                "text_for_embedding": prefix + cleaned,
                "text_original": cleaned,
                "metadata": {
                    "source": self.source,
                    "article": context["article"],
                    "article_title": context["article_title"],
                    "chapter": context["chapter"],
                    "chapter_title": context["chapter_title"],
                    "chunk_index": len(enriched),
                },
            })

        logger.info(f"Enriched {len(enriched)} chunks with structural metadata")
        return enriched

    # ------------------------------------------------------------------
    # Step 3: Embed
    # ------------------------------------------------------------------

    def embed(self, chunks: list[dict]) -> list[dict]:
        """
        Generate embeddings for all chunks.
        Adds an "embedding" key to each chunk dict (in-place + returns).

        Parameters
        ----------
        chunks : list[dict]
            Output from self.chunk(). Each must have "text_for_embedding".

        Returns
        -------
        Same list with "embedding" added to each dict.
        """
        texts = [c["text_for_embedding"] for c in chunks]
        vectors = self.embedder.embed_texts(texts)

        for chunk, vector in zip(chunks, vectors):
            chunk["embedding"] = vector

        logger.info(f"Attached {len(vectors)} embeddings "
                     f"(dim={len(vectors[0]) if vectors else 0})")
        return chunks

    # ------------------------------------------------------------------
    # Step 4: Store
    # ------------------------------------------------------------------

    def store_chunks(self, chunks: list[dict]) -> None:
        """
        Store embedded chunks in ChromaDB.

        Parameters
        ----------
        chunks : list[dict]
            Output from self.embed(). Each must have "embedding".
        """
        self.store.add(chunks)

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run(self) -> list[dict]:
        """
        Execute the full pipeline: extract → chunk → embed → store.

        Returns the enriched chunks (with embeddings) for inspection.
        """
        logger.info("=" * 50)
        logger.info("STARTING FULL PIPELINE")
        logger.info("=" * 50)

        start = time.time()

        chunks = self.chunk()
        chunks = self.embed(chunks)
        self.store_chunks(chunks)

        elapsed = time.time() - start
        logger.info(f"Pipeline complete in {elapsed:.1f}s: "
                     f"{len(chunks)} chunks embedded and stored")
        return chunks

    # ------------------------------------------------------------------
    # Query interface
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        top_k: int = 5,
        chapter_filter: Optional[str] = None,
    ) -> list[dict]:
        """
        Query the vector store with a natural language question.

        Parameters
        ----------
        question : str
            The user's question.
        top_k : int
            Number of results.
        chapter_filter : str | None
            Optional: restrict search to a specific chapter.
            Example: "CHAPTER II"

        Returns
        -------
        list[dict] with: text, metadata, score, text_for_embedding
        """
        query_vector = self.embedder.embed_query(question)

        where = {"chapter": chapter_filter} if chapter_filter else None
        results = self.store.query(query_vector, top_k=top_k, where=where)

        return results

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def chunk_report(self, chunks: Optional[list[dict]] = None) -> str:
        """Generate a human-readable quality report on chunks."""
        if chunks is None:
            chunks = self.chunk()

        lines = []
        lines.append(f"CHUNK REPORT: {self.source}")
        lines.append("=" * 50)
        lines.append(f"Total chunks: {len(chunks)}")

        sizes = [len(c["text_original"]) for c in chunks]
        lines.append(f"Size — min: {min(sizes)}, max: {max(sizes)}, "
                      f"avg: {sum(sizes)//len(sizes)}, "
                      f"median: {sorted(sizes)[len(sizes)//2]}")

        # By article
        from collections import Counter
        articles = Counter(c["metadata"]["article"] for c in chunks)
        lines.append(f"\nUnique articles/sections: {len(articles)}")
        lines.append(f"Top 5 by chunk count:")
        for article, count in articles.most_common(5):
            lines.append(f"  {article}: {count} chunks")

        # Quality
        tiny = sum(1 for s in sizes if s < 50)
        lines.append(f"\nQuality: tiny(<50)={tiny}, "
                      f"oversized(>1024)={sum(1 for s in sizes if s > 1024)}")

        # Samples
        lines.append(f"\n--- Sample chunk (Article 5, first part) ---")
        art5 = [c for c in chunks if c["metadata"]["article"] == "Article 5"]
        if art5:
            lines.append(f"Embedding text: {art5[0]['text_for_embedding'][:250]}...")
            lines.append(f"Metadata: {art5[0]['metadata']}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import os

    # Get the project folder root and then the PDF path relative to it
    project_root = Path(__file__).parent.parent
    pdf_path = os.path.join(project_root, "data", "raw", "OJ_L_202401689_EN_TXT.pdf")

    pdf_path = sys.argv[1] if len(sys.argv) > 1 else pdf_path
    action = sys.argv[2] if len(sys.argv) > 2 else "report"

    extractor = PDFBodyExtractor(pdf_path)
    pipeline = ChunkingPipeline(extractor)

    if action == "report":
        # Just chunk and report — no embedding needed
        chunks = pipeline.chunk()
        print(pipeline.chunk_report(chunks))

    elif action == "run":
        # Full pipeline: chunk + embed + store
        chunks = pipeline.run()
        print(pipeline.chunk_report(chunks))
        print(f"\nStored in: {pipeline.config.chroma_persist_dir}")

    elif action == "query":
        query_text = " ".join(sys.argv[3:]) if len(sys.argv) > 3 else "What are prohibited AI practices?"
        results = pipeline.query(query_text)
        print(f"\nQuery: {query_text}\n")
        for i, r in enumerate(results):
            print(f"  [{i+1}] Score: {r['score']:.4f}")
            print(f"      {r['metadata'].get('article', '')} — {r['metadata'].get('chapter', '')}")
            print(f"      {r['text'][:200]}...")
            print()

    extractor.close()