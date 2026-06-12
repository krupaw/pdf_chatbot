"""
RAG Engine
----------
Handles the full pipeline:
  1. PDF text extraction  (PyPDF)
  2. Chunking             (sliding window, word-based)
  3. Embeddings           (sentence-transformers all-MiniLM-L6-v2)
  4. Vector store         (FAISS)
  5. QA                   (deepset/roberta-base-squad2)
"""

from __future__ import annotations

import re
import textwrap
from typing import Any

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import pipeline


# ── Constants ─────────────────────────────────────────────────────────────────
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
QA_MODEL    = "deepset/roberta-base-squad2"
CHUNK_SIZE  = 200       # words per chunk
CHUNK_OVERLAP = 40      # words overlap between consecutive chunks
TOP_K       = 5         # chunks to retrieve per query


class RAGEngine:
    """End-to-end Retrieval-Augmented Generation over a single PDF document."""

    def __init__(self) -> None:
        self._embedder: SentenceTransformer | None = None
        self._qa: Any | None = None
        self.chunks: list[str] = []
        self.index: faiss.IndexFlatL2 | None = None
        self.doc_name: str = ""
        self.pages: int = 0

    # ── Lazy model loading ────────────────────────────────────────────────────
    @property
    def embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            self._embedder = SentenceTransformer(EMBED_MODEL)
        return self._embedder

    @property
    def qa(self) -> Any:
        if self._qa is None:
            self._qa = pipeline(
                "question-answering",
                model=QA_MODEL,
                tokenizer=QA_MODEL,
            )
        return self._qa

    # ── Step 1 : PDF extraction ───────────────────────────────────────────────
    def load_pdf(self, file_obj) -> dict:
        """Extract raw text from a PDF file object. Returns document metadata."""
        reader = PdfReader(file_obj)
        self.pages = len(reader.pages)
        self.doc_name = getattr(file_obj, "name", "document.pdf")

        raw_pages: list[str] = []
        for page in reader.pages:
            text = page.extract_text() or ""
            text = self._clean(text)
            if text.strip():
                raw_pages.append(text)

        self._raw_text = "\n\n".join(raw_pages)
        return {
            "name": self.doc_name,
            "pages": self.pages,
            "words": len(self._raw_text.split()),
            "chunks": 0,   # updated after build_index()
        }

    @staticmethod
    def _clean(text: str) -> str:
        """Light cleanup: collapse whitespace, remove lone page numbers."""
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\n\s*\d+\s*\n", "\n", text)
        return text.strip()

    # ── Step 2 : Chunking ─────────────────────────────────────────────────────
    def _chunk(self, text: str) -> list[str]:
        """Sliding-window word-level chunking with overlap."""
        words = text.split()
        chunks: list[str] = []
        step = CHUNK_SIZE - CHUNK_OVERLAP
        for i in range(0, len(words), step):
            chunk = " ".join(words[i : i + CHUNK_SIZE])
            if len(chunk.strip()) > 60:   # skip tiny trailing fragments
                chunks.append(chunk)
        return chunks

    # ── Step 3 + 4 : Embed and index ─────────────────────────────────────────
    def build_index(self) -> None:
        """Chunk text, embed with MiniLM, store in FAISS."""
        self.chunks = self._chunk(self._raw_text)

        # Embed all chunks at once (batched internally by sentence-transformers)
        embeddings: np.ndarray = self.embedder.encode(
            self.chunks,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,   # cosine via dot-product after norm
        ).astype("float32")

        dim = embeddings.shape[1]
        # IndexFlatIP = inner product (≡ cosine similarity on normalized vecs)
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

    # ── Step 5 : Retrieve + Answer ────────────────────────────────────────────
    def retrieve(self, query: str, k: int = TOP_K) -> list[str]:
        """Return the top-k most relevant chunks for a query."""
        if self.index is None or not self.chunks:
            return []
        q_emb = self.embedder.encode(
            [query], normalize_embeddings=True
        ).astype("float32")
        scores, indices = self.index.search(q_emb, k)
        return [self.chunks[i] for i in indices[0] if i < len(self.chunks)]

    def answer(self, query: str, chat_history: list[dict] | None = None) -> dict:
        """
        Full RAG pipeline for one user question.

        Returns a dict with:
          - answer  : str
          - chunks  : list[str]   (retrieved context)
          - score   : float       (QA model confidence)
        """
        # Optionally enrich the query with the last assistant turn for follow-ups
        contextual_query = query
        if chat_history:
            last_turns = [t for t in chat_history[-4:] if t["role"] == "assistant"]
            if last_turns:
                contextual_query = f"{last_turns[-1]['content'][:120]} {query}"

        top_chunks = self.retrieve(contextual_query, k=TOP_K)

        if not top_chunks:
            return {"answer": "No document loaded. Please upload and process a PDF first.", "chunks": [], "score": 0.0}

        try:
            # Run QA on each top chunk individually and pick the best span.
            # (roberta-squad2 scores drop sharply on long concatenated context,
            # so per-chunk scoring gives much better results.)
            best_answer, best_score = "", -1.0
            for chunk in top_chunks:
                chunk_words = chunk.split()[:380]
                chunk_context = " ".join(chunk_words)
                try:
                    result = self.qa(question=query, context=chunk_context)
                except Exception:
                    continue
                if result["score"] > best_score:
                    best_score = result["score"]
                    best_answer = result["answer"].strip()

            score = round(float(best_score), 3)
            answer_text = best_answer

            # Lowered threshold — roberta-squad2 confidence is often low even
            # for correct extractive answers.
            if score < 0.01 or len(answer_text) < 1:
                answer_text = (
                    "I couldn't find a confident answer in the document for that question. "
                    "Try rephrasing, or the topic may not be covered in this PDF."
                )
                score = 0.0
        except Exception as exc:
            answer_text = f"QA model error: {exc}"
            score = 0.0

        return {
            "answer": answer_text,
            "chunks": top_chunks,
            "score": score,
        }
