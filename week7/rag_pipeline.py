"""
RAG (Retrieval-Augmented Generation) Document Question Answering System
-------------------------------------------------------------------------
A fully local, no-API-key pipeline that:
  1. Ingests a document: .txt, .pdf, or a Hugging Face dataset
  2. Splits it into chunks
  3. Embeds each chunk with sentence-transformers
  4. Stores embeddings in a FAISS vector index
  5. Embeds the user's question
  6. Retrieves the most relevant chunks (vector search, optionally hybrid
     with BM25 keyword search, optionally re-ranked with a cross-encoder)
  7. Generates an answer with a local Hugging Face model (FLAN-T5)

Run with:  python rag_pipeline.py
"""

import os

# --- macOS fix: faiss and torch both bundle their own OpenMP runtime.        ---
# --- Loading both causes "Segmentation fault: 11" on Mac (esp. Apple         ---
# --- Silicon). This must be set BEFORE faiss/torch are imported.            ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pypdf import PdfReader
from rank_bm25 import BM25Okapi


# ---------------------------------------------------------------------------
# 1. DOCUMENT INGESTION
# ---------------------------------------------------------------------------
def load_document(file_path: str) -> str:
    """Load a .txt or .pdf file and return its raw text."""
    if file_path.lower().endswith(".pdf"):
        reader = PdfReader(file_path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    return text


def load_from_huggingface(
    dataset_name: str,
    split: str = "train",
    text_column: str = "text",
    num_rows: int = 50,
) -> str:
    """
    Load rows from a Hugging Face dataset (e.g. domain-specific archives such
    as 'vectara/open_ragbench') and concatenate the chosen text column into
    one raw text blob, ready for chunking.

    Example:
        text = load_from_huggingface("vectara/open_ragbench", split="train",
                                      text_column="text", num_rows=50)
    """
    from datasets import load_dataset  # imported lazily; only needed for this path

    ds = load_dataset(dataset_name, split=split, streaming=True)
    rows = []
    for i, row in enumerate(ds):
        if i >= num_rows:
            break
        value = row.get(text_column)
        if value:
            rows.append(str(value))
    if not rows:
        raise ValueError(
            f"No rows found for column '{text_column}' in {dataset_name}. "
            f"Check the dataset's schema/column names on the Hugging Face Hub."
        )
    return "\n\n".join(rows)


# ---------------------------------------------------------------------------
# 2. TEXT CHUNKING
# ---------------------------------------------------------------------------
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split text into overlapping chunks (by characters).
    Overlap helps preserve context that would otherwise be cut at chunk boundaries.
    """
    text = " ".join(text.split())  # normalize whitespace
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]


# ---------------------------------------------------------------------------
# 3 & 4. EMBEDDING CREATION + VECTOR DATABASE (+ optional hybrid / re-rank)
# ---------------------------------------------------------------------------
class VectorStore:
    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        use_hybrid: bool = False,
        rerank_model_name: str | None = None,
    ):
        print(f"Loading embedding model: {embedding_model_name} ...")
        self.embedder = SentenceTransformer(embedding_model_name)
        self.embedding_dim = self.embedder.get_sentence_embedding_dimension()
        self.index = None
        self.chunks: list[str] = []

        self.use_hybrid = use_hybrid
        self.bm25 = None  # built in build() if use_hybrid=True

        self.reranker = None
        if rerank_model_name:
            print(f"Loading re-ranking model: {rerank_model_name} ...")
            self.reranker = CrossEncoder(rerank_model_name)

    def build(self, chunks: list[str]):
        """Embed all chunks, build a FAISS index, and (optionally) a BM25 index."""
        self.chunks = chunks
        embeddings = self.embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
        dim = embeddings.shape[1]
        faiss.normalize_L2(embeddings)  # so inner product == cosine similarity
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        print(f"Indexed {len(chunks)} chunks (embedding dim = {dim}).")

        if self.use_hybrid:
            tokenized_chunks = [c.lower().split() for c in chunks]
            self.bm25 = BM25Okapi(tokenized_chunks)
            print("BM25 keyword index built for hybrid search.")

    def _vector_search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        query_vec = self.embedder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_vec)
        scores, indices = self.index.search(query_vec, top_k)
        return [(int(i), float(s)) for i, s in zip(indices[0], scores[0]) if i != -1]

    def _bm25_search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_indices]

    @staticmethod
    def _normalize(scores: list[float]) -> list[float]:
        if not scores:
            return scores
        lo, hi = min(scores), max(scores)
        if hi == lo:
            return [1.0 for _ in scores]
        return [(s - lo) / (hi - lo) for s in scores]

    def search(self, query: str, top_k: int = 3, fetch_k: int = 10) -> list[str]:
        """
        Retrieve the top_k most relevant chunks.
        - Vector-only mode: pure cosine similarity search.
        - Hybrid mode: combine normalized vector + BM25 scores (0.5/0.5).
        - If a reranker is loaded, first over-fetch `fetch_k` candidates, then
          re-score them with the cross-encoder and keep the true top_k.
        """
        candidate_k = fetch_k if self.reranker else top_k

        vector_hits = dict(self._vector_search(query, candidate_k))

        if self.use_hybrid and self.bm25 is not None:
            bm25_hits = dict(self._bm25_search(query, candidate_k))
            all_ids = set(vector_hits) | set(bm25_hits)
            v_scores = self._normalize([vector_hits.get(i, 0.0) for i in all_ids])
            b_scores = self._normalize([bm25_hits.get(i, 0.0) for i in all_ids])
            combined = {
                idx: 0.5 * v + 0.5 * b
                for idx, v, b in zip(all_ids, v_scores, b_scores)
            }
            ranked_ids = sorted(combined, key=combined.get, reverse=True)[:candidate_k]
        else:
            ranked_ids = sorted(vector_hits, key=vector_hits.get, reverse=True)[:candidate_k]

        candidates = [self.chunks[i] for i in ranked_ids]

        if self.reranker:
            pairs = [[query, c] for c in candidates]
            rerank_scores = self.reranker.predict(pairs)
            reranked = sorted(zip(candidates, rerank_scores), key=lambda x: x[1], reverse=True)
            return [c for c, _ in reranked[:top_k]]

        return candidates[:top_k]


# ---------------------------------------------------------------------------
# 7. ANSWER GENERATION
# ---------------------------------------------------------------------------
class AnswerGenerator:
    def __init__(self, model_name: str = "google/flan-t5-base"):
        print(f"Loading generation model: {model_name} ...")
        # Using the model/tokenizer directly (instead of pipeline()) because
        # transformers v5 removed the "text2text-generation" pipeline task.
        # This approach works on any transformers version.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model_name = model_name

    def generate(self, question: str, context_chunks: list[str]) -> str:
        context = "\n\n".join(context_chunks)
        prompt = (
            "Answer the question using only the context below. "
            "If the answer is not in the context, say you don't know.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        output_ids = self.model.generate(**inputs, max_new_tokens=200)
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# FULL PIPELINE
# ---------------------------------------------------------------------------
class RAGPipeline:
    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 50,
        top_k: int = 3,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        generation_model_name: str = "google/flan-t5-base",
        use_hybrid: bool = False,
        rerank_model_name: str | None = None,
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.top_k = top_k
        self.vector_store = VectorStore(
            embedding_model_name=embedding_model_name,
            use_hybrid=use_hybrid,
            rerank_model_name=rerank_model_name,
        )
        self.generator = AnswerGenerator(model_name=generation_model_name)

    def ingest(self, file_path: str):
        """Ingest a local .txt/.pdf file."""
        print(f"\nLoading document: {file_path}")
        text = load_document(file_path)
        self._ingest_text(text)

    def ingest_huggingface(self, dataset_name: str, split="train", text_column="text", num_rows=50):
        """Ingest text pulled from a Hugging Face dataset."""
        print(f"\nLoading Hugging Face dataset: {dataset_name} ({split}, {num_rows} rows)")
        text = load_from_huggingface(dataset_name, split, text_column, num_rows)
        self._ingest_text(text)

    def _ingest_text(self, text: str):
        chunks = chunk_text(text, self.chunk_size, self.overlap)
        self.vector_store.build(chunks)

    def ask(self, question: str):
        relevant_chunks = self.vector_store.search(question, self.top_k)
        answer = self.generator.generate(question, relevant_chunks)
        return answer, relevant_chunks


# ---------------------------------------------------------------------------
# MAIN — interactive demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    DOC_PATH = "sample_document.txt"  # change this to your own .txt or .pdf

    # Flip these to True to try hybrid search / re-ranking (requirement 8):
    USE_HYBRID = False
    RERANK_MODEL = None  # e.g. "cross-encoder/ms-marco-MiniLM-L-6-v2"

    rag = RAGPipeline(
        chunk_size=500,
        overlap=50,
        top_k=3,
        use_hybrid=USE_HYBRID,
        rerank_model_name=RERANK_MODEL,
    )
    rag.ingest(DOC_PATH)

    print("\n=== RAG Document Q&A ===")
    print("Type a question about the document (or 'exit' to quit)\n")

    while True:
        question = input("Your question: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue

        answer, sources = rag.ask(question)

        print("\n--- Answer ---")
        print(answer)
        print("\n--- Retrieved context (for reference) ---")
        for i, chunk in enumerate(sources, 1):
            print(f"[{i}] {chunk[:150]}...")
        print()
