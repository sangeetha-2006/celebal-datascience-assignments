"""
RAG (Retrieval-Augmented Generation) Document Question Answering System
-------------------------------------------------------------------------
A fully local, no-API-key pipeline that:
  1. Loads a document (.txt or .pdf)
  2. Splits it into chunks
  3. Embeds each chunk with sentence-transformers
  4. Stores embeddings in a FAISS vector index
  5. Embeds the user's question
  6. Retrieves the most relevant chunks
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
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pypdf import PdfReader


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
# 3 & 4. EMBEDDING CREATION + VECTOR DATABASE
# ---------------------------------------------------------------------------
class VectorStore:
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        print(f"Loading embedding model: {embedding_model_name} ...")
        self.embedder = SentenceTransformer(embedding_model_name)
        self.index = None
        self.chunks: list[str] = []

    def build(self, chunks: list[str]):
        """Embed all chunks and build a FAISS index."""
        self.chunks = chunks
        embeddings = self.embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
        dim = embeddings.shape[1]
        # Using L2 distance; normalize vectors for cosine-similarity-like behavior
        faiss.normalize_L2(embeddings)
        self.index = faiss.IndexFlatIP(dim)  # inner product on normalized vecs = cosine sim
        self.index.add(embeddings)
        print(f"Indexed {len(chunks)} chunks.")

    def search(self, query: str, top_k: int = 3) -> list[str]:
        """Embed the query and retrieve the top_k most similar chunks."""
        query_vec = self.embedder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_vec)
        scores, indices = self.index.search(query_vec, top_k)
        return [self.chunks[i] for i in indices[0] if i != -1]


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
    def __init__(self, chunk_size=500, overlap=50, top_k=3):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.top_k = top_k
        self.vector_store = VectorStore()
        self.generator = AnswerGenerator()

    def ingest(self, file_path: str):
        print(f"\nLoading document: {file_path}")
        text = load_document(file_path)
        chunks = chunk_text(text, self.chunk_size, self.overlap)
        self.vector_store.build(chunks)

    def ask(self, question: str) -> str:
        relevant_chunks = self.vector_store.search(question, self.top_k)
        answer = self.generator.generate(question, relevant_chunks)
        return answer, relevant_chunks


# ---------------------------------------------------------------------------
# MAIN — interactive demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    DOC_PATH = "sample_document.txt"  # change this to your own .txt or .pdf

    rag = RAGPipeline(chunk_size=500, overlap=50, top_k=3)
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
