"""
Step 2c: Build the vector index.

Reads data/processed/chunks.jsonl (produced by app/ingestion/chunker.py),
embeds every chunk, and saves a FAISS index + metadata to data/vectorstore/.

Usage:
    python -m app.rag.build_index
"""
import json
from pathlib import Path

from app.rag.embedder import embed_texts, get_embedding_dim
from app.rag.vectorstore import VectorStore

CHUNKS_FILE = Path("data/processed/chunks.jsonl")


def build_index(batch_size: int = 32) -> int:
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"{CHUNKS_FILE} not found. Run the ingestion pipeline first "
            "(python -m app.ingestion.run_pipeline --urls urls_example.txt)."
        )

    chunks = [json.loads(line) for line in CHUNKS_FILE.read_text(encoding="utf-8").splitlines()]
    if not chunks:
        print("[build_index] no chunks found, nothing to embed")
        return 0

    texts = [c["text"] for c in chunks]
    print(f"[build_index] embedding {len(texts)} chunks...")
    vectors = embed_texts(texts, batch_size=batch_size)

    dim = get_embedding_dim()
    store = VectorStore(dim=dim)

    # Metadata kept per-vector so retrieval results can cite their source
    metadata = [
        {
            "chunk_id": c["chunk_id"],
            "source_url": c["source_url"],
            "title": c.get("title"),
            "text": c["text"],
        }
        for c in chunks
    ]
    store.add(vectors, metadata)
    store.save()

    print(f"[build_index] done. {len(chunks)} chunks indexed.")
    return len(chunks)


if __name__ == "__main__":
    build_index()
