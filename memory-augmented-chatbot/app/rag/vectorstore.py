"""
Step 2b: Vector store.

Thin wrapper around a FAISS index that also keeps the chunk metadata
(text, source_url, title, etc.) alongside each vector, since FAISS
itself only stores raw vectors and integer IDs.

Storage layout in data/vectorstore/:
    index.faiss   - the FAISS index (vectors)
    metadata.jsonl - one JSON record per vector, in the same order as
                     the vectors were added (row i <-> metadata line i)
"""
import json
from pathlib import Path

import faiss
import numpy as np

VECTORSTORE_DIR = Path("data/vectorstore")
INDEX_PATH = VECTORSTORE_DIR / "index.faiss"
METADATA_PATH = VECTORSTORE_DIR / "metadata.jsonl"


class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        # Inner product on normalized vectors = cosine similarity
        self.index = faiss.IndexFlatIP(dim)
        self.metadata: list[dict] = []

    def add(self, vectors: np.ndarray, metadata: list[dict]):
        """Add vectors + their corresponding metadata records."""
        if vectors.shape[0] != len(metadata):
            raise ValueError("Number of vectors must match number of metadata records")
        self.index.add(vectors)
        self.metadata.extend(metadata)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[dict]:
        """Return top_k most similar chunks to the query vector,
        each with its metadata and similarity score.
        """
        query_vector = query_vector.reshape(1, -1).astype("float32")
        scores, indices = self.index.search(query_vector, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 when fewer than top_k results exist
                continue
            record = dict(self.metadata[idx])
            record["score"] = float(score)
            results.append(record)
        return results

    def save(self):
        VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(INDEX_PATH))
        with open(METADATA_PATH, "w", encoding="utf-8") as f:
            for record in self.metadata:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[vectorstore] saved {self.index.ntotal} vectors -> {INDEX_PATH}")

    @classmethod
    def load(cls) -> "VectorStore":
        if not INDEX_PATH.exists() or not METADATA_PATH.exists():
            raise FileNotFoundError(
                "No saved vector store found. Run app/rag/build_index.py first."
            )
        index = faiss.read_index(str(INDEX_PATH))
        store = cls.__new__(cls)  # bypass __init__ (don't need to re-create index)
        store.dim = index.d
        store.index = index
        store.metadata = [
            json.loads(line) for line in METADATA_PATH.read_text(encoding="utf-8").splitlines()
        ]
        return store
