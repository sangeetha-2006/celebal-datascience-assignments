"""
Step 2d: Retrieval.

Given a natural-language query, embeds it and searches the saved
FAISS vector store for the most relevant chunks.

Usage:
    python -m app.rag.retriever "What is retrieval-augmented generation?"
"""
import sys

from app.rag.embedder import embed_query
from app.rag.vectorstore import VectorStore

_store_cache: VectorStore | None = None


def _get_store() -> VectorStore:
    global _store_cache
    if _store_cache is None:
        _store_cache = VectorStore.load()
    return _store_cache


def retrieve(query: str, top_k: int = 5) -> list[dict]:
    """Return the top_k most relevant chunks for the query.
    Each result dict has: chunk_id, source_url, title, text, score.
    """
    store = _get_store()
    query_vector = embed_query(query)
    return store.search(query_vector, top_k=top_k)


def format_results(results: list[dict]) -> str:
    """Human-readable rendering of retrieval results, useful for debugging
    and for feeding into an LLM prompt as context.
    """
    lines = []
    for i, r in enumerate(results, start=1):
        lines.append(
            f"[{i}] score={r['score']:.3f} | {r.get('title') or r['source_url']}\n"
            f"    {r['text'][:300]}{'...' if len(r['text']) > 300 else ''}"
        )
    return "\n\n".join(lines)


def main():
    if len(sys.argv) < 2:
        print('Usage: python -m app.rag.retriever "your query here"')
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    results = retrieve(query, top_k=5)
    print(f"\nTop {len(results)} results for: {query}\n")
    print(format_results(results))


if __name__ == "__main__":
    main()
