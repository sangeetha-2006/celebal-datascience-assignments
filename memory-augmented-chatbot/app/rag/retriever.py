"""
Step 2d: Retrieval.

Given a natural-language query, embeds it and searches the saved
FAISS vector store for the most relevant chunks.

Usage:
    python -m app.rag.retriever "What is retrieval-augmented generation?"
"""
import re
import sys

from app.rag.embedder import embed_query
from app.rag.vectorstore import VectorStore

_store_cache: VectorStore | None = None

# Common words that shouldn't count as meaningful keyword matches on their own
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "what", "which", "who",
    "how", "why", "when", "where", "does", "do", "did", "explain", "about",
    "this", "that", "these", "those", "and", "for", "with", "used", "use",
}


def _get_store() -> VectorStore:
    global _store_cache
    if _store_cache is None:
        _store_cache = VectorStore.load()
    return _store_cache


def _extract_keywords(query: str) -> list[str]:
    """Pull out meaningful words from the query for exact-match boosting —
    e.g. 'Explain LangGraph.' -> ['LangGraph']. Case is preserved so
    distinctive CamelCase/ProperNoun terms are easy to match verbatim.
    """
    words = re.findall(r"[A-Za-z][A-Za-z0-9]{2,}", query)
    return [w for w in words if w.lower() not in _STOPWORDS]


def retrieve(query: str, top_k: int = 5, candidate_pool: int = 12) -> list[dict]:
    """Return the top_k most relevant chunks for the query.

    Uses semantic search (FAISS) to pull a larger candidate pool, then
    re-ranks with a small keyword-match bonus — this fixes cases where a
    chunk containing the exact term asked about (e.g. "LangGraph") loses
    to chunks that are only vaguely semantically similar (shared generic
    words like "graph"/"language") but never mention the term itself.

    Each result dict has: chunk_id, source_url, title, text, score.
    """
    store = _get_store()
    query_vector = embed_query(query)
    candidates = store.search(query_vector, top_k=max(candidate_pool, top_k))

    keywords = _extract_keywords(query)
    for c in candidates:
        bonus = 0.0
        if keywords:
            text_lower = c["text"].lower()
            matches = sum(1 for kw in keywords if kw.lower() in text_lower)
            bonus = 0.12 * matches  # meaningful but doesn't fully override semantic relevance
        c["score"] = c["score"] + bonus

    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates[:top_k]


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
