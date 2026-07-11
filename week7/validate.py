"""
Validation script — runs a set of test questions through the RAG pipeline
and writes a Markdown log recording the question, retrieved context, and
generated answer for each one.

This produces the "documented validation logs" evaluator artifact.

Run with:  python validate.py
"""

import datetime
from rag_pipeline import RAGPipeline

DOC_PATH = "sample_document.txt"
LOG_PATH = "validation_log.md"

# Edit this list to match whatever document you're testing with.
TEST_QUESTIONS = [
    "What is Retrieval-Augmented Generation?",
    "Why does RAG matter?",
    "What embedding model is mentioned in the document?",
    "What vector database is mentioned and who developed it?",
    "What are the limitations of RAG?",
    "What is a completely unrelated question with no answer in the document, like what is the capital of France?",
]


def run_validation():
    rag = RAGPipeline(chunk_size=500, overlap=50, top_k=3)
    rag.ingest(DOC_PATH)

    lines = []
    lines.append(f"# Validation Log\n")
    lines.append(f"Generated: {datetime.datetime.now().isoformat(timespec='seconds')}\n")
    lines.append(f"Document tested: `{DOC_PATH}`\n")
    lines.append(f"Chunk size: {rag.chunk_size}, Overlap: {rag.overlap}, Top-k: {rag.top_k}\n")
    lines.append("\n---\n")

    for i, question in enumerate(TEST_QUESTIONS, 1):
        answer, sources = rag.ask(question)

        lines.append(f"\n## Test {i}\n")
        lines.append(f"**Question:** {question}\n")
        lines.append(f"\n**Generated Answer:**\n> {answer}\n")
        lines.append(f"\n**Retrieved Context ({len(sources)} chunks):**\n")
        for j, chunk in enumerate(sources, 1):
            lines.append(f"- `[{j}]` {chunk[:200]}{'...' if len(chunk) > 200 else ''}\n")
        lines.append("\n---\n")

        # Also print to console as it runs
        print(f"\n[{i}] Q: {question}")
        print(f"    A: {answer}")

    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"\nValidation log written to {LOG_PATH}")


if __name__ == "__main__":
    run_validation()
