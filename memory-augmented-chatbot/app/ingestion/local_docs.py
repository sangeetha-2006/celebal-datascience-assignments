"""
Step 1d: Ingest local documentation (e.g. this project's own README) into
the same cleaned.jsonl format used by scraped web pages, so RAG can answer
questions about the project itself — not just external scraped content.

Usage:
    python -m app.ingestion.local_docs README.md
    python -m app.ingestion.local_docs README.md docs/architecture.md
"""
import json
import sys
from pathlib import Path

from app.ingestion.cleaner import clean_text

PROCESSED_DIR = Path("data/processed")
CLEANED_FILE = PROCESSED_DIR / "cleaned.jsonl"


def ingest_local_files(paths: list[str], min_chars: int = 100) -> int:
    """Clean and append local text/markdown files to cleaned.jsonl (the same
    file the scraper's output lands in), so the next chunk/index build picks
    them up automatically. Returns the number of files added.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    added = 0

    with open(CLEANED_FILE, "a", encoding="utf-8") as out_f:
        for p in paths:
            path = Path(p)
            if not path.exists():
                print(f"[local_docs] skipping missing file: {p}")
                continue

            raw_text = path.read_text(encoding="utf-8")
            cleaned = clean_text(raw_text)

            if len(cleaned) < min_chars:
                print(f"[local_docs] skipping {path.name} (too short after cleaning)")
                continue

            record = {
                "source_url": f"local://{path.name}",
                "title": path.stem.replace("_", " ").replace("-", " ").title(),
                "text": cleaned,
                "char_count": len(cleaned),
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            added += 1
            print(f"[local_docs] added {path.name} ({len(cleaned)} chars)")

    if added:
        print(
            f"\n[local_docs] added {added} file(s) to {CLEANED_FILE}. "
            "Now re-run: python -m app.ingestion.chunker && python -m app.rag.build_index"
        )
    return added


if __name__ == "__main__":
    file_paths = sys.argv[1:] or ["README.md"]
    ingest_local_files(file_paths)
