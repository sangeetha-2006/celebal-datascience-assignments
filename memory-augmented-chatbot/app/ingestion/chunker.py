"""
Step 1c: Text Chunking
Reads cleaned records from data/processed/cleaned.jsonl and splits
each document's text into overlapping chunks suitable for embedding.

Each chunk keeps a reference back to its source document/URL so
retrieved chunks can be cited.
"""
import json
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

PROCESSED_DIR = Path("data/processed")
INPUT_FILE = PROCESSED_DIR / "cleaned.jsonl"
OUTPUT_FILE = PROCESSED_DIR / "chunks.jsonl"


def chunk_documents(
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> int:
    """Split all cleaned documents into chunks, write to chunks.jsonl.
    Returns number of chunks written.
    """
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"{INPUT_FILE} not found. Run app/ingestion/cleaner.py first."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunk_count = 0
    with open(INPUT_FILE, encoding="utf-8") as in_f, open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        for line in in_f:
            doc = json.loads(line)
            pieces = splitter.split_text(doc["text"])

            for i, piece in enumerate(pieces):
                chunk_record = {
                    "chunk_id": f"{doc['source_url']}::chunk_{i}",
                    "source_url": doc["source_url"],
                    "title": doc.get("title"),
                    "chunk_index": i,
                    "text": piece,
                }
                out_f.write(json.dumps(chunk_record, ensure_ascii=False) + "\n")
                chunk_count += 1

    print(f"[chunker] wrote {chunk_count} chunks -> {OUTPUT_FILE}")
    return chunk_count


if __name__ == "__main__":
    chunk_documents()
