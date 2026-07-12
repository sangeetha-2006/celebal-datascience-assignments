"""
Step 1b: Data Cleaning
Reads raw scraped JSON files from data/raw/, cleans the text
(removes boilerplate, extra whitespace, junk characters),
and writes cleaned records to data/processed/cleaned.jsonl
"""
import json
import re
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = PROCESSED_DIR / "cleaned.jsonl"

# Common boilerplate phrases to strip out (extend as needed per site)
BOILERPLATE_PATTERNS = [
    r"cookie policy",
    r"all rights reserved",
    r"subscribe to our newsletter",
    r"accept all cookies",
    r"terms of service",
]


def clean_text(text: str) -> str:
    text = text.lower()

    # Remove boilerplate phrases
    for pattern in BOILERPLATE_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

    # Remove URLs, emails
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+\.\S+", " ", text)

    # Remove non-printable / weird unicode noise, keep basic punctuation
    text = re.sub(r"[^a-z0-9.,;:!?'\"()\-\s]", " ", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def clean_all(min_chars: int = 200) -> int:
    """Clean every JSON file in data/raw/, write results to cleaned.jsonl.
    Skips documents shorter than min_chars after cleaning (likely low-value pages).
    Returns number of records written.
    """
    records_written = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        for raw_path in RAW_DIR.glob("*.json"):
            raw = json.loads(raw_path.read_text(encoding="utf-8"))
            cleaned = clean_text(raw.get("text", ""))

            if len(cleaned) < min_chars:
                print(f"[cleaner] skipping {raw_path.name} (too short after cleaning)")
                continue

            record = {
                "source_url": raw.get("url"),
                "title": raw.get("title"),
                "text": cleaned,
                "char_count": len(cleaned),
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            records_written += 1

    print(f"[cleaner] wrote {records_written} cleaned records -> {OUTPUT_FILE}")
    return records_written


if __name__ == "__main__":
    clean_all()
