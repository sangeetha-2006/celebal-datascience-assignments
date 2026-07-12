"""
Runs the full ingestion pipeline in order:
  1. Scrape URLs -> data/raw/*.json
  2. Clean text  -> data/processed/cleaned.jsonl
  3. Chunk text  -> data/processed/chunks.jsonl

Usage:
    python -m app.ingestion.run_pipeline --urls urls.txt
"""
import argparse
from pathlib import Path

from app.ingestion.scraper import scrape_urls
from app.ingestion.cleaner import clean_all
from app.ingestion.chunker import chunk_documents


def main():
    parser = argparse.ArgumentParser(description="Run the full ingestion pipeline")
    parser.add_argument("--urls", type=str, required=True, help="Path to a text file, one URL per line")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between scrape requests")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=120)
    args = parser.parse_args()

    urls = Path(args.urls).read_text(encoding="utf-8").splitlines()

    print("\n=== STEP 1: SCRAPING ===")
    scrape_urls(urls, delay=args.delay)

    print("\n=== STEP 2: CLEANING ===")
    clean_all()

    print("\n=== STEP 3: CHUNKING ===")
    chunk_documents(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)

    print("\nPipeline complete. See data/processed/chunks.jsonl")


if __name__ == "__main__":
    main()
