"""
Step 3c: Build the knowledge graph.

Reads data/processed/cleaned.jsonl (one record per scraped document —
NOT the small RAG chunks, since entity extraction works better over
a full document than a tiny 800-char slice), runs LLM-based extraction
on each document, and loads the resulting triples into Neo4j.

Usage:
    python -m app.graph.build_graph
    python -m app.graph.build_graph --limit 5      # only process first 5 docs (cost control)
    python -m app.graph.build_graph --clear-first  # wipe the graph before rebuilding
"""
import argparse
import json
from pathlib import Path

from app.graph.extractor import extract_triples
from app.graph.neo4j_client import Neo4jClient

CLEANED_FILE = Path("data/processed/cleaned.jsonl")


def build_graph(limit: int | None = None, clear_first: bool = False) -> int:
    if not CLEANED_FILE.exists():
        raise FileNotFoundError(
            f"{CLEANED_FILE} not found. Run the ingestion pipeline first "
            "(python -m app.ingestion.run_pipeline --urls urls_example.txt)."
        )

    docs = [json.loads(line) for line in CLEANED_FILE.read_text(encoding="utf-8").splitlines()]
    if limit:
        docs = docs[:limit]

    if not docs:
        print("[build_graph] no documents found, nothing to extract")
        return 0

    with Neo4jClient() as client:
        client.verify_connectivity()
        print("[build_graph] connected to Neo4j successfully")

        if clear_first:
            print("[build_graph] clearing existing graph...")
            client.clear_all()

        total_triples = 0
        for i, doc in enumerate(docs, start=1):
            print(f"[build_graph] ({i}/{len(docs)}) extracting from: {doc.get('title') or doc['source_url']}")
            triples = extract_triples(doc["text"])
            print(f"[build_graph]   -> {len(triples)} triples extracted")

            client.write_triples(triples, source_url=doc["source_url"])
            total_triples += len(triples)

        print(f"\n[build_graph] done. {total_triples} total triples written across {len(docs)} documents.")
        return total_triples


def main():
    parser = argparse.ArgumentParser(description="Extract entities/relationships and load into Neo4j")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N documents (cost control)")
    parser.add_argument("--clear-first", action="store_true", help="Wipe the graph before rebuilding")
    args = parser.parse_args()

    build_graph(limit=args.limit, clear_first=args.clear_first)


if __name__ == "__main__":
    main()
