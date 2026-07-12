"""
Step 3d: Query the knowledge graph.

Simple CLI to look up everything the graph knows about an entity.

Usage:
    python -m app.graph.query_graph "RAG"
"""
import sys

from app.graph.neo4j_client import Neo4jClient


def query_entity(name: str) -> list[dict]:
    with Neo4jClient() as client:
        return client.query_entity(name)


def format_results(results: list[dict]) -> str:
    if not results:
        return "No matching relationships found."
    lines = []
    for r in results:
        lines.append(f"({r['source']}) -[{r['relation']}]-> ({r['target']})")
    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        print('Usage: python -m app.graph.query_graph "entity name"')
        sys.exit(1)

    name = " ".join(sys.argv[1:])
    results = query_entity(name)
    print(f"\nRelationships involving '{name}':\n")
    print(format_results(results))


if __name__ == "__main__":
    main()
