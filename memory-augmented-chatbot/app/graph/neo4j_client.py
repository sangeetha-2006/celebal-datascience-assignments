"""
Step 3a: Neo4j connection + write/query helpers.

Wraps the official neo4j driver with simple methods for:
- writing extracted (entity, relationship, entity) triples
- running arbitrary Cypher queries for structured lookups
"""
from neo4j import GraphDatabase

from app.config import settings


class Neo4jClient:
    def __init__(self, uri: str | None = None, username: str | None = None, password: str | None = None):
        self.uri = uri or settings.neo4j_uri
        self.username = username or settings.neo4j_username
        self.password = password or settings.neo4j_password

        if not self.uri or not self.password:
            raise ValueError(
                "Neo4j credentials missing. Set NEO4J_URI, NEO4J_USERNAME, "
                "NEO4J_PASSWORD in your .env file."
            )

        self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))

    def close(self):
        self.driver.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def verify_connectivity(self):
        self.driver.verify_connectivity()

    def write_triple(self, source: str, relation: str, target: str, source_url: str = ""):
        """Write one (entity)-[RELATION]->(entity) triple.
        Entities are MERGEd by name so repeated mentions don't create duplicates.
        The relationship type is sanitized to a valid Cypher identifier.
        """
        rel_type = _sanitize_relation(relation)
        cypher = (
            f"MERGE (a:Entity {{name: $source}}) "
            f"MERGE (b:Entity {{name: $target}}) "
            f"MERGE (a)-[r:{rel_type}]->(b) "
            f"SET r.source_url = $source_url"
        )
        with self.driver.session() as session:
            session.run(cypher, source=source, target=target, source_url=source_url)

    def write_triples(self, triples: list[dict], source_url: str = ""):
        """Bulk-write a list of {'source', 'relation', 'target'} dicts."""
        for t in triples:
            self.write_triple(t["source"], t["relation"], t["target"], source_url=source_url)

    def query_entity(self, name: str) -> list[dict]:
        """Return all relationships (in both directions) involving an entity,
        matched case-insensitively on a substring of the name.
        """
        cypher = (
            "MATCH (a:Entity)-[r]->(b:Entity) "
            "WHERE toLower(a.name) CONTAINS toLower($name) "
            "   OR toLower(b.name) CONTAINS toLower($name) "
            "RETURN a.name AS source, type(r) AS relation, b.name AS target, "
            "       r.source_url AS source_url "
            "LIMIT 50"
        )
        with self.driver.session() as session:
            result = session.run(cypher, name=name)
            return [dict(record) for record in result]

    def run_cypher(self, cypher: str, **params) -> list[dict]:
        """Run arbitrary Cypher, for ad-hoc exploration."""
        with self.driver.session() as session:
            result = session.run(cypher, **params)
            return [dict(record) for record in result]

    def clear_all(self):
        """Danger: deletes every node and relationship. Useful for re-running from scratch."""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")


def _sanitize_relation(relation: str) -> str:
    """Cypher relationship types can't have spaces/punctuation.
    Convert 'is part of' -> 'IS_PART_OF'.
    """
    cleaned = "".join(c if c.isalnum() else "_" for c in relation.upper())
    cleaned = "_".join(filter(None, cleaned.split("_")))  # collapse repeated underscores
    return cleaned or "RELATED_TO"
