"""
Step 4a: Database connection + schema.

Uses SQLite (built into Python, zero setup) to store:
- user preferences (key/value pairs, e.g. "likes_short_answers": "true")
- conversation history (every message exchanged, per user)

The database is a single file: data/memory.db
"""
import sqlite3
from pathlib import Path

DB_PATH = Path("data/memory.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS preferences (
    user_id TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (user_id, key),
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE INDEX IF NOT EXISTS idx_messages_user ON messages(user_id, created_at);
"""


def get_connection() -> sqlite3.Connection:
    """Return a connection to the memory database, creating the file/schema
    if it doesn't exist yet.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # access columns by name, e.g. row["content"]
    conn.executescript(SCHEMA)
    return conn


def init_db():
    """Explicitly initialize the database (useful for a setup script)."""
    conn = get_connection()
    conn.close()
    print(f"[db] initialized SQLite database at {DB_PATH}")


if __name__ == "__main__":
    init_db()
