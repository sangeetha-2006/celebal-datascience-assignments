"""
Step 4b: Memory store.

High-level functions for reading/writing long-term user memory:
- preferences: durable key/value facts about a user (e.g. "prefers_short_answers")
- messages: the running conversation history

Also provides get_memory_context(), which formats a user's preferences +
recent history into a text block ready to drop into an LLM prompt.
"""
from app.memory.db import get_connection


def ensure_user(user_id: str):
    """Create the user record if it doesn't already exist. Safe to call every time."""
    conn = get_connection()
    conn.execute("INSERT OR IGNORE INTO users (user_id) VALUES (?)", (user_id,))
    conn.commit()
    conn.close()


def set_preference(user_id: str, key: str, value: str):
    """Store or update a durable preference for a user."""
    ensure_user(user_id)
    conn = get_connection()
    conn.execute(
        """
        INSERT INTO preferences (user_id, key, value, updated_at)
        VALUES (?, ?, ?, datetime('now'))
        ON CONFLICT(user_id, key) DO UPDATE SET
            value = excluded.value,
            updated_at = excluded.updated_at
        """,
        (user_id, key, value),
    )
    conn.commit()
    conn.close()


def get_preferences(user_id: str) -> dict:
    """Return all preferences for a user as a plain dict."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT key, value FROM preferences WHERE user_id = ?", (user_id,)
    ).fetchall()
    conn.close()
    return {row["key"]: row["value"] for row in rows}


def add_message(user_id: str, role: str, content: str):
    """Record one message (user or assistant) in the conversation history."""
    ensure_user(user_id)
    conn = get_connection()
    conn.execute(
        "INSERT INTO messages (user_id, role, content) VALUES (?, ?, ?)",
        (user_id, role, content),
    )
    conn.commit()
    conn.close()


def get_recent_history(user_id: str, limit: int = 10) -> list[dict]:
    """Return the most recent `limit` messages for a user, oldest first."""
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT role, content, created_at FROM messages
        WHERE user_id = ?
        ORDER BY created_at DESC, id DESC
        LIMIT ?
        """,
        (user_id, limit),
    ).fetchall()
    conn.close()
    # rows come back newest-first; reverse so they read chronologically
    return [dict(r) for r in reversed(rows)]


def get_memory_context(user_id: str, history_limit: int = 10) -> str:
    """Build a text block summarizing what we know about a user, suitable
    for inserting into an LLM system/context prompt.
    """
    prefs = get_preferences(user_id)
    history = get_recent_history(user_id, limit=history_limit)

    parts = []

    if prefs:
        pref_lines = "\n".join(f"- {k}: {v}" for k, v in prefs.items())
        parts.append(f"User preferences:\n{pref_lines}")
    else:
        parts.append("User preferences: none recorded yet.")

    if history:
        history_lines = "\n".join(f"{m['role']}: {m['content']}" for m in history)
        parts.append(f"Recent conversation history:\n{history_lines}")
    else:
        parts.append("Recent conversation history: none yet (new conversation).")

    return "\n\n".join(parts)
