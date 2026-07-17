"""
Step 5f: Preference extraction.

Detects simple, explicit preference statements in user messages (name,
"my favorite X is Y") using regex — no LLM call needed, so it's free and
100% reliable (unlike asking a small local model to "remember" facts from
raw chat history, which is fragile).

Extracted preferences are written to the structured `preferences` table
(Phase 4), so later questions like "what's my name?" are answered from a
real stored fact instead of the model guessing from conversation text.
"""
import re

from app.memory.memory_store import set_preference

_PATTERNS = [
    (re.compile(r"\bmy name is ([A-Za-z][A-Za-z\-' ]{0,30})", re.IGNORECASE), "name"),
    (re.compile(r"\bcall me ([A-Za-z][A-Za-z\-' ]{0,30})", re.IGNORECASE), "name"),
    (re.compile(r"\bmy (?:city|hometown|location) is ([A-Za-z][A-Za-z\-' ]{0,40})", re.IGNORECASE), "city"),
    (re.compile(r"\bi live in ([A-Za-z][A-Za-z\-' ]{0,40})", re.IGNORECASE), "city"),
    (re.compile(r"\bi(?:'ve| have)? moved to ([A-Za-z][A-Za-z\-' ]{0,40})", re.IGNORECASE), "city"),
    (re.compile(r"\bi(?:'m| am)? now (?:living|based) in ([A-Za-z][A-Za-z\-' ]{0,40})", re.IGNORECASE), "city"),
    (re.compile(r"\bi relocated to ([A-Za-z][A-Za-z\-' ]{0,40})", re.IGNORECASE), "city"),
    (re.compile(r"\bmy (?:job|occupation|profession) is ([A-Za-z][A-Za-z\-' ]{0,40})", re.IGNORECASE), "occupation"),
    (re.compile(r"\bi work as an? ([A-Za-z][A-Za-z\-' ]{0,40})", re.IGNORECASE), "occupation"),
    (re.compile(r"\bmy favorite (\w+) is ([A-Za-z0-9\-' ]{1,40})", re.IGNORECASE), None),
    (re.compile(r"\bmy favourite (\w+) is ([A-Za-z0-9\-' ]{1,40})", re.IGNORECASE), None),
]


def extract_and_save_preferences(user_id: str, message: str) -> list[tuple[str, str]]:
    """Scan a user message for explicit preference statements and save any
    found to the structured preference store. Returns the list of
    (key, value) pairs that were saved, for logging/debugging.
    """
    saved = []

    for pattern, fixed_key in _PATTERNS:
        match = pattern.search(message)
        if not match:
            continue

        if fixed_key:
            value = match.group(1).strip().rstrip("?.!,")
            key = fixed_key
        else:
            # "my favorite <noun> is <value>" -> key = favorite_<noun>
            noun = match.group(1).strip().lower()
            value = match.group(2).strip().rstrip("?.!,")
            key = f"favorite_{noun}"

        if value:
            set_preference(user_id, key, value)
            saved.append((key, value))

    return saved
