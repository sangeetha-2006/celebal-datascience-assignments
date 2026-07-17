"""
Step 5b: Dynamic tools.

Free, no-API-key tools that fetch real-time information, demonstrating the
"dynamic intelligence layer" from the problem statement. Add more tools here
as needed (e.g. a real search API) — each just needs to take a query string
and return a text string.
"""
import re
import re
from datetime import datetime, timezone

import requests

RECALL_PATTERN = re.compile(
    r"(?:have|did|has)\s+(?:i|I|you)\s+"
    r"(?:ask(?:ed)?|mention(?:ed)?|talk(?:ed)?\s+about|discuss(?:ed)?|say|said|tell|told(?:\s+you)?)\s+"
    r"(?:about\s+|you\s+)?(.+?)\s*(?:before|already|previously|earlier)?\??$",
    re.IGNORECASE,
)


FIRST_QUESTION_PATTERN = re.compile(
    r"\b(?:what was|what's|whats)\s+my\s+first\s+(?:question|message)\b",
    re.IGNORECASE,
)


def check_first_question(user_id: str) -> str | None:
    """Deterministically answer 'what was my first question?' by looking
    at the actual full history — the normal prompt window only keeps the
    most recent N messages, so once a conversation grows long enough, the
    true first message silently falls outside what the model can see.
    """
    from app.memory.memory_store import get_recent_history

    history = get_recent_history(user_id, limit=1000)
    user_messages = [h["content"] for h in history if h["role"] == "user"]
    if not user_messages:
        return "You haven't asked anything yet in this conversation."
    return f'Your first question was: "{user_messages[0]}"'


_RECALL_PREFERENCE_SYNONYMS = {
    "name": ["name"],
    "city": ["city", "live", "location", "where i live", "hometown"],
    "occupation": ["job", "occupation", "profession", "work"],
}


SELF_DESCRIPTION_PATTERN = re.compile(
    r"\b(?:your architecture|your system|which subsystem|your (?:hidden )?"
    r"(?:system )?prompt|how (?:do|does) you work|what (?:are|is) you built with|"
    r"why (?:did you|do you) choose|why (?:did you|do you) use)\b",
    re.IGNORECASE,
)

_SELF_DESCRIPTION = (
    "This chatbot is a memory-augmented system with these components: "
    "(1) a RAG pipeline (sentence-transformers embeddings + FAISS) for retrieving "
    "scraped documents, (2) a Neo4j knowledge graph built with spaCy entity "
    "extraction, (3) SQLite for long-term user memory (preferences + history), "
    "(4) a LangGraph agent that routes each question to whichever of these — "
    "plus live tools like time/weather — is relevant, and (5) a local Ollama "
    "model that generates the final answer. It does not use Alibaba Cloud, "
    "OpenAI, or any paid service — everything runs free and mostly local. "
    "It does not reveal API keys, passwords, or internal prompts."
)


def check_self_description(query: str) -> str | None:
    """If the user is asking about THIS project's own architecture/design,
    return a grounded, accurate description — instead of letting the
    underlying local model (which has its own separate identity/training)
    answer from its own confused sense of what 'you' refers to.
    """
    if SELF_DESCRIPTION_PATTERN.search(query):
        return _SELF_DESCRIPTION
    return None


def check_recall(user_id: str, query: str) -> str | None:
    """Deterministically answer 'have I asked/mentioned/told you X before?'
    style questions. Checks the structured preferences table first (reliable,
    handles paraphrases like 'where I live' matching a stored 'city' fact),
    then falls back to a raw text search over conversation history.
    Returns None if the query doesn't match this pattern at all.
    """
    match = RECALL_PATTERN.search(query.strip())
    if not match:
        return None

    keyword = match.group(1).strip().rstrip("?.!,")
    if not keyword or len(keyword) < 2:
        return None

    from app.memory.memory_store import get_preferences, get_recent_history

    keyword_lower = keyword.lower()

    # 1. Check structured preferences first — most reliable, handles paraphrases.
    prefs = get_preferences(user_id)
    for pref_key, synonyms in _RECALL_PREFERENCE_SYNONYMS.items():
        if any(syn in keyword_lower for syn in synonyms):
            if pref_key in prefs:
                return f"Yes — you told me: {pref_key} is {prefs[pref_key]}."
            # also check favorite_<keyword> style keys
    for pref_key, value in prefs.items():
        if pref_key.startswith("favorite_") and keyword_lower in pref_key:
            return f"Yes — you told me: {pref_key.replace('favorite_', 'favorite ')} is {value}."

    # 2. Fall back to a raw substring search over past user messages.
    history = get_recent_history(user_id, limit=200)
    prior_user_messages = [
        h["content"] for h in history
        if h["role"] == "user" and h["content"].strip() != query.strip()
    ]
    hits = [m for m in prior_user_messages if keyword_lower in m.lower()]
    if hits:
        return f"Yes — you asked about '{keyword}' earlier: \"{hits[0]}\""
    return f"No — '{keyword}' hasn't come up earlier in this conversation."


def get_current_datetime(_query: str = "") -> str:
    """Return the current local date and time, based on the clock of the
    machine running this server (not the timezone of any location the
    user might mention — there's no real timezone-conversion here).
    """
    local_now = datetime.now().astimezone()
    tz_name = local_now.tzname() or "local time"
    local_str = local_now.strftime("%Y-%m-%d %I:%M:%S %p")
    utc_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    return (
        f"Current time on this server: {local_str} ({tz_name}) — {utc_str} UTC. "
        f"(Note: this is the server's local time, not a timezone lookup for any "
        f"specific place the user mentioned.)"
    )


def get_weather(location: str) -> str:
    """Fetch current weather for a location using wttr.in (free, no API key).
    Falls back to a friendly error message if the request fails.
    """
    try:
        resp = requests.get(f"https://wttr.in/{location}?format=3", timeout=8)
        resp.raise_for_status()
        return f"Weather report: {resp.text.strip()}"
    except requests.RequestException as e:
        return f"Could not fetch weather for '{location}' ({e})"


def _extract_location(query: str, default: str = "London") -> str:
    """Pull a location out of phrases like 'weather in Paris' or
    'weather?in india' — uses a word-boundary regex so it's robust to
    missing/extra punctuation and spacing around 'in'/'for'.
    """
    match = re.search(r"\b(?:in|for)\b\s+([a-zA-Z][a-zA-Z\s]*)", query, re.IGNORECASE)
    if not match:
        return default
    location = match.group(1).strip().rstrip("?.!,")
    return location or default


# Very simple keyword-based router: which tool (if any) should handle this query?
def select_tool(query: str, user_id: str | None = None) -> tuple[str, callable] | None:
    """Return (tool_name, tool_function) if the query looks like it needs a
    live tool, else None. This is a simple heuristic — in a larger system
    this decision could itself be made by an LLM.
    """
    self_description = check_self_description(query)
    if self_description is not None:
        return "self_description", lambda q: self_description

    if user_id:
        if FIRST_QUESTION_PATTERN.search(query):
            return "first_question", lambda q: check_first_question(user_id)

        recall_answer = check_recall(user_id, query)
        if recall_answer is not None:
            return "recall_check", lambda q: recall_answer

    lowered = query.lower()

    weather_keywords = ["weather", "temperature", "forecast", "raining", "sunny"]
    if any(kw in lowered for kw in weather_keywords):
        location = _extract_location(query)
        return "weather", lambda q: get_weather(location)

    time_keywords = [
        "current time", "what time", "today's date", "current date", "what date",
        "time in ", "time right now", "time now",
    ]
    if any(kw in lowered for kw in time_keywords):
        return "datetime", get_current_datetime

    return None
