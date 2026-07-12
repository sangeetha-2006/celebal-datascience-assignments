"""
Step 5b: Dynamic tools.

Free, no-API-key tools that fetch real-time information, demonstrating the
"dynamic intelligence layer" from the problem statement. Add more tools here
as needed (e.g. a real search API) — each just needs to take a query string
and return a text string.
"""
from datetime import datetime, timezone

import requests


def get_current_datetime(_query: str = "") -> str:
    """Return the current local date and time (using the machine's own
    timezone), plus UTC for reference.
    """
    local_now = datetime.now().astimezone()
    tz_name = local_now.tzname() or "local time"
    local_str = local_now.strftime("%Y-%m-%d %I:%M:%S %p")
    utc_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    return f"Current time: {local_str} ({tz_name}) — {utc_str} UTC"


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


# Very simple keyword-based router: which tool (if any) should handle this query?
def select_tool(query: str) -> tuple[str, callable] | None:
    """Return (tool_name, tool_function) if the query looks like it needs a
    live tool, else None. This is a simple heuristic — in a larger system
    this decision could itself be made by an LLM.
    """
    lowered = query.lower()

    weather_keywords = ["weather", "temperature", "forecast", "raining", "sunny"]
    if any(kw in lowered for kw in weather_keywords):
        # naive location extraction: text after "in" or "for", else default
        location = "London"
        for marker in [" in ", " for "]:
            if marker in lowered:
                location = query.split(marker, 1)[1].strip().rstrip("?.!")
                break
        return "weather", lambda q: get_weather(location)

    time_keywords = ["current time", "what time", "today's date", "current date", "what date"]
    if any(kw in lowered for kw in time_keywords):
        return "datetime", get_current_datetime

    return None
