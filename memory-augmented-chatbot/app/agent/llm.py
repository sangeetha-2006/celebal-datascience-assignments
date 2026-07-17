"""
Step 5c: LLM wrapper (Ollama, fully local — no API key, no quota).

Ollama runs an open model directly on your machine and exposes a simple
local HTTP API. Install: https://ollama.com/download
Then pull a model once: `ollama pull llama3.2:1b`
"""
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen2.5:7b"



def generate(prompt: str, model: str = DEFAULT_MODEL) -> str:
    """Send a prompt to a local Ollama server and return the text response."""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        response.raise_for_status()
    except requests.ConnectionError as e:
        raise RuntimeError(
            "Could not reach Ollama at http://localhost:11434. "
            "Make sure Ollama is installed and running (it usually starts "
            "automatically after installation; otherwise run `ollama serve`)."
        ) from e

    data = response.json()
    return data.get("response", "").strip()
