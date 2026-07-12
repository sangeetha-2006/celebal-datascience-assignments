"""
Step 4c: Demo / manual test for the memory store.

Simulates a short conversation with a user, storing a preference and a
couple of messages, then prints the assembled memory context — this is
what Phase 5 (the LangGraph agent) will later feed into the LLM.

Usage:
    python -m app.memory.demo
"""
from app.memory.memory_store import (
    add_message,
    get_memory_context,
    get_preferences,
    set_preference,
)

DEMO_USER = "demo_user_1"


def run_demo():
    print(f"=== Memory demo for user: {DEMO_USER} ===\n")

    # Record a preference
    set_preference(DEMO_USER, "preferred_answer_style", "short and to the point")
    set_preference(DEMO_USER, "interest", "knowledge graphs and RAG systems")
    print("Set preferences:", get_preferences(DEMO_USER))

    # Simulate a short conversation
    add_message(DEMO_USER, "user", "What is RAG?")
    add_message(DEMO_USER, "assistant", "RAG combines retrieval with generation to ground LLM answers in real documents.")
    add_message(DEMO_USER, "user", "Can you keep answers brief from now on?")

    print("\n--- Assembled memory context (what the agent would see) ---\n")
    print(get_memory_context(DEMO_USER))


if __name__ == "__main__":
    run_demo()
