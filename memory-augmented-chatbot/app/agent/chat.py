"""
Step 5e: Interactive chat CLI.

Ties together everything built so far: memory, RAG, knowledge graph,
dynamic tools, and Gemini for response generation.

Usage:
    python -m app.agent.chat
    (type 'exit' to quit)
"""
from app.agent.build_agent import build_agent

DEFAULT_USER_ID = "cli_user"


def main():
    print("=== Memory-Augmented Chatbot (type 'exit' to quit) ===\n")
    app = build_agent()

    while True:
        query = input("You: ").strip()
        if query.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        if not query:
            continue

        result = app.invoke({"user_id": DEFAULT_USER_ID, "query": query})
        print(f"\nBot: {result['answer']}\n")


if __name__ == "__main__":
    main()
