"""
Step 6b: Evaluation test set.

A small set of questions to run the agent against and score. Tailored to
the example dataset (RAG + Knowledge Graph Wikipedia pages) — replace or
extend this list with questions relevant to whatever you've scraped.
"""

TEST_QUESTIONS = [
    "What is retrieval-augmented generation?",
    "How does RAG reduce hallucination in language models?",
    "What is a knowledge graph?",
    "What are entities and relationships in a knowledge graph?",
    "What time is it right now?",  # exercises the tool-routing path, not RAG/KG
]
