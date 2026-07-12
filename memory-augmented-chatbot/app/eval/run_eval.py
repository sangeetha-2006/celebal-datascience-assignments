"""
Step 6c: Evaluation runner.

Runs the full agent (Phase 5) against a test set of questions, scores each
response on context relevance, answer relevance, and faithfulness, and
prints/saves a report.

Usage:
    python -m app.eval.run_eval
"""
import json
from pathlib import Path

from app.agent.build_agent import run_agent_full
from app.eval.metrics import answer_relevance, context_relevance, faithfulness
from app.eval.test_set import TEST_QUESTIONS

RESULTS_PATH = Path("data/eval_results.json")


def _contexts_from_state(state: dict) -> list[str]:
    """Pull out the RAG context lines as a list of strings for scoring.
    (kg_context/tool_context aren't embeddings-friendly free text in the
    same way, so context_relevance/faithfulness focus on the RAG context —
    the part that's meant to ground the answer in retrieved documents.)
    """
    rag_context = state.get("rag_context", "")
    if not rag_context or rag_context.startswith("RAG index not built") or rag_context.startswith("No relevant"):
        return []
    return [line.strip("- ").strip() for line in rag_context.split("\n") if line.strip()]


def evaluate_question(user_id: str, question: str) -> dict:
    state = run_agent_full(user_id, question)
    answer = state["answer"]
    contexts = _contexts_from_state(state)

    scores = {
        "question": question,
        "answer": answer,
        "context_relevance": round(context_relevance(question, contexts), 3),
        "answer_relevance": round(answer_relevance(question, answer), 3),
        "faithfulness": round(faithfulness(answer, contexts), 3) if contexts else None,
        "num_contexts_used": len(contexts),
    }
    return scores


def run_evaluation(user_id: str = "eval_user") -> list[dict]:
    results = []
    for i, question in enumerate(TEST_QUESTIONS, start=1):
        print(f"[eval] ({i}/{len(TEST_QUESTIONS)}) evaluating: {question}")
        scores = evaluate_question(user_id, question)
        results.append(scores)
        print(
            f"  context_relevance={scores['context_relevance']} "
            f"answer_relevance={scores['answer_relevance']} "
            f"faithfulness={scores['faithfulness']}"
        )

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[eval] full results saved -> {RESULTS_PATH}")

    _print_summary(results)
    return results


def _print_summary(results: list[dict]):
    def avg(key):
        values = [r[key] for r in results if r[key] is not None]
        return round(sum(values) / len(values), 3) if values else None

    print("\n=== Evaluation Summary ===")
    print(f"Questions evaluated:     {len(results)}")
    print(f"Avg context relevance:   {avg('context_relevance')}")
    print(f"Avg answer relevance:    {avg('answer_relevance')}")
    print(f"Avg faithfulness:        {avg('faithfulness')} (only over questions where RAG context was used)")


if __name__ == "__main__":
    run_evaluation()
