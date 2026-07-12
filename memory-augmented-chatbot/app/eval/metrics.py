"""
Step 6a: Evaluation metrics.

All metrics here are free — no paid API required:
- context_relevance / answer_relevance: cosine similarity using the same
  local embedding model from Phase 2 (sentence-transformers)
- faithfulness: uses your local Ollama model (Phase 5) as a free "LLM judge"
  to check whether the answer is actually supported by the retrieved context

These mirror what a tool like RAGAS measures, implemented locally so the
whole project stays free end-to-end.
"""
import re

import numpy as np

from app.agent.llm import generate
from app.rag.embedder import embed_query, embed_texts


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


def context_relevance(query: str, contexts: list[str]) -> float:
    """Average cosine similarity between the query and each retrieved chunk.
    Higher = the retriever pulled back genuinely relevant material.
    Returns a score from -1 to 1 (in practice usually 0 to 1).
    """
    if not contexts:
        return 0.0
    query_vec = embed_query(query)
    context_vecs = embed_texts(contexts)
    sims = [_cosine_similarity(query_vec, c) for c in context_vecs]
    return float(np.mean(sims))


def answer_relevance(query: str, answer: str) -> float:
    """Cosine similarity between the query and the generated answer.
    Higher = the answer is actually on-topic for what was asked.
    """
    query_vec = embed_query(query)
    answer_vec = embed_query(answer)
    return _cosine_similarity(query_vec, answer_vec)


FAITHFULNESS_PROMPT = """You are grading whether an AI-generated answer is faithful to (supported by) \
the given context. Faithful means every factual claim in the answer can be traced back to the context \
— not that the answer must use every part of the context.

Context:
{context}

Answer to grade:
{answer}

On a scale from 0 to 1 (0 = completely unsupported/made up, 1 = fully supported by the context), \
how faithful is this answer to the context?

Respond with ONLY a single number between 0 and 1 (e.g. "0.8"). No other text.
"""


def faithfulness(answer: str, contexts: list[str]) -> float:
    """Use the local LLM (Ollama) as a judge: is the answer actually
    supported by the retrieved context, or is it making things up?
    Returns a score from 0 to 1 (falls back to 0.5 if the judge's
    response can't be parsed as a number).
    """
    context_block = "\n".join(contexts) if contexts else "(no context was provided)"
    prompt = FAITHFULNESS_PROMPT.format(context=context_block, answer=answer)

    judge_response = generate(prompt)
    match = re.search(r"(\d*\.?\d+)", judge_response)
    if not match:
        return 0.5  # neutral fallback if the judge didn't return a parseable number

    score = float(match.group(1))
    return max(0.0, min(1.0, score))  # clamp to [0, 1] in case the model goes rogue
