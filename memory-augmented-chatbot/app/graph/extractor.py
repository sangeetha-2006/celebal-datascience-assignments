"""
Step 3b: Entity & relationship extraction (free, offline, no API key).

Uses spaCy's dependency parser to find subject-verb-object patterns in text
and turns them into (entity, relation, entity) triples for the knowledge graph.

This is a lightweight heuristic approach (not as accurate as an LLM), but it
runs entirely locally with no cost and no API key required.

First use downloads the small English model (~15MB) via:
    python -m spacy download en_core_web_sm
"""
from functools import lru_cache

# Verbs/relations extracted from these dependency labels
SUBJECT_DEPS = {"nsubj", "nsubjpass"}
OBJECT_DEPS = {"dobj", "pobj", "attr", "oprd"}


@lru_cache(maxsize=1)
def _get_nlp():
    import spacy

    try:
        return spacy.load("en_core_web_sm")
    except OSError as e:
        raise RuntimeError(
            "spaCy model 'en_core_web_sm' not found. Install it with:\n"
            "    python -m spacy download en_core_web_sm"
        ) from e


def _expand_to_noun_chunk(token, doc) -> str:
    """Expand a single token to its full noun phrase if it's part of one,
    e.g. 'Generation' -> 'Retrieval-Augmented Generation'.
    """
    for chunk in doc.noun_chunks:
        if chunk.start <= token.i < chunk.end:
            return chunk.text
    return token.text


def extract_triples(text: str, max_chars: int = 20000, max_triples: int = 200) -> list[dict]:
    """Extract (source, relation, target) triples from text using
    dependency-parsed subject-verb-object patterns.
    """
    nlp = _get_nlp()
    truncated = text[:max_chars]
    doc = nlp(truncated)

    triples = []
    for sent in doc.sents:
        for token in sent:
            if token.pos_ != "VERB":
                continue

            subjects = [w for w in token.lefts if w.dep_ in SUBJECT_DEPS]
            objects = [w for w in token.rights if w.dep_ in OBJECT_DEPS]

            # Also follow verb -> preposition -> object-of-preposition chains
            # (e.g. "used for storing knowledge graphs" -> object = "graphs")
            for right in token.rights:
                if right.dep_ == "prep":
                    for child in right.rights:
                        if child.dep_ == "pobj":
                            objects.append(child)

            if not subjects or not objects:
                continue

            for s in subjects:
                for o in objects:
                    source = _expand_to_noun_chunk(s, doc).strip()
                    target = _expand_to_noun_chunk(o, doc).strip()
                    if source and target and source.lower() != target.lower():
                        triples.append({
                            "source": source,
                            "relation": token.lemma_,
                            "target": target,
                        })

            if len(triples) >= max_triples:
                return triples

    return triples
