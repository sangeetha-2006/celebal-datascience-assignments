"""
Step 2a: Embedding generation.

Uses a local sentence-transformers model (no API key required).
Default model: all-MiniLM-L6-v2 (384-dim, fast, good quality for its size).

The model is downloaded once from Hugging Face and cached locally
(~/.cache/huggingface) on first use, so the very first run needs
internet access; subsequent runs work offline.
"""
from functools import lru_cache

import numpy as np

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _get_model(model_name: str = DEFAULT_MODEL_NAME):
    """Load (and cache) the sentence-transformers model.
    Cached so repeated calls don't reload the model from disk each time.
    """
    from sentence_transformers import SentenceTransformer

    print(f"[embedder] loading model '{model_name}' (first run downloads it)...")
    return SentenceTransformer(model_name)


def embed_texts(texts: list[str], model_name: str = DEFAULT_MODEL_NAME, batch_size: int = 32) -> np.ndarray:
    """Embed a list of texts. Returns an (n, dim) float32 numpy array,
    L2-normalized so inner product == cosine similarity.
    """
    model = _get_model(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # so we can use cosine similarity via dot product
    )
    return embeddings.astype("float32")


def embed_query(query: str, model_name: str = DEFAULT_MODEL_NAME) -> np.ndarray:
    """Embed a single query string. Returns a (dim,) float32 numpy array."""
    return embed_texts([query], model_name=model_name)[0]


def get_embedding_dim(model_name: str = DEFAULT_MODEL_NAME) -> int:
    model = _get_model(model_name)
    return model.get_sentence_embedding_dimension()
