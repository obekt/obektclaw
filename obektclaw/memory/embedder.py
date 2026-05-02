"""Local embedding generator using sentence-transformers.

Provides a singleton embedder for generating vector embeddings from text.
Uses all-MiniLM-L6-v2 model (384 dimensions, ~80MB).
"""

from __future__ import annotations

from typing import Optional

# Lazy import to avoid loading model at import time
_model: Optional[object] = None


def _get_model():
    """Lazy-load the sentence-transformers model."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def embed(text: str) -> list[float]:
    """Generate a vector embedding for a single text.

    Args:
        text: The text to embed.

    Returns:
        A list of floats representing the embedding vector (384 dimensions).
    """
    model = _get_model()
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Generate vector embeddings for multiple texts.

    Args:
        texts: List of texts to embed.

    Returns:
        List of embedding vectors, one per input text.
    """
    model = _get_model()
    embeddings = model.encode(texts, convert_to_numpy=True)
    return [e.tolist() for e in embeddings]


def get_embedding_dimension() -> int:
    """Return the dimension of embeddings produced by this model.

    Returns:
        384 for all-MiniLM-L6-v2.
    """
    return 384
