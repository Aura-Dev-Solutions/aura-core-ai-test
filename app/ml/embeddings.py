from typing import List
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.config import EMBEDDINGS_MODEL_NAME

_MAX_CHARS = 10_000

@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """
    Process:
      - Load Sentence-Transformers model once and cache it.

    Output:
      - Loaded SentenceTransformer instance.
    """
    model_name = EMBEDDINGS_MODEL_NAME or "sentence-transformers/all-MiniLM-L6-v2"
    return SentenceTransformer(model_name)


def generate_embeddings(text: str) -> List[float]:
    """
    Input:
      - text: raw text to encode as a vector.

    Process:
      - Normalize and trim input.
      - Short-circuit empty input.
      - Optionally truncate very long texts.
      - Encode using SentenceTransformer.

    Output:
      - List of floats representing the embedding vector.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return []

    if len(cleaned) > _MAX_CHARS:
        cleaned = cleaned[:_MAX_CHARS]

    model = _get_model()
    vector = model.encode(cleaned)
    # Ensure we always return a plain Python list
    try:
        return vector.tolist()
    except AttributeError:
        return list(map(float, vector))
