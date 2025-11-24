from typing import List
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.config import EMBEDDINGS_MODEL_NAME

@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """
    Input:
      - None (uses EMBEDDINGS_MODEL_NAME from config).

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
      - text: input text (document or query).

    Process:
      - Encode text into a dense vector using the Sentence-Transformers model.

    Output:
      - Embedding vector as a list of floats.
    """
    model = _get_model()
    vector = model.encode([text])[0]
    return vector.tolist()
