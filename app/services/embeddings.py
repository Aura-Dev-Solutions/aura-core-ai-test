from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np
from functools import lru_cache
from app.config import settings

@lru_cache(maxsize=1)
def _load_model():
    return SentenceTransformer(settings.sentence_model)

def embed_texts(texts: List[str]) -> np.ndarray:
    model = _load_model()
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.array(emb, dtype="float32")
