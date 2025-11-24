from typing import Dict, Any, List, Tuple
from functools import lru_cache
import math

from app.ml.embeddings import generate_embeddings

_CATEGORY_DEFINITIONS: List[Tuple[str, str]] = [
    (
        "politics",
        "News about government, elections, public policy, political parties or politicians.",
    ),
    (
        "economy",
        "Content about economy, finance, markets, inflation, companies and business.",
    ),
    (
        "technology",
        "Articles about software, hardware, AI, gadgets, startups and innovation.",
    ),
    (
        "sports",
        "News about football, soccer, basketball, tennis or other sports and competitions.",
    ),
    (
        "science",
        "Content about science, research, discoveries, health and environment.",
    ),
    (
        "entertainment",
        "Articles about movies, series, music, celebrities and culture.",
    ),
    (
        "other",
        "General content that does not clearly belong to a specific domain.",
    ),
]


def _cosine(a: List[float], b: List[float]) -> float:
    """
    Input:
      - a, b: numeric vectors of the same length.

    Process:
      - Compute cosine similarity between both vectors.

    Output:
      - Similarity score in [-1, 1].
    """
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0

    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


@lru_cache(maxsize=1)
def _get_category_vectors() -> List[Tuple[str, List[float]]]:
    """
    Process:
      - Generate embedding vectors for each category description.
      - Cache results for reuse.

    Output:
      - List of (category_label, embedding_vector).
    """
    vectors: List[Tuple[str, List[float]]] = []
    for label, description in _CATEGORY_DEFINITIONS:
        vec = generate_embeddings(description)
        vectors.append((label, vec))
    return vectors


def run_classification(text: str, title: str | None = None) -> Dict[str, Any]:
    """
    Input:
      - text: main content of the document.
      - title: optional document title or headline.

    Process:
      - Build a combined representation (title + body if title is provided).
      - Encode the document into an embedding vector.
      - Compute cosine similarity against each category vector.
      - Select the category with the highest similarity.
      - Sort all candidates by score.

    Output:
      - Dict with:
        - category: best matching category label.
        - confidence: best similarity score (clipped to [0, 1]).
        - candidates: list of {category, score} for all categories.
        - strategy: description of classification method.
    """
    if title:
        combined = f"{title}\n\n{text}"
    else:
        combined = text

    doc_vec = generate_embeddings(combined)
    category_vectors = _get_category_vectors()

    scores: List[Tuple[str, float]] = []
    for label, vec in category_vectors:
        score = _cosine(doc_vec, vec)
        scores.append((label, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    best_label, best_score = scores[0]
    confidence = max(0.0, min(1.0, best_score))

    return {
        "category": best_label,
        "confidence": confidence,
        "candidates": [
            {"category": label, "score": float(score)} for label, score in scores
        ],
        "strategy": "semantic_category_similarity",
    }
