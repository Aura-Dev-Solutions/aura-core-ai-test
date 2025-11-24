from typing import Dict, Any, List
from functools import lru_cache

import spacy

from app.config import NER_MODEL_NAME

@lru_cache(maxsize=1)
def _get_nlp():
    """
    Input:
      - None (uses NER_MODEL_NAME from config).

    Process:
      - Load a spaCy pipeline once and cache it.
      - Raise a clear error if the model cannot be loaded.

    Output:
      - Loaded spaCy Language object.
    """
    model_name = NER_MODEL_NAME or "en_core_web_sm"
    try:
        return spacy.load(model_name)
    except OSError as e:
        raise RuntimeError(
            f"Failed to load spaCy model '{model_name}'. "
            f"Make sure it is installed: python -m spacy download {model_name}"
        ) from e


def run_ner(text: str) -> Dict[str, Any]:
    """
    Input:
      - text: plain text to analyze.

    Process:
      - If the text is empty, return an empty entity list.
      - Otherwise, run NER using the configured spaCy pipeline.

    Output:
      - Dict with a list of entities and basic attributes.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return {"entities": []}

    nlp = _get_nlp()
    doc = nlp(cleaned)
    entities: List[Dict[str, Any]] = [
        {
            "text": ent.text,
            "label": ent.label_,
            "start_char": ent.start_char,
            "end_char": ent.end_char,
        }
        for ent in doc.ents
    ]
    return {"entities": entities}

