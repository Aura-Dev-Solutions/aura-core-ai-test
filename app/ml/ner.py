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

    Output:
      - Loaded spaCy Language object.
    """
    model_name = NER_MODEL_NAME or "en_core_web_sm"
    return spacy.load(model_name)


def run_ner(text: str) -> Dict[str, Any]:
    """
    Input:
      - text: input text to analyze.

    Process:
      - Run spaCy NER over the text.
      - Collect entities with text, label and character offsets.

    Output:
      - Dict with key 'entities' containing a list of entities.
    """
    nlp = _get_nlp()
    doc = nlp(text)

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
