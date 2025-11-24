from typing import Any, Dict, Tuple
import io

from docx import Document


def extract(raw: bytes) -> Tuple[str, Dict[str, Any]]:
    """
    Input:
      - raw: raw DOCX bytes.

    Process:
      - Parse the DOCX file using python-docx.
      - Extract text from all paragraphs.
      - Build a basic structure dict (placeholder for future metadata).

    Output:
      - text: extracted plain text.
      - structure: dict with basic document structure info.
    """
    doc = Document(io.BytesIO(raw))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    text = "\n".join(paragraphs)
    structure: Dict[str, Any] = {}
    return text, structure
