from typing import Any, Dict, Tuple

def extract(raw: bytes) -> Tuple[str, Dict[str, Any]]:
    """
    Input:
      - raw: raw DOCX bytes (simplified, treated as UTF-8 text here).

    Process:
      - Decode bytes as UTF-8.
      - Build a basic empty structure dict.

    Output:
      - text: extracted plain text.
      - structure: empty dict (placeholder for future layout info).
    """
    text = raw.decode("utf-8", errors="ignore")
    structure: Dict[str, Any] = {}
    return text, structure
