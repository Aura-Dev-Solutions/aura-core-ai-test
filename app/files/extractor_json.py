import json
from typing import Any, Dict, Tuple, List

def extract(raw: bytes) -> Tuple[str, Dict[str, Any]]:
    """
    Input:
      - raw: raw JSON bytes.

    Process:
      - Parse JSON into a Python structure.
      - Collect all string values into a single text blob.
      - Keep original JSON as structure.

    Output:
      - text: concatenated string values.
      - structure: original JSON content as dict.
    """
    data = json.loads(raw.decode("utf-8", errors="ignore"))

    parts: List[str] = []

    def _collect(obj: Any) -> None:
        if isinstance(obj, dict):
            for v in obj.values():
                _collect(v)
        elif isinstance(obj, list):
            for v in obj:
                _collect(v)
        elif isinstance(obj, str):
            parts.append(obj)

    _collect(data)
    text = "\n".join(parts)
    structure: Dict[str, Any] = data
    return text, structure
