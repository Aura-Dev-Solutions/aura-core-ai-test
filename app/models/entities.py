from dataclasses import dataclass
from typing import Any, Dict, Optional

from app.models.statuses import DocumentStatus

@dataclass
class Document:
    """
      - Acts as an in-memory representation of a document in the domain layer.
    """
    id: str
    filename: str
    content_type: str
    status: DocumentStatus
    text: Optional[str] = None
    structure: Optional[Dict[str, Any]] = None
    classification: Optional[Dict[str, Any]] = None
    entities: Optional[Dict[str, Any]] = None
    embeddings: Optional[Any] = None
