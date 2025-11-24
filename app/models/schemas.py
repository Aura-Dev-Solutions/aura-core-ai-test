from typing import Any, Dict, List, Optional
from pydantic import BaseModel

from app.models.statuses import DocumentStatus

class DocumentResponse(BaseModel):
    """
      - Serialize document data as JSON for API responses.
    """
    id: str
    filename: str
    content_type: str
    status: DocumentStatus
    text: Optional[str] = None
    structure: Optional[Dict[str, Any]] = None
    classification: Optional[Dict[str, Any]] = None
    entities: Optional[Dict[str, Any]] = None

    class Config:
        orm_mode = True


class SearchResult(BaseModel):
    """
      - Normalize fields to a stable schema.
    """
    document_id: str
    score: float
    filename: str
    snippet: Optional[str] = None


class SearchResponse(BaseModel):
    """
      - Wrap results into a single response object.
    """
    query: str
    results: List[SearchResult]
