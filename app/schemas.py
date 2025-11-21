# app/schemas.py
from pydantic import BaseModel
from typing import List, Dict, Any

class SearchResult(BaseModel):
    text: str
    score: float
    metadata: Dict[str, Any]  # ← Aquí vienen: entities, doc_category, source, etc.

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]