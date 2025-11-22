# app/schemas.py
from pydantic import BaseModel
from typing import List, Dict, Any

class SearchResult(BaseModel):
    text: str
    score: float
    metadata: Dict[str, Any]  

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]