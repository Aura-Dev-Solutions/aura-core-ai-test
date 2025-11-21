from pydantic import BaseModel
from typing import List

class SearchResult(BaseModel):
    text: str
    score: float
    metadata: dict

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]