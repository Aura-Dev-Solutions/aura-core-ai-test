from pydantic import BaseModel
from typing import Optional, List

class IngestResponse(BaseModel):
    doc_id: str
    status: str

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

class SearchResult(BaseModel):
    doc_id: str
    score: float
    text_snippet: str

class ClassifyRequest(BaseModel):
    doc_id: str

class ClassifyResponse(BaseModel):
    doc_id: str
    labels: List[str]
    scores: List[float]

