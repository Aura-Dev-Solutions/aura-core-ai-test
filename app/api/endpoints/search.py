from fastapi import APIRouter, Query
from app.retrieval.searcher import SemanticSearcher
from app.retrieval.rag import RAGService
from app.schemas import SearchResponse, ChatRequest, ChatResponse
from typing import Optional

router = APIRouter()
searcher = SemanticSearcher()
rag_service = RAGService()

@router.get("/search", response_model=SearchResponse)
async def search(
    q: str = Query(..., min_length=3), 
    limit: int = Query(5, ge=1, le=20),
    entity_filter: Optional[str] = Query(None),
    category_filter: Optional[str] = Query(None)
):
    """Búsqueda sin cambios (usa Qdrant)."""
    results = searcher.search(q, limit=limit, entity_filter=entity_filter, category_filter=category_filter)
    return {"query": q, "results": results}

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Endpoint RAG: Responde preguntas usando el contexto de los documentos."""
    return rag_service.generate_answer(request.query)
