# app/main.py
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from app.retrieval.searcher import SemanticSearcher
from app.retrieval.rag import RAGService
from app.schemas import SearchResponse, ChatRequest, ChatResponse
from app.services.document_service import DocumentService
from typing import Optional

app = FastAPI(title="Aura Research - Document Analysis System (MinIO + Celery)")

# Instanciamos servicios
searcher = SemanticSearcher()
rag_service = RAGService()
document_service = DocumentService()

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "aura-api"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    return await document_service.upload_document(file)

@app.get("/status/{task_id}")
async def status(task_id: str):
    return document_service.get_task_status(task_id)

@app.get("/search", response_model=SearchResponse)
async def search(
    q: str = Query(..., min_length=3), 
    limit: int = Query(5, ge=1, le=20),
    entity_filter: Optional[str] = Query(None),
    category_filter: Optional[str] = Query(None)
):
    """Búsqueda sin cambios (usa Qdrant)."""
    results = searcher.search(q, limit=limit, entity_filter=entity_filter, category_filter=category_filter)
    return {"query": q, "results": results}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Endpoint RAG: Responde preguntas usando el contexto de los documentos."""
    return rag_service.generate_answer(request.query)

@app.get("/documents/{doc_id}")
async def get_document_details(doc_id: str):
    """Recupera los detalles de un documento procesado (categoría, entidades, chunks)."""
    details = document_service.get_document_details(doc_id)
    if not details:
        raise HTTPException(status_code=404, detail="Document not found")
    return details

@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Elimina un documento y sus vectores asociados."""
    return document_service.delete_document(doc_id)
