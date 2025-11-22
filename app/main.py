# app/main.py
from fastapi import FastAPI, UploadFile, File, Query
from app.tasks import process_document_task, celery_app  
from celery.result import AsyncResult
from app.storage.minio_client import minio_client
from app.retrieval.searcher import SemanticSearcher
from app.retrieval.rag import RAGService
from app.schemas import SearchResponse, ChatRequest, ChatResponse
import uuid
import shutil
import os
from typing import Optional

app = FastAPI(title="Aura Research - Document Analysis System (MinIO + Celery)")

searcher = SemanticSearcher()
rag_service = RAGService()

UPLOAD_DIR = "/tmp/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "aura-api"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    temp_path = f"{UPLOAD_DIR}/{uuid.uuid4()}_{file.filename}"
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    object_name = f"{uuid.uuid4()}_{file.filename}"
    minio_client.upload_file(temp_path, object_name)
    os.unlink(temp_path)

    task = process_document_task.delay(object_name, file.filename)
    return {"task_id": task.id, "filename": file.filename, "status": "en cola"}

@app.get("/status/{task_id}")
async def status(task_id: str):
    task = AsyncResult(task_id, app=celery_app)
    return {
        "task_id": task_id,
        "status": task.state,
        "info": task.info if task.info else None
    }


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
    from app.core.vector_store import QdrantManager
    
    # Instanciamos QdrantManager directamente aquí (o inyectamos dependencia)
    qdrant = QdrantManager()
    chunks = qdrant.get_document_chunks(doc_id)
    
    if not chunks:
        return {"error": "Document not found", "doc_id": doc_id}
    
    # Agregamos metadatos
    first_chunk = chunks[0]
    category = first_chunk.payload.get("metadata", {}).get("doc_category", "unknown")
    filename = first_chunk.payload.get("metadata", {}).get("source", "unknown")
    
    # Consolidar entidades únicas
    all_entities = {}
    for chunk in chunks:
        entities = chunk.payload.get("metadata", {}).get("entities", {})
        for label, values in entities.items():
            if label not in all_entities:
                all_entities[label] = set()
            all_entities[label].update(values)
            
    # Convertir sets a listas para JSON
    consolidated_entities = {k: list(v) for k, v in all_entities.items()}
    
    return {
        "doc_id": doc_id,
        "filename": filename,
        "category": category,
        "total_chunks": len(chunks),
        "entities": consolidated_entities,
        # "chunks": [c.payload for c in chunks] # Opcional: devolver todos los chunks
    }

@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Elimina un documento y sus vectores asociados."""
    from app.core.vector_store import QdrantManager
    
    qdrant = QdrantManager()
    qdrant.delete_document(doc_id)
    
    return {"status": "deleted", "doc_id": doc_id}
