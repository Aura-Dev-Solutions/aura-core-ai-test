# app/main.py
from fastapi import FastAPI, UploadFile, File, Query
from app.tasks import process_document_task, celery_app  # FIX: Import celery_app para AsyncResult
from celery.result import AsyncResult
from app.storage.minio_client import minio_client
from app.retrieval.searcher import SemanticSearcher
from app.schemas import SearchResponse
import uuid
import shutil
import os
from typing import Optional

app = FastAPI(title="Aura Research - Document Analysis System (MinIO + Celery)")

searcher = SemanticSearcher()

UPLOAD_DIR = "/tmp/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

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
