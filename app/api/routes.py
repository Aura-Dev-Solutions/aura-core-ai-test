from fastapi import APIRouter, UploadFile, File, BackgroundTasks
from pathlib import Path
import uuid, os
from app.config import settings
from app.storage.db import SessionLocal
from app.services.extract import extract_text
from app.services.index import get_index
from app.services.search import semantic_search
from app.services.classify import classify_text
from sqlalchemy import text
from app.models import (
    IngestResponse, SearchRequest, SearchResult, 
    ClassifyRequest, ClassifyResponse
)

router = APIRouter()

@router.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    os.makedirs(settings.docs_storage_path, exist_ok=True)
    doc_id = str(uuid.uuid4())
    dest = Path(settings.docs_storage_path) / f"{doc_id}_{file.filename}"
    with dest.open("wb") as f:
        f.write(await file.read())

    doc_text, mime = extract_text(dest)

    db = SessionLocal()
    db.execute(
        text("INSERT INTO documents (id, path, mime, text) VALUES (:id, :path, :mime, :text)"),
        {"id": doc_id, "path": str(dest), "mime": mime, "text": doc_text}
    )
    db.commit()
    db.close()

    idx = get_index()
    idx.add(doc_id, doc_text)
    if background_tasks:
        background_tasks.add_task(idx.build)

    return IngestResponse(doc_id=doc_id, status="queued")

@router.post("/search", response_model=list[SearchResult])
async def search(req: SearchRequest):
    return semantic_search(req.query, top_k=req.top_k)

@router.post("/classify", response_model=ClassifyResponse)
async def classify(req: ClassifyRequest):
    db = SessionLocal()
    row = db.execute(text("SELECT text FROM documents WHERE id = :id"), {"id": req.doc_id}).fetchone()
    db.close()
    doc_text = row[0] if row else ""
    labels, scores = classify_text(doc_text)
    return ClassifyResponse(doc_id=req.doc_id, labels=labels, scores=scores)
