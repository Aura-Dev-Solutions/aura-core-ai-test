# app/main.py
from fastapi import FastAPI, UploadFile, File, Query
from app.ingestion.pipeline import DocumentProcessor
from app.retrieval.searcher import SemanticSearcher
from app.schemas import SearchResponse
import shutil
import os
from typing import Optional

app = FastAPI(title="Aura Research - Document Analysis System")

processor = DocumentProcessor()
searcher = SemanticSearcher()

UPLOAD_DIR = "/tmp/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = f"{UPLOAD_DIR}/{file.filename}"
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    result = processor.process(file_path)
    return {"message": "Document processed with GLiNER2 NER + Classification", "doc_id": result["doc_id"], "category": result.get("category", "unknown")}

@app.get("/search", response_model=SearchResponse)
async def search(
    q: str = Query(..., min_length=3), 
    limit: int = Query(5, ge=1, le=20),
    entity_filter: Optional[str] = Query(None, description="Filter by entity type, e.g., 'money'"),
    category_filter: Optional[str] = Query(None, description="Filter by doc category, e.g., 'contract'")  # ← NUEVO
):
    results = searcher.search(q, limit=limit, entity_filter=entity_filter, category_filter=category_filter)
    return {"query": q, "results": results}
