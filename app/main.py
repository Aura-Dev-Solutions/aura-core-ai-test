from fastapi import FastAPI, UploadFile, File, Query
from app.ingestion.pipeline import DocumentProcessor
from app.retrieval.searcher import SemanticSearcher
from app.schemas import SearchResponse
import shutil
import os

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
    return {"message": "Document processed", "doc_id": result["doc_id"]}

@app.get("/search", response_model=SearchResponse)
async def search(q: str = Query(..., min_length=3), limit: int = 5):
    results = searcher.search(q, limit=limit)
    return {"query": q, "results": results}
