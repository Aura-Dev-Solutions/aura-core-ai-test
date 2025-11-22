# app/main.py
from fastapi import FastAPI
from app.api.router import api_router
import os

# Asegurar que existe el directorio de uploads
UPLOAD_DIR = "/tmp/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="Aura Research - Document Analysis System (MinIO + Celery)")

app.include_router(api_router)
