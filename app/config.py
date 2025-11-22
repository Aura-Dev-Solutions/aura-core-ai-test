# app/config.py
from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    # Qdrant
    qdrant_host: str = "qdrant"
    qdrant_port: int = 6333
    qdrant_collection: str = "document_chunks"
    embedding_model: str = "all-MiniLM-L6-v2"
    
    # OpenRouter / LLM
    openrouter_api_key: str = "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"
    openrouter_model: str = "kwaipilot/kat-coder-pro:free"
    
    # Celery
    celery_broker_url: str = "redis://redis:6379/0"
    celery_result_backend: str = "redis://redis:6379/1"
    
    # MinIO
    minio_endpoint: str = "minio:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin123"
    minio_bucket: str = "documents"
    minio_secure: bool = False
    
    model_config = {"env_file": ".env"}

settings = Settings()