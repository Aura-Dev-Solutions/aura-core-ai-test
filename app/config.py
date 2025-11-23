from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    # Qdrant
    qdrant_host: str 
    qdrant_port: int 
    qdrant_collection: str 
    embedding_model: str 
    
    # OpenRouter / LLM
    openrouter_api_key: str
    openrouter_model: str 
    
    # Celery
    celery_broker_url: str 
    celery_result_backend: str 
    
    # MinIO
    minio_endpoint: str 
    minio_access_key: str
    minio_secret_key: str
    minio_bucket: str 
    minio_secure: bool 
    
    model_config = {"env_file": ".env"}

settings = Settings()