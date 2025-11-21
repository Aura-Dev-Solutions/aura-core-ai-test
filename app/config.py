from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    qdrant_host: str = "qdrant"
    qdrant_port: int = 6333
    qdrant_collection: str = "document_chunks"
    embedding_model: str = "all-MiniLM-L6-v2"
    
    model_config = {"env_file": ".env"}

settings = Settings()