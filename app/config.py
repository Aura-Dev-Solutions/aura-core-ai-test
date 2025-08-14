from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    docs_storage_path: str = Field(default="/data/docs", alias="DOCS_STORAGE_PATH")
    embeddings_path: str = Field(default="/data/faiss.index", alias="EMBEDDINGS_PATH")
    sqlite_path: str = Field(default="/data/docai.db", alias="SQLITE_PATH")
    sentence_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", alias="SENTENCE_MODEL")
    max_workers: int = Field(default=4, alias="MAX_WORKERS")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "populate_by_name": True,
    }

settings = Settings()
