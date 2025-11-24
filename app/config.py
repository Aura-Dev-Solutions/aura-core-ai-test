# app/config.py
import logging
import os

"""
Global configuration and logging setup.
"""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)

DB_HOST = os.getenv("DB_HOST", "db")
DB_PORT = int(os.getenv("DB_PORT", 5432))
DB_NAME = os.getenv("DB_NAME", "aura_core")
DB_USER = os.getenv("DB_USER", "core_ai")
DB_PASSWORD = os.getenv("DB_PASSWORD", "core_ai")

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
)

INBOX_PATH = os.getenv("INBOX_PATH", "/data/inbox")

GATEWAY_API_URL = os.getenv("GATEWAY_API_URL", "http://app:8000")

NER_MODEL_NAME = os.getenv("NER_MODEL_NAME", "en_core_web_sm")
EMBEDDINGS_MODEL_NAME = os.getenv(
    "EMBEDDINGS_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
)
