from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from app.config import settings
import os

os.makedirs(os.path.dirname(settings.sqlite_path), exist_ok=True)
engine = create_engine(f"sqlite:///{settings.sqlite_path}", future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

def init_db():
    with engine.begin() as conn:
        conn.execute(text('''
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            path TEXT NOT NULL,
            mime TEXT NOT NULL,
            text TEXT
        )''' ))
