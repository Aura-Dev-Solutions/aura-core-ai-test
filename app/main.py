# app/main.py
from fastapi import FastAPI

from app.api import documents, search
from app.models.repository import init_db

app = FastAPI(title="Document Analysis System")


@app.on_event("startup")
def on_startup() -> None:
    """
      - Initialize the database schema if it does not exist.
    """
    init_db()


# Mount API routers.
app.include_router(documents.router)
app.include_router(search.router)
