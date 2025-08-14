from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.api.routes import router
from app.storage.db import init_db
from app.services.index import get_index

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    get_index().load()
    yield

def create_app() -> FastAPI:
    app = FastAPI(title="DocAI Pipeline", lifespan=lifespan)
    app.include_router(router, prefix="/api")
    return app

app = create_app()