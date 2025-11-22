from fastapi import APIRouter
from app.api.endpoints import documents, search, system

api_router = APIRouter()

api_router.include_router(system.router, tags=["system"])
api_router.include_router(documents.router, tags=["documents"])
api_router.include_router(search.router, tags=["search"])
