from fastapi import APIRouter, UploadFile, File, Query
from app.controllers.document_controller import DocumentController
from app.services.document_service import DocumentService
from app.repositories.sqlite_repository import DocumentRepository
from app.db.database import DB_PATH


def get_routes():
    router = APIRouter()
    repo = DocumentRepository(DB_PATH)
    service = DocumentService(repo)
    controller = DocumentController(service)

    @router.post("/process")
    async def process_document(file: UploadFile = File(...)):
        return await controller.process_document(file)

    @router.get("/document/{doc_id}")
    def get_document(doc_id: int):
        return controller.get_document(doc_id)

    @router.get("/search")
    def search(q: str = Query(...), top_k: int = 5):
        return controller.search(q, top_k)

    return router
