from fastapi import APIRouter, UploadFile, File, HTTPException
from uuid import uuid4

from app.domain import services
from app.models.schemas import DocumentResponse

router = APIRouter(prefix="/documents", tags=["documents"])

@router.post("", response_model=DocumentResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Input:
      - file: uploaded document (PDF, DOCX, JSON, etc.).

    Process:
      - Generate a document ID.
      - Read file bytes.
      - Call domain ingestion service.

    """
    document_id = str(uuid4())
    content = await file.read()

    try:
        doc = services.ingest_document(
            document_id=document_id,
            filename=file.filename,
            content=content,
            content_type=file.content_type or "application/octet-stream",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return doc


@router.get("/{document_id}", response_model=DocumentResponse)
def get_document(document_id: str):
    """
    Input:
      - document_id: identifier of the document.

    Process:
      - Fetch document from domain service.

    """
    doc = services.get_document_status(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc
