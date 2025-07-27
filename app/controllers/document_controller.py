import logging
from fastapi import UploadFile, HTTPException
from app.services.document_service import DocumentService
from app.exceptions import InvalidDocumentFormatError, EmptyDocumentError, DocumentNotFoundError

logger = logging.getLogger(__name__)


class DocumentController:
    def __init__(self, service: DocumentService):
        self.service = service

    async def process_document(self, file: UploadFile):
        logger.info(f"Processing document: {file.filename}")
        try:
            return await self.service.process_document(file)
        except InvalidDocumentFormatError as e:
            logger.warning(f"Unsupported document format: {e}")
            raise HTTPException(status_code=415, detail=str(e))
        except EmptyDocumentError as e:
            logger.warning(f"Document is empty: {e}")
            raise HTTPException(status_code=422, detail=str(e))
        except Exception as e:
            logger.error(f"Error in process_document: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Unexpected error during document processing")

    def get_document(self, doc_id: int):
        try:
            return self.service.get_document(doc_id)
        except DocumentNotFoundError as e:
            logger.warning(f"{e}")
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logger.error(f"Error in get_document: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Unexpected error retrieving document")

    def search(self, query: str, top_k: int):
        try:
            return self.service.search(query, top_k)
        except Exception as e:
            logger.error(f"Error in search: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Unexpected error during search")
