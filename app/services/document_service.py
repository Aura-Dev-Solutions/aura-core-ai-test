import logging
from fastapi import UploadFile
from prometheus_client import Counter, Histogram
from starlette.concurrency import run_in_threadpool
from app.services.extractor import extract_text
from app.services.embedder import get_embedding
from app.services.classifier import classify
from app.services.ner import extract_entities
from app.repositories.sqlite_repository import DocumentRepository
from app.exceptions import InvalidDocumentFormatError, EmptyDocumentError, DocumentNotFoundError

logger = logging.getLogger(__name__)
# Métricas
documents_processed = Counter("documents_processed_total", "Número de documentos procesados")
documents_processed_with_error = Counter("documents_processed_error", "Número de documentos con error sin procesar")
processing_time = Histogram("document_processing_seconds", "Tiempo en segundos para procesar un documento")
getting_time = Histogram("getting_document_seconds", "Tiempo en segundos en obtener un documento")
getting_document = Counter("getting_documents_total", "Número de documentos obtenidos")
getting_document_with_error = Counter("getting_documents_error", "Número de documentos que se quisieron obtener")
searching_time = Histogram("searching_document_seconds", "Tiempo en segundos en buscar un documento")
searching_document = Counter("searching_document_total", "Número de documentos buscados")


class DocumentService:
    def __init__(self, repository: DocumentRepository):
        self.repository = repository

    @processing_time.time()
    async def process_document(self, file: UploadFile):
        content = await file.read()
        logger.debug("Extracting text...")
        try:
            text = await run_in_threadpool(extract_text, content, file.filename)
        except ValueError as e:
            documents_processed_with_error.inc()
            raise InvalidDocumentFormatError(str(e))

        if not text.strip():
            documents_processed_with_error.inc()
            raise EmptyDocumentError("The document contains no extractable text.")

        logger.debug("Generating embedding...")
        embedding = await run_in_threadpool(get_embedding, text)
        logger.debug("Classifying document...")
        category = await run_in_threadpool(classify, text)
        logger.debug("Extracting named entities...")
        entities = await run_in_threadpool(extract_entities, text)

        logger.debug("Saving document to database...")
        await run_in_threadpool(self.repository.insert_document, file.filename, text, embedding, category, entities)

        documents_processed.inc()
        return {
            "text": text,
            "embedding": embedding,
            "category": category,
            "entities": entities
        }

    @getting_time.time()
    def get_document(self, doc_id: int):
        logger.debug("Retrieving document from database...")
        document = self.repository.get_document_by_id(doc_id)
        if "error" in document:
            getting_document_with_error.inc()
            raise DocumentNotFoundError(f"Document ID {doc_id} not found")

        getting_document.inc()
        return document

    @searching_time.time()
    def search(self, query: str, top_k: int):
        embedding = get_embedding(query)
        logger.debug("Searching for similar documents...")
        searching_document.inc()
        return self.repository.search_similar_documents(embedding, top_k)
