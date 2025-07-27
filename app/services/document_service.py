import logging
from fastapi import UploadFile
from app.services.extractor import extract_text
from app.services.embedder import get_embedding
from app.services.classifier import classify
from app.services.ner import extract_entities
from app.repositories.sqlite_repository import DocumentRepository
from app.exceptions import InvalidDocumentFormatError, EmptyDocumentError, DocumentNotFoundError

logger = logging.getLogger(__name__)


class DocumentService:
    def __init__(self, repository: DocumentRepository):
        self.repository = repository

    async def process_document(self, file: UploadFile):
        content = await file.read()
        logger.debug("Extracting text...")
        try:
            text = extract_text(content, file.filename)
        except ValueError as e:
            raise InvalidDocumentFormatError(str(e))

        if not text.strip():
            raise EmptyDocumentError("The document contains no extractable text.")

        logger.debug("Generating embedding...")
        embedding = get_embedding(text)
        logger.debug("Classifying document...")
        category = classify(text)
        logger.debug("Extracting named entities...")
        entities = extract_entities(text)

        logger.debug("Saving document to database...")
        self.repository.insert_document(file.filename, text, embedding, category, entities)

        return {
            "text": text,
            "embedding": embedding,
            "category": category,
            "entities": entities
        }

    def get_document(self, doc_id: int):
        logger.debug("Retrieving document from database...")
        document = self.repository.get_document_by_id(doc_id)
        if "error" in document:
            raise DocumentNotFoundError(f"Document ID {doc_id} not found")

        return document

    def search(self, query: str, top_k: int):
        embedding = get_embedding(query)
        logger.debug("Searching for similar documents...")
        return self.repository.search_similar_documents(embedding, top_k)
