import logging
from typing import Optional, List, Dict, Any

from app.models.statuses import DocumentStatus
from app.models.entities import Document
from app.models import repository
from app.files import storage, extractor_pdf, extractor_docx, extractor_json
from app.ml import embeddings, classifier, ner

logger = logging.getLogger(__name__)

def ingest_document(
    document_id: str,
    filename: str,
    content: bytes,
    content_type: str,
) -> Document:
    """
    Input:
      - document_id: unique ID assigned to the document.
      - filename: original file name.
      - content: raw file bytes.
      - content_type: MIME type of the uploaded file.

    Process:
      - Save raw file to storage.
      - Create initial DB record with PENDING status.
      - Run processing pipeline synchronously.

    """
    logger.info("Ingesting document %s (%s)", document_id, filename)

    storage.save_file(document_id, content)
    repository.create_document(
        document_id=document_id,
        filename=filename,
        content_type=content_type,
        status=DocumentStatus.PENDING,
    )

    process_document(document_id, content_type, title=filename)

    doc = repository.get_document(document_id)
    logger.info("Ingestion completed for document %s", document_id)
    return doc


def process_document(document_id: str, content_type: str, title: str | None = None) -> None:
    """
    Input:
      - document_id: document identifier.
      - content_type: MIME type of the document.
      - title: optional title or filename for better classification.

    Process:
      - Mark document as PROCESSING.
      - Load raw file bytes.
      - Extract text and basic structure according to content type.
      - Generate embeddings.
      - Run semantic classification (topic) using embeddings.
      - Run NER to extract entities.
      - Persist all results.
      - Mark document as DONE or FAILED.

    """
    logger.info("Processing document %s", document_id)
    repository.update_document_status(document_id, DocumentStatus.PROCESSING)

    try:
        raw = storage.load_file(document_id)

        if content_type == "application/pdf":
            text, structure = extractor_pdf.extract(raw)
        elif content_type in (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ):
            text, structure = extractor_docx.extract(raw)
        elif content_type == "application/json":
            text, structure = extractor_json.extract(raw)
        else:
            text, structure = raw.decode("utf-8", errors="ignore"), {}

        repository.save_extracted_text(document_id, text, structure)

        doc_embeddings = embeddings.generate_embeddings(text)
        doc_classification = classifier.run_classification(text, title=title)
        doc_entities = ner.run_ner(text)

        repository.save_embeddings(document_id, doc_embeddings)
        repository.save_classification(document_id, doc_classification)
        repository.save_entities(document_id, doc_entities)

        repository.update_document_status(document_id, DocumentStatus.DONE)
        logger.info("Document %s processed successfully", document_id)

    except Exception as exc:
        logger.exception("Error processing document %s: %s", document_id, exc)
        repository.update_document_status(document_id, DocumentStatus.FAILED)
        raise


def get_document_status(document_id: str) -> Optional[Document]:
    """
    Input:
      - document_id: identifier of the document.

    Process:
      - Fetch the document from the repository.
      - Map DB model to domain entity.

    """
    return repository.get_document(document_id)


def search_documents(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Input:
      - query: free-text search query.
      - limit: maximum number of results to return.

    Process:
      - Generate embedding for the query.
      - Delegate to repository semantic_search, which uses text search now,
        but is shaped to support vector similarity later.

    Output:
      - List of dicts containing search results (document_id, score, filename, snippet).
    """
    logger.info("Semantic search: query='%s', limit=%d", query, limit)
    query_embedding = embeddings.generate_embeddings(query)
    return repository.semantic_search(query_embedding, query, limit=limit)
