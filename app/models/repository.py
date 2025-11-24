from typing import Any, Dict, List, Optional

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.config import DATABASE_URL
from app.models.orm import Base, DocumentORM
from app.models.entities import Document
from app.models.statuses import DocumentStatus

engine = create_engine(DATABASE_URL, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def init_db() -> None:
    """
    Create all tables defined by the ORM models.
    """
    Base.metadata.create_all(bind=engine)


def _to_domain(model: DocumentORM) -> Document:
    """
    Input:
      - DocumentORM instance.

    Process:
      - Map ORM fields to domain Document fields.

    Output:
      - Domain Document entity.
    """
    return Document(
        id=model.id,
        filename=model.filename,
        content_type=model.content_type,
        status=DocumentStatus(model.status),
        text=model.text,
        structure=model.structure,
        classification=model.classification,
        entities=model.entities,
        embeddings=model.embeddings,
    )


def create_document(
    document_id: str,
    filename: str,
    content_type: str,
    status: DocumentStatus,
) -> None:
    """
    Input:
      - document_id, filename, content_type, status.

    Process:
      - Insert a new row into the documents table.
    """
    with SessionLocal() as session:
        db_doc = DocumentORM(
            id=document_id,
            filename=filename,
            content_type=content_type,
            status=status.value,
        )
        session.add(db_doc)
        session.commit()


def update_document_status(document_id: str, status: DocumentStatus) -> None:
    """
    Input:
      - document_id: identifier of the document.
      - status: new status value.

    Process:
      - Load document by ID and update its status field.
    """
    with SessionLocal() as session:
        doc = session.get(DocumentORM, document_id)
        if not doc:
            return
        doc.status = status.value
        session.commit()


def save_extracted_text(document_id: str, text: str, structure: Dict[str, Any]) -> None:
    """
    Input:
      - document_id: identifier of the document.
      - text: extracted plain text.
      - structure: extracted structure (e.g. sections, metadata).

    Process:
      - Update document text and structure fields.
    """
    with SessionLocal() as session:
        doc = session.get(DocumentORM, document_id)
        if not doc:
            return
        doc.text = text
        doc.structure = structure
        session.commit()


def save_embeddings(document_id: str, emb: Any) -> None:
    """
    Input:
      - document_id: identifier of the document.
      - emb: embedding vector.

    Process:
      - Store embeddings as JSON under the 'embeddings' column.
    """
    with SessionLocal() as session:
        doc = session.get(DocumentORM, document_id)
        if not doc:
            return
        doc.embeddings = {"vector": emb}
        session.commit()


def save_classification(document_id: str, classification: Dict[str, Any]) -> None:
    """
    Input:
      - document_id: identifier of the document.
      - classification: classification result dict.

    Process:
      - Update classification column.
    """
    with SessionLocal() as session:
        doc = session.get(DocumentORM, document_id)
        if not doc:
            return
        doc.classification = classification
        session.commit()


def save_entities(document_id: str, entities: Dict[str, Any]) -> None:
    """
    Input:
      - document_id: identifier of the document.
      - entities: NER result dict.

    Process:
      - Update entities column.
    """
    with SessionLocal() as session:
        doc = session.get(DocumentORM, document_id)
        if not doc:
            return
        doc.entities = entities
        session.commit()


def get_document(document_id: str) -> Optional[Document]:
    """
    Input:
      - document_id: identifier of the document.

    Process:
      - Load ORM instance by ID.
      - Convert to domain Document.

    Output:
      - Domain Document or None.
    """
    with SessionLocal() as session:
        doc = session.get(DocumentORM, document_id)
        if not doc:
            return None
        return _to_domain(doc)


def semantic_search(
    query_embedding: Any,
    query_text: str,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """
    Input:
      - query_embedding: embedding vector for the query (not used yet).
      - query_text: raw query text.
      - limit: maximum number of results.

    Process:
      - Perform a LIKE-based search over the text column for now.
      - Compute a simple snippet around the first match.

    Output:
      - List of dicts: {document_id, score, filename, snippet}.
    """
    pattern = f"%{query_text}%"
    results: List[Dict[str, Any]] = []

    with SessionLocal() as session:
        stmt = (
            select(DocumentORM)
            .where(DocumentORM.text.ilike(pattern))
            .limit(limit)
        )
        for row in session.execute(stmt).scalars():
            snippet = None
            if row.text:
                idx = row.text.lower().find(query_text.lower())
                if idx != -1:
                    start = max(0, idx - 40)
                    end = min(len(row.text), idx + 40)
                    snippet = row.text[start:end]
            results.append(
                {
                    "document_id": row.id,
                    "score": 1.0,  
                    "filename": row.filename,
                    "snippet": snippet,
                }
            )

    return results
