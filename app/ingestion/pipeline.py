# app/ingestion/pipeline.py
from uuid import uuid4
from app.ingestion.document_loader import DocumentLoader
from app.ingestion.text_splitter import HierarchicalSplitter
from app.ingestion.embedder import Embedder
from app.core.vector_store import QdrantManager
from app.models.ner import get_extractor  # ← IMPORT (lazy)
from qdrant_client.http.models import PointStruct
from loguru import logger
import uuid

class DocumentProcessor:
    def __init__(self):
        self.loader = DocumentLoader()
        self.splitter = HierarchicalSplitter()
        self.embedder = Embedder()
        self.vector_store = QdrantManager()
        self.ner_extractor = None  # ← Lazy init (evita crash en startup)

    def _get_ner_extractor(self):
        """Lazy load NER solo cuando se necesita (primer process)."""
        if self.ner_extractor is None:
            self.ner_extractor = get_extractor(device="cpu")  # ← FIX: Sin threshold
        return self.ner_extractor

    def process(self, file_path: str, doc_id: str = None):
        doc_id = doc_id or str(uuid4())
        logger.info(f"Processing document: {file_path}")

        # 1. Load
        docs = self.loader.load(file_path)
        for doc in docs:
            doc.metadata["doc_id"] = doc_id

        # 2. Split
        chunks = self.splitter.split_documents(docs)

        # 3. NER Enrichment (Lazy: Carga solo aquí, no en __init__)
        enriched_chunks = []
        ner_extractor = self._get_ner_extractor()  # ← Lazy call
        for i, chunk in enumerate(chunks):
            entities = ner_extractor.extract_entities(chunk.page_content)
            chunk.metadata["entities"] = entities  # Enriquecemos metadata
            enriched_chunks.append(chunk)
            logger.debug(f"Chunk {i}: Extracted {len(entities)} entity types")

        # 4. Embed (usa chunks enriquecidos)
        texts = [chunk.page_content for chunk in enriched_chunks]
        vectors = self.embedder.embed_documents(texts)

        # 5. Store con UUID válidos y metadata rica
        points = [
            PointStruct(
                id=str(uuid.uuid4()),  # UUID limpio
                vector=vector,
                payload={
                    "text": chunk.page_content,
                    "metadata": {
                        **chunk.metadata,
                        "doc_id": doc_id,
                        "chunk_index": i
                    }
                }
            )
            for i, (chunk, vector) in enumerate(zip(enriched_chunks, vectors))
        ]
        self.vector_store.upsert(points)
        logger.success(f"Document {doc_id} processed and stored ({len(chunks)} chunks with GLiNER2-large-v1 NER)")
        return {"doc_id": doc_id, "chunks": len(chunks)}