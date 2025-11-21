from uuid import uuid4
from app.ingestion.document_loader import DocumentLoader
from app.ingestion.text_splitter import HierarchicalSplitter
from app.ingestion.embedder import Embedder
from app.core.vector_store import QdrantManager
from qdrant_client.http.models import PointStruct
from loguru import logger
import uuid

class DocumentProcessor:
    def __init__(self):
        self.loader = DocumentLoader()
        self.splitter = HierarchicalSplitter()
        self.embedder = Embedder()
        self.vector_store = QdrantManager()

    def process(self, file_path: str, doc_id: str = None):
        doc_id = doc_id or str(uuid4())
        logger.info(f"Processing document: {file_path}")

        # 1. Load
        docs = self.loader.load(file_path)
        for doc in docs:
            doc.metadata["doc_id"] = doc_id

        # 2. Split
        chunks = self.splitter.split_documents(docs)

        # 3. Embed
        texts = [chunk.page_content for chunk in chunks]
        vectors = self.embedder.embed_documents(texts)

        # 4. Store
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "text": chunk.page_content,
                    "metadata": chunk.metadata
                }
            )
            for i, (chunk, vector) in enumerate(zip(chunks, vectors))
        ]
        self.vector_store.upsert(points)
        logger.success(f"Document {doc_id} processed and stored ({len(chunks)} chunks)")
        return {"doc_id": doc_id, "chunks": len(chunks)}