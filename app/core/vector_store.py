# app/core/vector_store.py
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from app.config import settings
from loguru import logger
import uuid

class QdrantManager:
    def __init__(self):
        self.client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        self.collection_name = settings.qdrant_collection
        self._create_collection_if_not_exists()

    def _create_collection_if_not_exists(self):
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
            logger.info(f"Collection '{self.collection_name}' created")

    def upsert(self, points: list[PointStruct]):
        # Aseguramos que todos los IDs sean UUID válidos
        valid_points = []
        for point in points:
            if not self._is_valid_uuid(point.id):
                # Generamos un UUID limpio
                point.id = str(uuid.uuid4())
            valid_points.append(point)
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=valid_points,
            wait=True
        )

    def search(self, query_vector, limit=5, filter_dict=None):
        return self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            query_filter=filter_dict,
            with_payload=True
        )

    @staticmethod
    def _is_valid_uuid(val):
        try:
            uuid.UUID(str(val))
            return True
        except (ValueError, AttributeError):
            return False