# app/retrieval/searcher.py
from app.ingestion.embedder import Embedder
from app.core.vector_store import QdrantManager
from qdrant_client.http.models import Filter, FieldCondition, MatchAny
from typing import Optional

class SemanticSearcher:
    def __init__(self):
        self.embedder = Embedder()
        self.vector_store = QdrantManager()

    def search(
        self, 
        query: str, 
        limit: int = 5, 
        entity_filter: Optional[str] = None
    ):
        query_vector = self.embedder.embed_query(query)
        
        # Filtro por entidades (NUEVO: Usa Qdrant filter para metadata.entities)
        filter_dict = None
        if entity_filter:
            filter_dict = Filter(
                must=[
                    FieldCondition(
                        key="metadata.entities.%s" % entity_filter,
                        match=MatchAny(any=[])  # Existe al menos una entidad de este tipo
                    )
                ]
            )
        
        results = self.vector_store.search(
            query_vector, 
            limit=limit, 
            filter_dict=filter_dict
        )
        return [
            {
                "text": hit.payload["text"],
                "score": float(hit.score),
                "metadata": hit.payload["metadata"],
                "entities": hit.payload["metadata"].get("entities", {})
            }
            for hit in results
        ]