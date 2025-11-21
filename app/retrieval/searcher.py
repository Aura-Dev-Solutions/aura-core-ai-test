from app.ingestion.embedder import Embedder
from app.core.vector_store import QdrantManager

class SemanticSearcher:
    def __init__(self):
        self.embedder = Embedder()
        self.vector_store = QdrantManager()

    def search(self, query: str, limit: int = 5, filter_dict=None):
        query_vector = self.embedder.embed_query(query)
        results = self.vector_store.search(query_vector, limit=limit, filter_dict=filter_dict)
        return [
            {
                "text": hit.payload["text"],
                "score": hit.score,
                "metadata": hit.payload["metadata"]
            }
            for hit in results
        ]