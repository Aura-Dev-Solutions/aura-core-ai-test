from app.services.index import get_index

def semantic_search(query: str, top_k: int = 5):
    idx = get_index()
    return idx.search(query, top_k=top_k)
