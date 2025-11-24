from fastapi import APIRouter, HTTPException, Query

from app.domain import services
from app.models.schemas import SearchResponse

router = APIRouter(tags=["search"])

@router.get("/search", response_model=SearchResponse)
def search(q: str = Query(..., description="Search query"), limit: int = 10):
    """
    Input:
      - q: free-text search query.
      - limit: maximum number of results to return.

    Process:
      - Call domain search service to perform semantic-like search.

    Output:
      - SearchResponse containing list of results with scores.
    """
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty")

    results = services.search_documents(query=q, limit=limit)
    return SearchResponse(query=q, results=results)
