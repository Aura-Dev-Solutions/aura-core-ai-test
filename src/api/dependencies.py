"""
API dependencies for dependency injection.
Provides singleton instances of services and processors.
"""

from functools import lru_cache
from typing import Optional

from src.document_processor.processor import DocumentProcessor
from src.ai_models.ai_pipeline import AIAnalysisPipeline
from src.ai_models.semantic_search import SemanticSearchService
from src.core.config import settings


# Global instances (singletons)
_document_processor: Optional[DocumentProcessor] = None
_ai_pipeline: Optional[AIAnalysisPipeline] = None
_semantic_search: Optional[SemanticSearchService] = None


@lru_cache()
def get_document_processor() -> DocumentProcessor:
    """Get document processor singleton."""
    global _document_processor
    
    if _document_processor is None:
        _document_processor = DocumentProcessor(
            max_workers=settings.max_workers
        )
    
    return _document_processor


@lru_cache()
def get_ai_pipeline() -> AIAnalysisPipeline:
    """Get AI pipeline singleton."""
    global _ai_pipeline
    
    if _ai_pipeline is None:
        _ai_pipeline = AIAnalysisPipeline(
            embedding_model_name=settings.embedding_model,
            enable_classification=True,
            enable_ner=True,
            enable_semantic_search=True
        )
    
    return _ai_pipeline


@lru_cache()
def get_semantic_search() -> SemanticSearchService:
    """Get semantic search service singleton."""
    global _semantic_search
    
    if _semantic_search is None:
        _semantic_search = SemanticSearchService(
            embedding_model_name=settings.embedding_model
        )
    
    return _semantic_search


async def initialize_services():
    """Initialize all services on startup."""
    # Initialize AI pipeline
    ai_pipeline = get_ai_pipeline()
    await ai_pipeline.initialize()
    
    # Initialize semantic search
    semantic_search = get_semantic_search()
    await semantic_search.initialize()


async def shutdown_services():
    """Shutdown all services on app shutdown."""
    global _ai_pipeline, _semantic_search
    
    if _ai_pipeline:
        await _ai_pipeline.shutdown()
    
    # Semantic search cleanup would go here
