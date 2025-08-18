"""
Custom exceptions for Aura Document Analyzer.
Provides specific error types for different failure scenarios.
"""

from typing import Any, Dict, Optional


class AuraDocumentAnalyzerError(Exception):
    """Base exception for all application errors."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        super().__init__(self.message)


class DocumentProcessingError(AuraDocumentAnalyzerError):
    """Raised when document processing fails."""
    pass


class UnsupportedDocumentTypeError(DocumentProcessingError):
    """Raised when trying to process an unsupported document type."""
    pass


class DocumentCorruptedError(DocumentProcessingError):
    """Raised when a document is corrupted or unreadable."""
    pass


class FileTooLargeError(DocumentProcessingError):
    """Raised when a file exceeds the maximum allowed size."""
    pass


class AIModelError(AuraDocumentAnalyzerError):
    """Base exception for AI model related errors."""
    pass


class ModelNotFoundError(AIModelError):
    """Raised when a required model is not found."""
    pass


class ModelLoadError(AIModelError):
    """Raised when a model fails to load."""
    pass


class EmbeddingGenerationError(AIModelError):
    """Raised when embedding generation fails."""
    pass


class ClassificationError(AIModelError):
    """Raised when document classification fails."""
    pass


class NERExtractionError(AIModelError):
    """Raised when NER extraction fails."""
    pass


class DatabaseError(AuraDocumentAnalyzerError):
    """Base exception for database related errors."""
    pass


class DocumentNotFoundError(DatabaseError):
    """Raised when a document is not found in the database."""
    pass


class DuplicateDocumentError(DatabaseError):
    """Raised when trying to insert a duplicate document."""
    pass


class VectorDatabaseError(AuraDocumentAnalyzerError):
    """Base exception for vector database related errors."""
    pass


class SearchError(VectorDatabaseError):
    """Raised when vector search fails."""
    pass


class IndexError(VectorDatabaseError):
    """Raised when vector indexing fails."""
    pass


class ValidationError(AuraDocumentAnalyzerError):
    """Raised when input validation fails."""
    pass


class AuthenticationError(AuraDocumentAnalyzerError):
    """Raised when authentication fails."""
    pass


class AuthorizationError(AuraDocumentAnalyzerError):
    """Raised when authorization fails."""
    pass


class RateLimitError(AuraDocumentAnalyzerError):
    """Raised when rate limit is exceeded."""
    pass


class ExternalServiceError(AuraDocumentAnalyzerError):
    """Raised when an external service fails."""
    pass


# Error code mapping for API responses
ERROR_CODE_MAP = {
    DocumentProcessingError: "DOCUMENT_PROCESSING_ERROR",
    UnsupportedDocumentTypeError: "UNSUPPORTED_DOCUMENT_TYPE",
    DocumentCorruptedError: "DOCUMENT_CORRUPTED",
    FileTooLargeError: "FILE_TOO_LARGE",
    AIModelError: "AI_MODEL_ERROR",
    ModelNotFoundError: "MODEL_NOT_FOUND",
    ModelLoadError: "MODEL_LOAD_ERROR",
    EmbeddingGenerationError: "EMBEDDING_GENERATION_ERROR",
    ClassificationError: "CLASSIFICATION_ERROR",
    NERExtractionError: "NER_EXTRACTION_ERROR",
    DatabaseError: "DATABASE_ERROR",
    DocumentNotFoundError: "DOCUMENT_NOT_FOUND",
    DuplicateDocumentError: "DUPLICATE_DOCUMENT",
    VectorDatabaseError: "VECTOR_DATABASE_ERROR",
    SearchError: "SEARCH_ERROR",
    IndexError: "INDEX_ERROR",
    ValidationError: "VALIDATION_ERROR",
    AuthenticationError: "AUTHENTICATION_ERROR",
    AuthorizationError: "AUTHORIZATION_ERROR",
    RateLimitError: "RATE_LIMIT_ERROR",
    ExternalServiceError: "EXTERNAL_SERVICE_ERROR",
}
