"""
API response models for Aura Document Analyzer.
Defines request/response schemas for all API endpoints.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from uuid import UUID

from pydantic import BaseModel, Field, validator
from fastapi import UploadFile

from src.core.models import DocumentCategory, ProcessingStatus


class APIResponse(BaseModel):
    """Base API response model."""
    success: bool = True
    message: str = "Operation completed successfully"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ErrorResponse(APIResponse):
    """Error response model."""
    success: bool = False
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str = "0.1.0"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    components: Dict[str, str] = Field(default_factory=dict)
    uptime_seconds: Optional[float] = None


class DocumentUploadRequest(BaseModel):
    """Document upload request."""
    filename: str
    process_immediately: bool = True
    enable_classification: bool = True
    enable_ner: bool = True
    enable_embeddings: bool = True


class DocumentUploadResponse(APIResponse):
    """Document upload response."""
    document_id: UUID
    filename: str
    file_size: int
    document_type: str
    status: ProcessingStatus
    preview: Optional[str] = None
    extracted_text: Optional[str] = None


class DocumentProcessingResponse(APIResponse):
    """Document processing response."""
    document_id: UUID
    status: ProcessingStatus
    text_length: int
    chunk_count: int
    processing_time: float
    classification: Optional[Dict[str, Any]] = None
    entities: List[Dict[str, Any]] = Field(default_factory=list)
    embeddings_generated: bool = False


class ClassificationResponse(BaseModel):
    """Classification result response."""
    category: DocumentCategory
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: Dict[str, float] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EntityResponse(BaseModel):
    """Named entity response."""
    text: str
    label: str
    start_char: int
    end_char: int
    confidence: float = Field(ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchRequest(BaseModel):
    """Search request model."""
    query: str = Field(..., min_length=1, max_length=500)
    limit: int = Field(default=10, ge=1, le=100)
    threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    filters: Dict[str, Any] = Field(default_factory=dict)
    include_metadata: bool = True


class SearchResponse(APIResponse):
    """Search response model."""
    query: str
    results: List[Dict[str, Any]] = Field(default_factory=list)
    total_found: int
    search_time: float


class DocumentListResponse(APIResponse):
    """Document list response."""
    documents: List[Dict[str, Any]] = Field(default_factory=list)
    total_count: int
    page: int = 1
    page_size: int = 10


class DocumentDetailResponse(APIResponse):
    """Document detail response."""
    document_id: UUID
    filename: str
    document_type: str
    status: ProcessingStatus
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)
    text_content: Optional[str] = None
    classification: Optional[ClassificationResponse] = None
    entities: List[EntityResponse] = Field(default_factory=list)
    chunk_count: int = 0


class BatchProcessingRequest(BaseModel):
    """Batch processing request."""
    document_ids: List[UUID] = Field(..., min_items=1, max_items=100)
    processing_options: Dict[str, Any] = Field(default_factory=dict)
    priority: int = Field(default=0, ge=0, le=10)


class BatchProcessingResponse(APIResponse):
    """Batch processing response."""
    batch_id: UUID
    document_count: int
    estimated_time: float
    status: str = "queued"


class StatsResponse(BaseModel):
    """System statistics response."""
    total_documents: int
    processed_documents: int
    failed_documents: int
    average_processing_time: float
    documents_per_minute: float
    storage_used: int  # bytes
    uptime_seconds: float
    ai_models_loaded: Dict[str, bool]


class ModelInfoResponse(BaseModel):
    """AI model information response."""
    model_type: str
    model_name: str
    version: str
    loaded: bool
    capabilities: List[str]
    performance_metrics: Dict[str, Any] = Field(default_factory=dict)
    justification: Dict[str, Any] = Field(default_factory=dict)


class ClassifyRequest(BaseModel):
    """Classification request."""
    text: str = Field(..., min_length=10, max_length=10000)
    include_probabilities: bool = True


class ExtractEntitiesRequest(BaseModel):
    """Entity extraction request."""
    text: str = Field(..., min_length=10, max_length=10000)
    entity_types: Optional[List[str]] = None
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class GenerateEmbeddingsRequest(BaseModel):
    """Embedding generation request."""
    texts: List[str] = Field(..., min_items=1, max_items=100)
    normalize: bool = True


class GenerateEmbeddingsResponse(APIResponse):
    """Embedding generation response."""
    embeddings: List[List[float]]
    dimension: int
    model_used: str
    processing_time: float


class SimilarDocumentsRequest(BaseModel):
    """Similar documents request."""
    document_id: UUID
    limit: int = Field(default=5, ge=1, le=20)
    threshold: float = Field(default=0.7, ge=0.0, le=1.0)


class AnalyzeDocumentRequest(BaseModel):
    """Complete document analysis request."""
    text: str = Field(..., min_length=10, max_length=50000)
    include_classification: bool = True
    include_ner: bool = True
    include_embeddings: bool = True
    include_chunks: bool = True


class AnalyzeDocumentResponse(APIResponse):
    """Complete document analysis response."""
    text_length: int
    processing_time: float
    classification: Optional[ClassificationResponse] = None
    entities: List[EntityResponse] = Field(default_factory=list)
    embeddings: Optional[List[float]] = None
    chunks: List[Dict[str, Any]] = Field(default_factory=list)
    analysis_metadata: Dict[str, Any] = Field(default_factory=dict)


class WebhookRequest(BaseModel):
    """Webhook configuration request."""
    url: str = Field(..., pattern=r'^https?://.+')
    events: List[str] = Field(..., min_length=1)
    secret: Optional[str] = None
    active: bool = True


class WebhookResponse(APIResponse):
    """Webhook configuration response."""
    webhook_id: UUID
    url: str
    events: List[str]
    active: bool
    created_at: datetime


# Validation helpers
class DocumentTypeValidator:
    """Validator for document types."""
    
    ALLOWED_TYPES = {
        'application/pdf': ['.pdf'],
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
        'application/json': ['.json'],
        'text/plain': ['.txt']
    }
    
    @classmethod
    def validate_file(cls, file: UploadFile) -> bool:
        """Validate uploaded file type."""
        if not file.content_type:
            return False
        
        if file.content_type not in cls.ALLOWED_TYPES:
            return False
        
        if not file.filename:
            return False
        
        # Check file extension
        file_ext = '.' + file.filename.split('.')[-1].lower()
        allowed_extensions = cls.ALLOWED_TYPES[file.content_type]
        
        return file_ext in allowed_extensions


# Request/Response examples for OpenAPI documentation
class APIExamples:
    """API request/response examples for documentation."""
    
    UPLOAD_DOCUMENT = {
        "request": {
            "filename": "contract.pdf",
            "process_immediately": True,
            "enable_classification": True,
            "enable_ner": True,
            "enable_embeddings": True
        },
        "response": {
            "success": True,
            "message": "Document uploaded successfully",
            "document_id": "123e4567-e89b-12d3-a456-426614174000",
            "filename": "contract.pdf",
            "file_size": 1024000,
            "document_type": "pdf",
            "status": "processing"
        }
    }
    
    SEARCH_DOCUMENTS = {
        "request": {
            "query": "machine learning algorithms",
            "limit": 5,
            "threshold": 0.8,
            "filters": {"document_type": "report"},
            "include_metadata": True
        },
        "response": {
            "success": True,
            "message": "Search completed successfully",
            "query": "machine learning algorithms",
            "results": [
                {
                    "document_id": "123e4567-e89b-12d3-a456-426614174000",
                    "title": "AI Research Report",
                    "score": 0.95,
                    "snippet": "This report analyzes machine learning algorithms...",
                    "metadata": {"document_type": "report", "author": "Dr. Smith"}
                }
            ],
            "total_found": 1,
            "search_time": 0.045
        }
    }
    
    CLASSIFY_TEXT = {
        "request": {
            "text": "This contract establishes the terms and conditions for software development services.",
            "include_probabilities": True
        },
        "response": {
            "category": "contract",
            "confidence": 0.92,
            "probabilities": {
                "contract": 0.92,
                "legal": 0.05,
                "technical": 0.02,
                "other": 0.01
            },
            "metadata": {
                "model_used": "TF-IDF + SVM",
                "processing_time": 0.015
            }
        }
    }
    
    EXTRACT_ENTITIES = {
        "request": {
            "text": "Contact John Doe at john.doe@company.com or call +1 (555) 123-4567 for contract CONT-2024-001.",
            "confidence_threshold": 0.8
        },
        "response": [
            {
                "text": "John Doe",
                "label": "PERSON",
                "start_char": 8,
                "end_char": 16,
                "confidence": 0.95,
                "metadata": {"source": "spacy"}
            },
            {
                "text": "john.doe@company.com",
                "label": "EMAIL",
                "start_char": 20,
                "end_char": 40,
                "confidence": 0.98,
                "metadata": {"source": "regex"}
            }
        ]
    }
