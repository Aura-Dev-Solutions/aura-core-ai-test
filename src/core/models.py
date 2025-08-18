"""
Core data models for Aura Document Analyzer.
Defines the structure for documents, metadata, and processing results.
"""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator


class DocumentType(str, Enum):
    """Supported document types."""
    PDF = "pdf"
    DOCX = "docx"
    JSON = "json"
    TXT = "txt"


class ProcessingStatus(str, Enum):
    """Document processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DocumentCategory(str, Enum):
    """Document classification categories."""
    CONTRACT = "contract"
    REPORT = "report"
    LEGAL = "legal"
    CORRESPONDENCE = "correspondence"
    TECHNICAL = "technical"
    FINANCIAL = "financial"
    OTHER = "other"


class DocumentMetadata(BaseModel):
    """Document metadata structure."""
    title: Optional[str] = None
    author: Optional[str] = None
    created_date: Optional[datetime] = None
    modified_date: Optional[datetime] = None
    language: Optional[str] = "en"
    page_count: Optional[int] = None
    word_count: Optional[int] = None
    file_size: Optional[int] = None
    mime_type: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    custom_fields: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }


class TextChunk(BaseModel):
    """Text chunk with metadata."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    text: str
    start_char: int
    end_char: int
    page_number: Optional[int] = None
    section: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class NamedEntity(BaseModel):
    """Named entity extraction result."""
    text: str
    label: str
    start_char: int
    end_char: int
    confidence: float = Field(ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ClassificationResult(BaseModel):
    """Document classification result."""
    category: DocumentCategory
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: Dict[str, float] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ProcessingResult(BaseModel):
    """Complete document processing result."""
    document_id: UUID
    status: ProcessingStatus
    text_content: str
    chunks: List[TextChunk] = Field(default_factory=list)
    entities: List[NamedEntity] = Field(default_factory=list)
    classification: Optional[ClassificationResult] = None
    embeddings: Optional[List[float]] = None
    processing_time: Optional[float] = None
    error_message: Optional[str] = None
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }


class Document(BaseModel):
    """Main document model."""
    id: UUID = Field(default_factory=uuid4)
    filename: str
    file_path: Path
    document_type: DocumentType
    status: ProcessingStatus = ProcessingStatus.PENDING
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
    processing_result: Optional[ProcessingResult] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None,
            Path: lambda v: str(v)
        }
    
    @validator('document_type', pre=True)
    def determine_document_type(cls, v, values):
        """Determine document type from filename if not provided."""
        if isinstance(v, str) and v in DocumentType.__members__.values():
            return v
        
        filename = values.get('filename', '')
        if filename:
            suffix = Path(filename).suffix.lower()
            type_mapping = {
                '.pdf': DocumentType.PDF,
                '.docx': DocumentType.DOCX,
                '.json': DocumentType.JSON,
                '.txt': DocumentType.TXT,
            }
            return type_mapping.get(suffix, DocumentType.TXT)
        return DocumentType.TXT
    
    @validator('file_path', pre=True)
    def convert_to_path(cls, v):
        """Convert string to Path object."""
        return Path(v) if not isinstance(v, Path) else v
    
    class Config:
        use_enum_values = True
        json_encoders = {
            Path: str,
            datetime: lambda v: v.isoformat(),
            UUID: str,
        }


class SearchQuery(BaseModel):
    """Search query model."""
    query: str
    filters: Dict[str, Any] = Field(default_factory=dict)
    limit: int = Field(default=10, ge=1, le=100)
    threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    include_metadata: bool = True


class SearchResult(BaseModel):
    """Search result model."""
    document_id: Union[UUID, str]
    score: float = Field(ge=0.0, le=1.0)
    title: Optional[str] = None
    snippet: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    chunk_id: Optional[str] = None


class BatchProcessingRequest(BaseModel):
    """Batch processing request model."""
    document_ids: List[UUID]
    processing_options: Dict[str, Any] = Field(default_factory=dict)
    priority: int = Field(default=0, ge=0, le=10)


class ProcessingStats(BaseModel):
    """Processing statistics model."""
    total_documents: int = 0
    processed_documents: int = 0
    failed_documents: int = 0
    average_processing_time: float = 0.0
    total_processing_time: float = 0.0
    documents_per_minute: float = 0.0
    error_rate: float = 0.0
