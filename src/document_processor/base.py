"""
Base classes for document processing.
Defines the interface and common functionality for all document extractors.
"""

import asyncio
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.exceptions import DocumentProcessingError, UnsupportedDocumentTypeError
from src.core.logging import LoggerMixin, log_performance
from src.core.models import Document, DocumentMetadata, DocumentType, TextChunk


class BaseDocumentExtractor(ABC, LoggerMixin):
    """Abstract base class for document extractors."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the extractor.
        
        Args:
            chunk_size: Maximum size of text chunks in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.supported_types: List[DocumentType] = []
    
    @abstractmethod
    async def extract_text(self, file_path: Path) -> str:
        """
        Extract text content from a document.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Extracted text content
            
        Raises:
            DocumentProcessingError: If extraction fails
        """
        pass
    
    @abstractmethod
    async def extract_metadata(self, file_path: Path) -> DocumentMetadata:
        """
        Extract metadata from a document.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Document metadata
            
        Raises:
            DocumentProcessingError: If metadata extraction fails
        """
        pass
    
    def can_process(self, document_type: DocumentType) -> bool:
        """
        Check if this extractor can process the given document type.
        
        Args:
            document_type: Type of document to check
            
        Returns:
            True if the extractor can process this type
        """
        return document_type in self.supported_types
    
    def create_chunks(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[TextChunk]:
        """
        Split text into chunks with overlap.
        
        Args:
            text: Text to chunk
            metadata: Optional metadata to include in chunks
            
        Returns:
            List of text chunks
        """
        if not text.strip():
            return []
        
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            
            # Try to break at word boundaries
            if end < len(text):
                # Look for the last space within the chunk
                last_space = text.rfind(' ', start, end)
                if last_space > start:
                    end = last_space
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunk = TextChunk(
                    text=chunk_text,
                    start_char=start,
                    end_char=end,
                    metadata=metadata or {}
                )
                chunks.append(chunk)
                chunk_id += 1
            
            # Move start position with overlap
            start = max(start + 1, end - self.chunk_overlap)
            if start >= end:
                break
        
        return chunks
    
    @log_performance("document_extraction")
    async def process_document(self, file_path: Path) -> Document:
        """
        Process a complete document.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Processed document with text and metadata
            
        Raises:
            DocumentProcessingError: If processing fails
            UnsupportedDocumentTypeError: If document type is not supported
        """
        if not file_path.exists():
            raise DocumentProcessingError(f"File not found: {file_path}")
        
        # Determine document type
        document_type = self._get_document_type(file_path)
        
        if not self.can_process(document_type):
            raise UnsupportedDocumentTypeError(
                f"Document type {document_type} not supported by {self.__class__.__name__}"
            )
        
        try:
            # Extract text and metadata concurrently
            text_task = asyncio.create_task(self.extract_text(file_path))
            metadata_task = asyncio.create_task(self.extract_metadata(file_path))
            
            text_content, metadata = await asyncio.gather(text_task, metadata_task)
            
            # Create document
            document = Document(
                filename=file_path.name,
                file_path=file_path,
                document_type=document_type,
                metadata=metadata
            )
            
            self.logger.info(
                "Document processed successfully",
                document_id=str(document.id),
                filename=file_path.name,
                text_length=len(text_content),
                chunk_count=len(self.create_chunks(text_content))
            )
            
            return document
            
        except Exception as e:
            self.logger.error(
                "Document processing failed",
                filename=file_path.name,
                error=str(e)
            )
            raise DocumentProcessingError(f"Failed to process document: {str(e)}") from e
    
    def _get_document_type(self, file_path: Path) -> DocumentType:
        """Get document type from file extension."""
        suffix = file_path.suffix.lower()
        type_mapping = {
            '.pdf': DocumentType.PDF,
            '.docx': DocumentType.DOCX,
            '.json': DocumentType.JSON,
            '.txt': DocumentType.TXT,
        }
        return type_mapping.get(suffix, DocumentType.TXT)
    
    def _estimate_reading_time(self, text: str, words_per_minute: int = 200) -> float:
        """Estimate reading time in minutes."""
        word_count = len(text.split())
        return word_count / words_per_minute
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        import re
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Remove control characters except newlines and tabs
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
        
        return text.strip()


# Alias para compatibilidad
BaseExtractor = BaseDocumentExtractor


class DocumentProcessorRegistry:
    """Registry for document processors."""
    
    def __init__(self):
        self._processors: Dict[DocumentType, BaseDocumentExtractor] = {}
    
    def register(self, document_type: DocumentType, processor: BaseDocumentExtractor):
        """Register a processor for a document type."""
        self._processors[document_type] = processor
    
    def get_processor(self, document_type: DocumentType) -> Optional[BaseDocumentExtractor]:
        """Get processor for a document type."""
        return self._processors.get(document_type)
    
    def get_supported_types(self) -> List[DocumentType]:
        """Get list of supported document types."""
        return list(self._processors.keys())
    
    async def process_document(self, file_path: Path) -> Document:
        """Process a document using the appropriate processor."""
        # Determine document type
        suffix = file_path.suffix.lower()
        type_mapping = {
            '.pdf': DocumentType.PDF,
            '.docx': DocumentType.DOCX,
            '.json': DocumentType.JSON,
            '.txt': DocumentType.TXT,
        }
        document_type = type_mapping.get(suffix)
        
        if not document_type:
            raise UnsupportedDocumentTypeError(f"Unsupported file type: {suffix}")
        
        processor = self.get_processor(document_type)
        if not processor:
            raise UnsupportedDocumentTypeError(f"No processor registered for type: {document_type}")
        
        return await processor.process_document(file_path)


# Global registry instance
document_registry = DocumentProcessorRegistry()
