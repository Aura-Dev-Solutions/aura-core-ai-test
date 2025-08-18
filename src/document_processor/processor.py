"""
Main document processor with parallel processing capabilities.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Dict, Any
from uuid import uuid4

from src.core.config import settings
from src.core.exceptions import DocumentProcessingError, UnsupportedDocumentTypeError
from src.core.logging import LoggerMixin, log_performance
from src.core.models import Document, ProcessingResult, ProcessingStatus, ProcessingStats
from .base import document_registry
from .pdf_extractor import PDFExtractor
from .docx_extractor import DOCXExtractor
from .json_extractor import JSONExtractor
from .txt_extractor import TXTExtractor


class DocumentProcessor(LoggerMixin):
    """Main document processor with parallel processing."""
    
    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or settings.max_workers
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self._setup_extractors()
        self.stats = ProcessingStats()
    
    def _setup_extractors(self):
        """Register all document extractors."""
        from src.core.models import DocumentType
        
        # Register extractors
        document_registry.register(DocumentType.PDF, PDFExtractor())
        document_registry.register(DocumentType.DOCX, DOCXExtractor())
        document_registry.register(DocumentType.JSON, JSONExtractor())
        document_registry.register(DocumentType.TXT, TXTExtractor())
        
        self.logger.info(
            "Document extractors registered",
            supported_types=document_registry.get_supported_types()
        )
    
    @log_performance("single_document_processing")
    async def process_document(self, file_path: Path) -> ProcessingResult:
        """
        Process a single document.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Processing result with extracted content
        """
        document_id = uuid4()
        start_time = asyncio.get_event_loop().time()
        
        try:
            self.logger.info(
                "Starting document processing",
                document_id=str(document_id),
                filename=file_path.name
            )
            
            # Process document
            document = await document_registry.process_document(file_path)
            
            # Extract text content
            processor = document_registry.get_processor(document.document_type)
            text_content = await processor.extract_text(file_path)
            
            # Create chunks
            chunks = processor.create_chunks(text_content)
            
            # Calculate processing time
            processing_time = asyncio.get_event_loop().time() - start_time
            
            # Create processing result
            result = ProcessingResult(
                document_id=document.id,
                status=ProcessingStatus.COMPLETED,
                text_content=text_content,
                chunks=chunks,
                processing_time=processing_time,
                metadata=document.metadata
            )
            
            # Update stats
            self.stats.total_documents += 1
            self.stats.processed_documents += 1
            self.stats.total_processing_time += processing_time
            self._update_stats()
            
            self.logger.info(
                "Document processing completed",
                document_id=str(document_id),
                processing_time=processing_time,
                text_length=len(text_content),
                chunk_count=len(chunks)
            )
            
            return result
            
        except Exception as e:
            processing_time = asyncio.get_event_loop().time() - start_time
            
            # Update error stats
            self.stats.total_documents += 1
            self.stats.failed_documents += 1
            self.stats.total_processing_time += processing_time
            self._update_stats()
            
            self.logger.error(
                "Document processing failed",
                document_id=str(document_id),
                filename=file_path.name,
                error=str(e),
                processing_time=processing_time
            )
            
            return ProcessingResult(
                document_id=document_id,
                status=ProcessingStatus.FAILED,
                text_content="",
                processing_time=processing_time,
                error_message=str(e)
            )
    
    @log_performance("batch_document_processing")
    async def process_documents_batch(
        self, 
        file_paths: List[Path],
        max_concurrent: Optional[int] = None
    ) -> List[ProcessingResult]:
        """
        Process multiple documents in parallel.
        
        Args:
            file_paths: List of file paths to process
            max_concurrent: Maximum concurrent processing tasks
            
        Returns:
            List of processing results
        """
        max_concurrent = max_concurrent or min(len(file_paths), self.max_workers)
        
        self.logger.info(
            "Starting batch processing",
            total_documents=len(file_paths),
            max_concurrent=max_concurrent
        )
        
        # Create semaphore to limit concurrent processing
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(file_path: Path) -> ProcessingResult:
            async with semaphore:
                return await self.process_document(file_path)
        
        # Process all documents concurrently
        tasks = [process_with_semaphore(path) for path in file_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(
                    "Batch processing task failed",
                    filename=file_paths[i].name,
                    error=str(result)
                )
                processed_results.append(
                    ProcessingResult(
                        document_id=uuid4(),
                        status=ProcessingStatus.FAILED,
                        text_content="",
                        error_message=str(result)
                    )
                )
            else:
                processed_results.append(result)
        
        self.logger.info(
            "Batch processing completed",
            total_documents=len(file_paths),
            successful=sum(1 for r in processed_results if r.status == ProcessingStatus.COMPLETED),
            failed=sum(1 for r in processed_results if r.status == ProcessingStatus.FAILED)
        )
        
        return processed_results
    
    async def process_directory(
        self, 
        directory_path: Path,
        recursive: bool = True,
        file_patterns: Optional[List[str]] = None
    ) -> List[ProcessingResult]:
        """
        Process all supported documents in a directory.
        
        Args:
            directory_path: Directory to process
            recursive: Whether to process subdirectories
            file_patterns: File patterns to match (e.g., ['*.pdf', '*.docx'])
            
        Returns:
            List of processing results
        """
        if not directory_path.exists() or not directory_path.is_dir():
            raise DocumentProcessingError(f"Directory not found: {directory_path}")
        
        # Find all supported files
        file_paths = []
        supported_extensions = {'.pdf', '.docx', '.json', '.txt'}
        
        if recursive:
            pattern = "**/*"
        else:
            pattern = "*"
        
        for file_path in directory_path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                if file_patterns:
                    if any(file_path.match(pattern) for pattern in file_patterns):
                        file_paths.append(file_path)
                else:
                    file_paths.append(file_path)
        
        self.logger.info(
            "Found files for processing",
            directory=str(directory_path),
            file_count=len(file_paths),
            recursive=recursive
        )
        
        if not file_paths:
            return []
        
        return await self.process_documents_batch(file_paths)
    
    def get_supported_types(self) -> List[str]:
        """Get list of supported document types."""
        return [doc_type.value for doc_type in document_registry.get_supported_types()]
    
    def get_processing_stats(self) -> ProcessingStats:
        """Get current processing statistics."""
        return self.stats
    
    def reset_stats(self):
        """Reset processing statistics."""
        self.stats = ProcessingStats()
    
    def _update_stats(self):
        """Update calculated statistics."""
        if self.stats.processed_documents > 0:
            self.stats.average_processing_time = (
                self.stats.total_processing_time / self.stats.processed_documents
            )
        
        if self.stats.total_documents > 0:
            self.stats.error_rate = self.stats.failed_documents / self.stats.total_documents
        
        if self.stats.total_processing_time > 0:
            self.stats.documents_per_minute = (
                self.stats.processed_documents / (self.stats.total_processing_time / 60)
            )
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the processor."""
        try:
            # Test with a simple text processing
            test_file = Path("test.txt")
            test_content = "This is a test document for health check."
            
            # Create temporary test file
            test_file.write_text(test_content)
            
            try:
                result = await self.process_document(test_file)
                success = result.status == ProcessingStatus.COMPLETED
            finally:
                test_file.unlink(missing_ok=True)
            
            return {
                "status": "healthy" if success else "unhealthy",
                "supported_types": self.get_supported_types(),
                "max_workers": self.max_workers,
                "stats": self.stats.dict()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "supported_types": self.get_supported_types(),
                "max_workers": self.max_workers
            }
    
    def __del__(self):
        """Cleanup executor on deletion."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
