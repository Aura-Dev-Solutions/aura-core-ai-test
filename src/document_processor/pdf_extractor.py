"""
PDF document extractor using multiple libraries for robust extraction.
Handles various PDF formats and extraction challenges.
"""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional

import pdfplumber
import PyPDF2
from PyPDF2 import PdfReader

from src.core.exceptions import DocumentCorruptedError, DocumentProcessingError
from src.core.models import DocumentMetadata, DocumentType
from .base import BaseDocumentExtractor


class PDFExtractor(BaseDocumentExtractor):
    """PDF document extractor with fallback strategies."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        super().__init__(chunk_size, chunk_overlap)
        self.supported_types = [DocumentType.PDF]
    
    async def extract_text(self, file_path: Path) -> str:
        """
        Extract text from PDF using multiple strategies.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Extracted text content
            
        Raises:
            DocumentProcessingError: If all extraction methods fail
        """
        try:
            # Try pdfplumber first (better for complex layouts)
            text = await self._extract_with_pdfplumber(file_path)
            if text and len(text.strip()) > 50:  # Reasonable amount of text
                return self._clean_text(text)
            
            self.logger.warning(
                "pdfplumber extraction yielded minimal text, trying PyPDF2",
                filename=file_path.name,
                text_length=len(text) if text else 0
            )
            
            # Fallback to PyPDF2
            text = await self._extract_with_pypdf2(file_path)
            if text and len(text.strip()) > 10:
                return self._clean_text(text)
            
            # If both methods fail, return what we have
            self.logger.warning(
                "Both extraction methods yielded minimal text",
                filename=file_path.name,
                final_text_length=len(text) if text else 0
            )
            
            return self._clean_text(text) if text else ""
            
        except Exception as e:
            self.logger.error(
                "PDF text extraction failed",
                filename=file_path.name,
                error=str(e)
            )
            raise DocumentProcessingError(f"Failed to extract text from PDF: {str(e)}") from e
    
    async def _extract_with_pdfplumber(self, file_path: Path) -> str:
        """Extract text using pdfplumber (better for tables and complex layouts)."""
        def _extract():
            text_parts = []
            try:
                with pdfplumber.open(file_path) as pdf:
                    for page_num, page in enumerate(pdf.pages, 1):
                        try:
                            page_text = page.extract_text()
                            if page_text:
                                text_parts.append(f"[Page {page_num}]\n{page_text}\n")
                        except Exception as e:
                            self.logger.warning(
                                "Failed to extract text from page",
                                page_number=page_num,
                                error=str(e)
                            )
                            continue
                return "\n".join(text_parts)
            except Exception as e:
                raise DocumentProcessingError(f"pdfplumber extraction failed: {str(e)}") from e
        
        return await asyncio.get_event_loop().run_in_executor(None, _extract)
    
    async def _extract_with_pypdf2(self, file_path: Path) -> str:
        """Extract text using PyPDF2 (fallback method)."""
        def _extract():
            text_parts = []
            try:
                with open(file_path, 'rb') as file:
                    pdf_reader = PdfReader(file)
                    
                    for page_num, page in enumerate(pdf_reader.pages, 1):
                        try:
                            page_text = page.extract_text()
                            if page_text:
                                text_parts.append(f"[Page {page_num}]\n{page_text}\n")
                        except Exception as e:
                            self.logger.warning(
                                "Failed to extract text from page with PyPDF2",
                                page_number=page_num,
                                error=str(e)
                            )
                            continue
                
                return "\n".join(text_parts)
            except Exception as e:
                raise DocumentProcessingError(f"PyPDF2 extraction failed: {str(e)}") from e
        
        return await asyncio.get_event_loop().run_in_executor(None, _extract)
    
    async def extract_metadata(self, file_path: Path) -> DocumentMetadata:
        """
        Extract metadata from PDF.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Document metadata
        """
        def _extract_metadata():
            try:
                with open(file_path, 'rb') as file:
                    pdf_reader = PdfReader(file)
                    
                    # Get basic info
                    page_count = len(pdf_reader.pages)
                    file_size = file_path.stat().st_size
                    
                    # Try to get PDF metadata
                    metadata_dict = {}
                    if pdf_reader.metadata:
                        metadata_dict = {
                            'title': pdf_reader.metadata.get('/Title'),
                            'author': pdf_reader.metadata.get('/Author'),
                            'subject': pdf_reader.metadata.get('/Subject'),
                            'creator': pdf_reader.metadata.get('/Creator'),
                            'producer': pdf_reader.metadata.get('/Producer'),
                            'creation_date': pdf_reader.metadata.get('/CreationDate'),
                            'modification_date': pdf_reader.metadata.get('/ModDate'),
                        }
                    
                    # Parse dates
                    created_date = None
                    modified_date = None
                    
                    if metadata_dict.get('creation_date'):
                        created_date = self._parse_pdf_date(metadata_dict['creation_date'])
                    
                    if metadata_dict.get('modification_date'):
                        modified_date = self._parse_pdf_date(metadata_dict['modification_date'])
                    
                    # Estimate word count (rough approximation)
                    word_count = None
                    try:
                        # Quick text extraction for word count
                        sample_text = ""
                        for i, page in enumerate(pdf_reader.pages[:3]):  # Sample first 3 pages
                            sample_text += page.extract_text() or ""
                        
                        if sample_text:
                            words_per_page = len(sample_text.split()) / min(3, page_count)
                            word_count = int(words_per_page * page_count)
                    except Exception:
                        pass
                    
                    return DocumentMetadata(
                        title=metadata_dict.get('title'),
                        author=metadata_dict.get('author'),
                        created_date=created_date,
                        modified_date=modified_date,
                        page_count=page_count,
                        word_count=word_count,
                        file_size=file_size,
                        mime_type="application/pdf",
                        custom_fields={
                            'pdf_version': getattr(pdf_reader, 'pdf_header', None),
                            'encrypted': pdf_reader.is_encrypted,
                            'creator': metadata_dict.get('creator'),
                            'producer': metadata_dict.get('producer'),
                            'subject': metadata_dict.get('subject'),
                        }
                    )
                    
            except Exception as e:
                self.logger.error(
                    "PDF metadata extraction failed",
                    filename=file_path.name,
                    error=str(e)
                )
                # Return basic metadata if detailed extraction fails
                return DocumentMetadata(
                    file_size=file_path.stat().st_size,
                    mime_type="application/pdf"
                )
        
        return await asyncio.get_event_loop().run_in_executor(None, _extract_metadata)
    
    def _parse_pdf_date(self, date_str: str) -> Optional[datetime]:
        """Parse PDF date format (D:YYYYMMDDHHmmSSOHH'mm')."""
        if not date_str:
            return None
        
        try:
            # Remove D: prefix if present
            if date_str.startswith('D:'):
                date_str = date_str[2:]
            
            # Take only the date part (first 14 characters: YYYYMMDDHHMMSS)
            date_part = date_str[:14]
            
            # Parse the date
            return datetime.strptime(date_part, '%Y%m%d%H%M%S')
        except (ValueError, IndexError):
            try:
                # Try with just date part
                date_part = date_str[:8]
                return datetime.strptime(date_part, '%Y%m%d')
            except (ValueError, IndexError):
                return None
    
    async def extract_images_info(self, file_path: Path) -> list:
        """Extract information about images in the PDF."""
        def _extract_images():
            images_info = []
            try:
                with pdfplumber.open(file_path) as pdf:
                    for page_num, page in enumerate(pdf.pages, 1):
                        if hasattr(page, 'images'):
                            for img in page.images:
                                images_info.append({
                                    'page': page_num,
                                    'bbox': img.get('bbox'),
                                    'width': img.get('width'),
                                    'height': img.get('height'),
                                })
            except Exception as e:
                self.logger.warning(
                    "Failed to extract image information",
                    filename=file_path.name,
                    error=str(e)
                )
            return images_info
        
        return await asyncio.get_event_loop().run_in_executor(None, _extract_images)
    
    async def extract_tables_info(self, file_path: Path) -> list:
        """Extract information about tables in the PDF."""
        def _extract_tables():
            tables_info = []
            try:
                with pdfplumber.open(file_path) as pdf:
                    for page_num, page in enumerate(pdf.pages, 1):
                        tables = page.extract_tables()
                        if tables:
                            for i, table in enumerate(tables):
                                tables_info.append({
                                    'page': page_num,
                                    'table_index': i,
                                    'rows': len(table),
                                    'columns': len(table[0]) if table else 0,
                                    'preview': table[:2] if table else []  # First 2 rows
                                })
            except Exception as e:
                self.logger.warning(
                    "Failed to extract table information",
                    filename=file_path.name,
                    error=str(e)
                )
            return tables_info
        
        return await asyncio.get_event_loop().run_in_executor(None, _extract_tables)
