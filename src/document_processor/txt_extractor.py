"""
TXT document extractor for plain text files.
"""

import asyncio
from pathlib import Path
from src.core.exceptions import DocumentProcessingError
from src.core.models import DocumentMetadata, DocumentType
from .base import BaseDocumentExtractor


class TXTExtractor(BaseDocumentExtractor):
    """Plain text document extractor."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        super().__init__(chunk_size, chunk_overlap)
        self.supported_types = [DocumentType.TXT]
    
    async def extract_text(self, file_path: Path) -> str:
        """Extract text from TXT file."""
        def _extract():
            try:
                # Try different encodings
                encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
                
                for encoding in encodings:
                    try:
                        with open(file_path, 'r', encoding=encoding) as f:
                            return f.read()
                    except UnicodeDecodeError:
                        continue
                
                # If all encodings fail, read as binary and decode with errors='replace'
                with open(file_path, 'rb') as f:
                    return f.read().decode('utf-8', errors='replace')
                    
            except Exception as e:
                raise DocumentProcessingError(f"Failed to extract TXT text: {str(e)}") from e
        
        text = await asyncio.get_event_loop().run_in_executor(None, _extract)
        return self._clean_text(text)
    
    async def extract_metadata(self, file_path: Path) -> DocumentMetadata:
        """Extract metadata from TXT file."""
        def _extract_metadata():
            try:
                # Get basic file stats
                stat = file_path.stat()
                
                # Try to read file to count words
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    word_count = len(content.split())
                    line_count = len(content.splitlines())
                except:
                    word_count = None
                    line_count = None
                
                return DocumentMetadata(
                    word_count=word_count,
                    file_size=stat.st_size,
                    mime_type="text/plain",
                    custom_fields={
                        'line_count': line_count,
                        'encoding': 'utf-8'  # Assumed
                    }
                )
                
            except Exception as e:
                return DocumentMetadata(
                    file_size=file_path.stat().st_size,
                    mime_type="text/plain"
                )
        
        return await asyncio.get_event_loop().run_in_executor(None, _extract_metadata)
