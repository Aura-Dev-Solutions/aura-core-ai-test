"""
JSON document extractor for structured data documents.
"""

import asyncio
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict
from src.core.exceptions import DocumentProcessingError
from src.core.models import DocumentMetadata, DocumentType
from .base import BaseDocumentExtractor


class JSONExtractor(BaseDocumentExtractor):
    """JSON document extractor."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        super().__init__(chunk_size, chunk_overlap)
        self.supported_types = [DocumentType.JSON]
    
    async def extract_text(self, file_path: Path) -> str:
        """Extract text from JSON file."""
        def _extract():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                return self._extract_text_from_json(data)
                
            except Exception as e:
                raise DocumentProcessingError(f"Failed to extract JSON text: {str(e)}") from e
        
        text = await asyncio.get_event_loop().run_in_executor(None, _extract)
        return self._clean_text(text)
    
    def _extract_text_from_json(self, data: Any, prefix: str = "") -> str:
        """Recursively extract text from JSON structure."""
        text_parts = []
        
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, str) and value.strip():
                    text_parts.append(f"{key}: {value}")
                elif isinstance(value, (dict, list)):
                    nested_text = self._extract_text_from_json(value, f"{prefix}{key}.")
                    if nested_text:
                        text_parts.append(nested_text)
                else:
                    text_parts.append(f"{key}: {str(value)}")
        
        elif isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, str) and item.strip():
                    text_parts.append(item)
                elif isinstance(item, (dict, list)):
                    nested_text = self._extract_text_from_json(item, f"{prefix}[{i}].")
                    if nested_text:
                        text_parts.append(nested_text)
                else:
                    text_parts.append(str(item))
        
        elif isinstance(data, str):
            text_parts.append(data)
        
        return "\n".join(text_parts)
    
    async def extract_metadata(self, file_path: Path) -> DocumentMetadata:
        """Extract metadata from JSON file."""
        def _extract_metadata():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Try to extract common metadata fields
                title = None
                author = None
                created_date = None
                tags = []
                
                if isinstance(data, dict):
                    title = data.get('title') or data.get('name') or data.get('filename')
                    author = data.get('author') or data.get('creator') or data.get('user')
                    
                    # Try to parse date fields
                    date_fields = ['created_date', 'date', 'timestamp', 'created_at']
                    for field in date_fields:
                        if field in data:
                            try:
                                created_date = datetime.fromisoformat(str(data[field]).replace('Z', '+00:00'))
                                break
                            except:
                                continue
                    
                    # Extract tags
                    if 'tags' in data and isinstance(data['tags'], list):
                        tags = [str(tag) for tag in data['tags']]
                
                # Count words in extracted text
                text = self._extract_text_from_json(data)
                word_count = len(text.split()) if text else 0
                
                return DocumentMetadata(
                    title=title,
                    author=author,
                    created_date=created_date,
                    word_count=word_count,
                    file_size=file_path.stat().st_size,
                    mime_type="application/json",
                    tags=tags,
                    custom_fields={
                        'json_structure': self._analyze_json_structure(data),
                        'total_keys': self._count_keys(data),
                    }
                )
                
            except Exception as e:
                return DocumentMetadata(
                    file_size=file_path.stat().st_size,
                    mime_type="application/json"
                )
        
        return await asyncio.get_event_loop().run_in_executor(None, _extract_metadata)
    
    def _analyze_json_structure(self, data: Any) -> Dict[str, Any]:
        """Analyze JSON structure."""
        if isinstance(data, dict):
            return {
                'type': 'object',
                'keys': list(data.keys())[:10],  # First 10 keys
                'total_keys': len(data),
            }
        elif isinstance(data, list):
            return {
                'type': 'array',
                'length': len(data),
                'item_types': list(set(type(item).__name__ for item in data[:10]))
            }
        else:
            return {'type': type(data).__name__}
    
    def _count_keys(self, data: Any) -> int:
        """Count total number of keys in nested JSON."""
        count = 0
        if isinstance(data, dict):
            count += len(data)
            for value in data.values():
                count += self._count_keys(value)
        elif isinstance(data, list):
            for item in data:
                count += self._count_keys(item)
        return count
