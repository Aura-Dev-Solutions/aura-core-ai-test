"""
File validation utilities.
"""

from pathlib import Path
from typing import List
from fastapi import UploadFile


class DocumentTypeValidator:
    """Validator for document types."""
    
    ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.json', '.txt'}
    ALLOWED_MIME_TYPES = {
        'application/pdf',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'application/json',
        'text/plain',
        'text/json'
    }
    
    @classmethod
    def validate_file(cls, file: UploadFile) -> bool:
        """Validate if file is supported."""
        if not file.filename:
            return False
            
        # Check extension
        file_path = Path(file.filename)
        if file_path.suffix.lower() not in cls.ALLOWED_EXTENSIONS:
            return False
            
        # Check MIME type if available
        if file.content_type and file.content_type not in cls.ALLOWED_MIME_TYPES:
            # Allow if extension is valid even if MIME type is not recognized
            pass
            
        return True
    
    @classmethod
    def get_allowed_extensions(cls) -> List[str]:
        """Get list of allowed extensions."""
        return list(cls.ALLOWED_EXTENSIONS)
