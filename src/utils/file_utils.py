"""
File utilities for document processing.
"""

import hashlib
import mimetypes
from pathlib import Path
from typing import Optional, Tuple

from src.core.config import settings
from src.core.exceptions import FileTooLargeError, UnsupportedDocumentTypeError
from src.core.logging import LoggerMixin


class FileValidator(LoggerMixin):
    """File validation utilities."""
    
    def __init__(self):
        self.max_file_size = settings.max_file_size
        self.allowed_extensions = settings.allowed_extensions
    
    def validate_file(self, file_path: Path) -> Tuple[bool, Optional[str]]:
        """
        Validate a file for processing.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check if file exists
            if not file_path.exists():
                return False, f"File not found: {file_path}"
            
            # Check if it's a file (not directory)
            if not file_path.is_file():
                return False, f"Path is not a file: {file_path}"
            
            # Check file size
            file_size = file_path.stat().st_size
            if file_size > self.max_file_size:
                return False, f"File too large: {file_size} bytes (max: {self.max_file_size})"
            
            # Check file extension
            if file_path.suffix.lower() not in self.allowed_extensions:
                return False, f"Unsupported file type: {file_path.suffix}"
            
            # Check if file is readable
            try:
                with open(file_path, 'rb') as f:
                    f.read(1)  # Try to read first byte
            except PermissionError:
                return False, f"File not readable: {file_path}"
            except Exception as e:
                return False, f"File access error: {str(e)}"
            
            return True, None
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def get_file_info(self, file_path: Path) -> dict:
        """Get comprehensive file information."""
        try:
            stat = file_path.stat()
            mime_type, _ = mimetypes.guess_type(str(file_path))
            
            return {
                'name': file_path.name,
                'size': stat.st_size,
                'extension': file_path.suffix.lower(),
                'mime_type': mime_type,
                'created': stat.st_ctime,
                'modified': stat.st_mtime,
                'is_readable': self._is_readable(file_path),
                'hash': self.calculate_file_hash(file_path),
            }
        except Exception as e:
            self.logger.error(f"Failed to get file info: {str(e)}")
            return {}
    
    def calculate_file_hash(self, file_path: Path, algorithm: str = 'sha256') -> str:
        """Calculate file hash."""
        try:
            hash_obj = hashlib.new(algorithm)
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_obj.update(chunk)
            return hash_obj.hexdigest()
        except Exception as e:
            self.logger.error(f"Failed to calculate hash: {str(e)}")
            return ""
    
    def _is_readable(self, file_path: Path) -> bool:
        """Check if file is readable."""
        try:
            with open(file_path, 'rb') as f:
                f.read(1)
            return True
        except:
            return False


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage."""
    import re
    
    # Remove or replace dangerous characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove control characters
    filename = ''.join(char for char in filename if ord(char) >= 32)
    
    # Limit length
    if len(filename) > 255:
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        filename = name[:255-len(ext)-1] + '.' + ext if ext else name[:255]
    
    return filename.strip()


def ensure_directory(directory: Path) -> Path:
    """Ensure directory exists."""
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def get_unique_filename(directory: Path, filename: str) -> Path:
    """Get unique filename in directory."""
    base_path = directory / filename
    
    if not base_path.exists():
        return base_path
    
    # Add counter to make unique
    name_part = base_path.stem
    ext_part = base_path.suffix
    counter = 1
    
    while True:
        new_name = f"{name_part}_{counter}{ext_part}"
        new_path = directory / new_name
        if not new_path.exists():
            return new_path
        counter += 1
