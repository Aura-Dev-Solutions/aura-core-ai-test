"""
Registry para gestionar extractores de documentos.

Este módulo proporciona un registro centralizado para todos los extractores
de documentos disponibles en el sistema.
"""

import logging
from typing import Dict, List, Optional
from .base import BaseExtractor
from .pdf_extractor import PDFExtractor
from .docx_extractor import DOCXExtractor
from .json_extractor import JSONExtractor
from .txt_extractor import TXTExtractor

logger = logging.getLogger(__name__)


class ProcessorRegistry:
    """
    Registro centralizado para extractores de documentos.
    
    Permite registrar, obtener y gestionar diferentes tipos de extractores
    de documentos de forma centralizada.
    """
    
    def __init__(self):
        """Inicializar el registro con extractores por defecto."""
        self._processors: Dict[str, BaseExtractor] = {}
        self._register_default_processors()
    
    def _register_default_processors(self):
        """Registrar extractores por defecto."""
        try:
            self.register("pdf", PDFExtractor())
            self.register("docx", DOCXExtractor())
            self.register("json", JSONExtractor())
            self.register("txt", TXTExtractor())
        except Exception as e:
            # Si hay error registrando algún extractor, continuar con los otros
            logger.warning(f"Error registering default processors: {e}")
    
    def register(self, file_type: str, processor: BaseExtractor):
        """
        Registrar un extractor para un tipo de archivo.
        
        Args:
            file_type: Tipo de archivo (ej: 'pdf', 'docx')
            processor: Instancia del extractor
        """
        self._processors[file_type.lower()] = processor
    
    def get_processor(self, file_type: str) -> Optional[BaseExtractor]:
        """
        Obtener extractor para un tipo de archivo.
        
        Args:
            file_type: Tipo de archivo
            
        Returns:
            Extractor correspondiente o None si no existe
        """
        return self._processors.get(file_type.lower())
    
    def get_supported_types(self) -> List[str]:
        """
        Obtener lista de tipos de archivo soportados.
        
        Returns:
            Lista de tipos de archivo soportados
        """
        return list(self._processors.keys())
    
    def is_supported(self, file_type: str) -> bool:
        """
        Verificar si un tipo de archivo está soportado.
        
        Args:
            file_type: Tipo de archivo a verificar
            
        Returns:
            True si está soportado, False en caso contrario
        """
        return file_type.lower() in self._processors
