"""
Configuración de logging para el sistema de análisis de documentos.

Este módulo configura el sistema de logging estructurado usando structlog
para proporcionar logs consistentes y fáciles de analizar.
"""

import logging
import sys
from typing import Any, Dict
import structlog
from structlog.stdlib import LoggerFactory


def setup_logging(
    level: str = "INFO",
    format_json: bool = False,
    include_timestamp: bool = True
) -> structlog.stdlib.BoundLogger:
    """
    Configurar el sistema de logging estructurado.
    
    Args:
        level: Nivel de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_json: Si usar formato JSON para los logs
        include_timestamp: Si incluir timestamp en los logs
        
    Returns:
        Logger configurado
    """
    
    # Configurar logging estándar de Python
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper())
    )
    
    # Configurar procesadores de structlog
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    
    if include_timestamp:
        processors.append(structlog.processors.TimeStamper(fmt="ISO"))
    
    # Agregar procesador de formato
    if format_json:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    
    # Configurar structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=LoggerFactory(),
        context_class=dict,
        cache_logger_on_first_use=True,
    )
    
    # Crear y retornar logger
    logger = structlog.get_logger("aura_document_analyzer")
    
    return logger


def get_logger(name: str = "aura_document_analyzer") -> structlog.stdlib.BoundLogger:
    """
    Obtener un logger configurado.
    
    Args:
        name: Nombre del logger
        
    Returns:
        Logger configurado
    """
    return structlog.get_logger(name)


class LoggerMixin:
    """
    Mixin para agregar capacidades de logging a cualquier clase.
    
    Proporciona un logger configurado automáticamente basado en el nombre
    de la clase.
    """
    
    @property
    def logger(self) -> structlog.stdlib.BoundLogger:
        """Obtener logger para esta clase."""
        if not hasattr(self, '_logger'):
            self._logger = get_logger(self.__class__.__name__)
        return self._logger
    
    def log_operation(self, operation: str, **kwargs):
        """
        Log de una operación con contexto adicional.
        
        Args:
            operation: Nombre de la operación
            **kwargs: Contexto adicional para el log
        """
        self.logger.info(f"Operation: {operation}", **kwargs)
    
    def log_error(self, error: Exception, operation: str = None, **kwargs):
        """
        Log de un error con contexto.
        
        Args:
            error: Excepción ocurrida
            operation: Operación donde ocurrió el error
            **kwargs: Contexto adicional
        """
        context = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            **kwargs
        }
        
        if operation:
            context['operation'] = operation
        
        self.logger.error("Error occurred", **context)
    
    def log_performance(self, operation: str, duration: float, **kwargs):
        """
        Log de métricas de performance.
        
        Args:
            operation: Nombre de la operación
            duration: Duración en segundos
            **kwargs: Métricas adicionales
        """
        self.logger.info(
            f"Performance: {operation}",
            duration_seconds=duration,
            **kwargs
        )
