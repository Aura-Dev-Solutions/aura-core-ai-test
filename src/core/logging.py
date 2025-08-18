"""
Structured logging configuration for Aura Document Analyzer.
Provides JSON and text logging with proper formatting and context.
"""

import logging
import sys
import time
from typing import Any, Dict, Optional

import structlog
from rich.console import Console
from rich.logging import RichHandler

from .config import settings


def configure_logging() -> None:
    """Configure structured logging for the application."""
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="ISO"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer() if settings.log_format == "json" 
            else structlog.dev.ConsoleRenderer(colors=True),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        context_class=dict,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard logging
    if settings.log_format == "json":
        logging.basicConfig(
            format="%(message)s",
            stream=sys.stdout,
            level=getattr(logging, settings.log_level.upper())
        )
    else:
        logging.basicConfig(
            format="%(message)s",
            level=getattr(logging, settings.log_level.upper()),
            handlers=[RichHandler(console=Console(stderr=True), show_path=False)]
        )


def get_logger(name: str = __name__) -> structlog.stdlib.BoundLogger:
    """Get a configured logger instance."""
    return structlog.get_logger(name)


class LoggerMixin:
    """Mixin to add logging capabilities to any class."""
    
    @property
    def logger(self) -> structlog.stdlib.BoundLogger:
        """Get logger for this class."""
        if not hasattr(self, '_logger'):
            self._logger = get_logger(self.__class__.__name__)
        return self._logger


def log_performance(operation: str):
    """Decorator to log performance metrics."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.info(
                    f"Operation completed: {operation}",
                    duration=duration,
                    function=func.__name__
                )
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(
                    f"Operation failed: {operation}",
                    duration=duration,
                    function=func.__name__,
                    error=str(e)
                )
                raise
        return wrapper
    return decorator
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="ISO"),
            _add_app_context,
            structlog.processors.JSONRenderer() if settings.log_format == "json" 
            else structlog.dev.ConsoleRenderer(colors=True),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.log_level.upper())
        ),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard logging
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format="%(message)s",
        handlers=[
            RichHandler(
                console=Console(stderr=True),
                show_time=False,
                show_path=False,
                markup=True,
            ) if settings.log_format == "text" else logging.StreamHandler(sys.stdout)
        ],
    )
    
    # Silence noisy loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


def _add_app_context(logger: Any, method_name: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Add application context to log events."""
    event_dict["app"] = settings.app_name
    event_dict["version"] = settings.app_version
    event_dict["environment"] = settings.environment
    return event_dict


def get_logger(name: Optional[str] = None) -> structlog.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)


class LoggerMixin:
    """Mixin class to add logging capabilities to any class."""
    
    @property
    def logger(self) -> structlog.BoundLogger:
        """Get logger instance for this class."""
        return get_logger(self.__class__.__name__)


# Performance logging decorator
def log_performance(operation: str):
    """Decorator to log operation performance."""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            logger = get_logger(func.__name__)
            start_time = structlog.get_logger().info("Operation started", operation=operation)
            
            try:
                result = await func(*args, **kwargs)
                logger.info("Operation completed", operation=operation)
                return result
            except Exception as e:
                logger.error("Operation failed", operation=operation, error=str(e))
                raise
        
        def sync_wrapper(*args, **kwargs):
            logger = get_logger(func.__name__)
            logger.info("Operation started", operation=operation)
            
            try:
                result = func(*args, **kwargs)
                logger.info("Operation completed", operation=operation)
                return result
            except Exception as e:
                logger.error("Operation failed", operation=operation, error=str(e))
                raise
        
        return async_wrapper if hasattr(func, '__code__') and func.__code__.co_flags & 0x80 else sync_wrapper
    return decorator
