"""
Main FastAPI application for Aura Document Analyzer.
Provides REST API for document processing and AI analysis.
"""

import time
from contextlib import asynccontextmanager
from typing import Dict, Any
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.utils import get_openapi

from src.core.config import settings
from src.core.logging import configure_logging, get_logger
from .endpoints import router
from .dependencies import initialize_services, shutdown_services
from .models import ErrorResponse
from ..web.routes import web_router


# Configure logging
configure_logging()
logger = get_logger("api.main")

# Application startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Aura Document Analyzer API")
    
    try:
        # Initialize services
        await initialize_services()
        logger.info("Services initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize services: {str(e)}")
        # Continue anyway for demo purposes
    
    yield
    
    # Shutdown
    logger.info("Shutting down Aura Document Analyzer API")
    try:
        await shutdown_services()
        logger.info("Services shutdown completed")
    except Exception as e:
        logger.error(f"Error during shutdown: {str(e)}")


# Create FastAPI application
app = FastAPI(
    title="Aura Document Analyzer API",
    description="""
    ## Scalable Document Analysis System with AI

    A comprehensive REST API for document processing, analysis, and semantic search using artificial intelligence.

    ### Features

    * **Document Processing**: Upload and process PDF, DOCX, JSON, and TXT files
    * **AI Analysis**: Automatic classification and named entity recognition
    * **Semantic Search**: Advanced search using embeddings and vector similarity
    * **Batch Processing**: Handle multiple documents efficiently
    * **Real-time Stats**: Monitor system performance and usage

    ### AI Models

    * **Classification**: TF-IDF + SVM for document categorization
    * **NER**: spaCy + custom patterns for entity extraction
    * **Embeddings**: sentence-transformers for semantic understanding
    * **Search**: FAISS for efficient vector similarity search

    ### Authentication

    Currently in development mode. Production deployment will include JWT authentication.
    """,
    version="0.1.0",
    contact={
        "name": "Aura Research Team",
        "email": "dev@auraresearch.ai",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
    lifespan=lifespan
)

# Mount static files
static_dir = Path(__file__).parent.parent / "web" / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time to response headers."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests."""
    start_time = time.time()
    
    # Log request
    logger.info(
        "Request started",
        method=request.method,
        url=str(request.url),
        client_ip=request.client.host if request.client else None
    )
    
    response = await call_next(request)
    
    # Log response
    process_time = time.time() - start_time
    logger.info(
        "Request completed",
        method=request.method,
        url=str(request.url),
        status_code=response.status_code,
        process_time=process_time
    )
    
    return response


# Include API routes
app.include_router(
    router,
    prefix=settings.api_prefix,
    tags=["documents", "ai", "search"]
)

# Include web interface routes
app.include_router(web_router, tags=["Web Interface"])


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Aura Document Analyzer API",
        "version": "0.1.0",
        "description": "Scalable Document Analysis System with AI",
        "docs_url": "/docs",
        "health_url": f"{settings.api_prefix}/health",
        "features": [
            "Document Processing (PDF, DOCX, JSON, TXT)",
            "AI Classification and NER",
            "Semantic Search",
            "Batch Processing",
            "Real-time Statistics"
        ],
        "ai_models": {
            "classification": "TF-IDF + SVM",
            "ner": "spaCy + Custom Patterns",
            "embeddings": "sentence-transformers",
            "search": "FAISS Vector Search"
        }
    }


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(
        "Unhandled exception",
        method=request.method,
        url=str(request.url),
        error=str(exc),
        exc_info=True
    )
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            message="Internal server error",
            error_code="INTERNAL_ERROR",
            details={"error": str(exc)} if settings.debug else {}
        ).model_dump(mode='json')
    )


# HTTP exception handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP exception handler."""
    logger.warning(
        "HTTP exception",
        method=request.method,
        url=str(request.url),
        status_code=exc.status_code,
        detail=exc.detail
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            message=exc.detail,
            error_code=f"HTTP_{exc.status_code}"
        ).dict()
    )


# Custom OpenAPI schema
def custom_openapi():
    """Generate custom OpenAPI schema."""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="Aura Document Analyzer API",
        version="0.1.0",
        description=app.description,
        routes=app.routes,
    )
    
    # Add custom schema information
    openapi_schema["info"]["x-logo"] = {
        "url": "https://auraresearch.ai/logo.png"
    }
    
    # Add server information
    openapi_schema["servers"] = [
        {
            "url": "http://localhost:8000",
            "description": "Development server"
        },
        {
            "url": "https://api.auraresearch.ai",
            "description": "Production server"
        }
    ]
    
    # Add security schemes (for future authentication)
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT"
        },
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key"
        }
    }
    
    # Add example responses
    openapi_schema["components"]["examples"] = {
        "DocumentUploadSuccess": {
            "summary": "Successful document upload",
            "value": {
                "success": True,
                "message": "Document uploaded successfully",
                "document_id": "123e4567-e89b-12d3-a456-426614174000",
                "filename": "contract.pdf",
                "file_size": 1024000,
                "document_type": "pdf",
                "status": "processing"
            }
        },
        "SearchResults": {
            "summary": "Search results",
            "value": {
                "success": True,
                "query": "machine learning",
                "results": [
                    {
                        "document_id": "123e4567-e89b-12d3-a456-426614174000",
                        "title": "ML Research Paper",
                        "score": 0.95,
                        "snippet": "This paper discusses machine learning algorithms..."
                    }
                ],
                "total_found": 1,
                "search_time": 0.045
            }
        }
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


# Health check for load balancers
@app.get("/ping")
async def ping():
    """Simple ping endpoint for health checks."""
    return {"status": "ok", "timestamp": time.time()}


# Metrics endpoint (basic)
@app.get("/metrics")
async def metrics():
    """Basic metrics endpoint."""
    return {
        "uptime_seconds": time.time(),
        "requests_total": "N/A",  # Would track in production
        "active_connections": "N/A",  # Would track in production
        "memory_usage": "N/A",  # Would track in production
        "cpu_usage": "N/A"  # Would track in production
    }


# Development helper endpoints
if settings.debug:
    @app.get("/debug/config")
    async def debug_config():
        """Debug endpoint to show configuration (development only)."""
        return {
            "environment": settings.environment,
            "debug": settings.debug,
            "api_prefix": settings.api_prefix,
            "max_workers": settings.max_workers,
            "max_file_size": settings.max_file_size,
            "embedding_model": settings.embedding_model,
            "cors_origins": settings.cors_origins
        }
    
    @app.get("/debug/routes")
    async def debug_routes():
        """Debug endpoint to show all routes (development only)."""
        routes = []
        for route in app.routes:
            if hasattr(route, 'methods') and hasattr(route, 'path'):
                routes.append({
                    "path": route.path,
                    "methods": list(route.methods),
                    "name": route.name
                })
        return {"routes": routes}


# Main entry point
def main():
    """Main entry point for running the application."""
    import uvicorn
    
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
        access_log=True
    )


if __name__ == "__main__":
    main()
