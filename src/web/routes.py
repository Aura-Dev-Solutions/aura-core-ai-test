"""
Web interface routes for Aura Document Analyzer.

This module provides the web interface routes that serve HTML templates
and handle web-specific functionality for the document analysis system.
"""

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
import logging

# Setup logging
logger = logging.getLogger(__name__)

# Initialize router
web_router = APIRouter()

# Setup templates
templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

@web_router.get("/", response_class=HTMLResponse)
async def home_page(request: Request):
    """
    Serve the main home page with system overview and quick start guide.
    
    Args:
        request: FastAPI request object
        
    Returns:
        HTMLResponse: Rendered home page template
    """
    try:
        return templates.TemplateResponse("index.html", {"request": request})
    except Exception as e:
        logger.error(f"Error serving home page: {e}")
        raise HTTPException(status_code=500, detail="Error loading home page")

@web_router.get("/analyze", response_class=HTMLResponse)
async def analyze_page(request: Request):
    """
    Serve the document analysis page with upload and analysis functionality.
    
    Args:
        request: FastAPI request object
        
    Returns:
        HTMLResponse: Rendered analysis page template
    """
    try:
        return templates.TemplateResponse("analyze.html", {"request": request})
    except Exception as e:
        logger.error(f"Error serving analyze page: {e}")
        raise HTTPException(status_code=500, detail="Error loading analyze page")

@web_router.get("/samples", response_class=HTMLResponse)
async def samples_page(request: Request):
    """
    Serve the sample files page with downloadable test documents.
    
    Args:
        request: FastAPI request object
        
    Returns:
        HTMLResponse: Rendered samples page template
    """
    try:
        return templates.TemplateResponse("samples.html", {"request": request})
    except Exception as e:
        logger.error(f"Error serving samples page: {e}")
        raise HTTPException(status_code=500, detail="Error loading samples page")

@web_router.get("/guide", response_class=HTMLResponse)
async def guide_page(request: Request):
    """
    Serve the user guide page with detailed instructions and documentation.
    
    Args:
        request: FastAPI request object
        
    Returns:
        HTMLResponse: Rendered guide page template
    """
    try:
        return templates.TemplateResponse("guide.html", {"request": request})
    except Exception as e:
        logger.error(f"Error serving guide page: {e}")
        raise HTTPException(status_code=500, detail="Error loading guide page")
