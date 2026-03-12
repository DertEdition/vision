"""
FastAPI Application Factory

Creates and configures the FastAPI application instance.
"""

import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

from .routes import analysis_router, health_router
from .routes.medical import router as medical_router
from .exceptions import register_exception_handlers
from .dependencies import get_config


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("Starting Drug Image Analysis API...")
    
    # Pre-load configuration (optional: also pre-load pipeline here)
    config = get_config()
    logger.info(f"API configured: host={config.api.host}, port={config.api.port}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Drug Image Analysis API...")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI application instance
    """
    # Get configuration
    config = get_config()
    
    # Create FastAPI app
    app = FastAPI(
        title="Drug Image Analysis API",
        description="""
REST API for analyzing pharmaceutical drug images and medical image diagnosis.

## Features

- **Drug Image Analysis**: Detect drug packaging, extract text, identify medications
- **Dermatology Diagnosis**: Classify skin lesions for malignancy and disease type
- **Chest X-ray Analysis**: Multi-label thoracic disease detection
- **Knowledge Retrieval**: Match against drug knowledge base
- **Response Generation**: Generate natural language explanations

## Usage

Upload a drug image, skin lesion photo, or chest X-ray to get detailed analysis.

⚠️ **Disclaimer**: This API is for educational purposes only and is not
a substitute for professional medical advice.
        """,
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Register exception handlers
    register_exception_handlers(app)
    
    # Include routers
    app.include_router(health_router)
    app.include_router(analysis_router)
    app.include_router(medical_router)
    
    # Mount static files for test UI
    static_dir = Path(__file__).resolve().parent.parent.parent / "static"
    if static_dir.is_dir():
        app.mount("/static", StaticFiles(directory=str(static_dir), html=True), name="static")
        
        @app.get("/", include_in_schema=False)
        async def root_redirect():
            """Redirect root to test UI."""
            return RedirectResponse(url="/static/test_ui.html")
    
    logger.info("FastAPI application created successfully")
    
    return app


# Create default application instance for uvicorn
app = create_app()
