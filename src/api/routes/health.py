"""
Health Check Routes

Health and readiness endpoints for monitoring.
"""

from fastapi import APIRouter, Depends
from datetime import datetime

from ..models.responses import HealthResponse
from ..dependencies import get_analysis_service


router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check if the API is healthy and ready to serve requests."
)
async def health_check():
    """
    Basic health check endpoint.
    
    Returns the API status and version.
    """
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.now(),
        components=None
    )


@router.get(
    "/health/ready",
    response_model=HealthResponse,
    summary="Readiness Check",
    description="Check if the API has all dependencies loaded and is ready to process requests."
)
async def readiness_check(service = Depends(get_analysis_service)):
    """
    Readiness check that verifies all components are loaded.
    
    This will trigger pipeline initialization on first call.
    """
    # If we get here, the service was successfully created
    # which means all ML models are loaded
    return HealthResponse(
        status="ready",
        version="1.0.0",
        timestamp=datetime.now(),
        components={
            "pipeline": "loaded",
            "service": "ready"
        }
    )
