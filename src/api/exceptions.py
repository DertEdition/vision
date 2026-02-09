"""
API Exception Handlers

Custom exception handlers for the Drug Image Analysis API.
Maps domain exceptions to appropriate HTTP responses.
"""

import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from typing import Union

from ..domain.exceptions import (
    DomainException,
    InvalidImageError,
    ValidationError,
    VisionAnalysisError,
    TextExtractionError,
    EntityExtractionError,
    KnowledgeRetrievalError,
    ResponseGenerationError,
    PipelineConfigurationError,
)
from .models.responses import ErrorResponse, ErrorDetail


logger = logging.getLogger(__name__)


def create_error_response(
    error: Union[Exception, DomainException],
    status_code: int,
    request_id: str = None
) -> ErrorResponse:
    """Create a standardized error response."""
    
    if isinstance(error, DomainException):
        return ErrorResponse(
            success=False,
            error=error.message,
            error_type=error.__class__.__name__,
            details=[
                ErrorDetail(
                    error_type=error.__class__.__name__,
                    message=error.message,
                    is_recoverable=error.is_recoverable
                )
            ],
            request_id=request_id
        )
    else:
        return ErrorResponse(
            success=False,
            error=str(error),
            error_type=error.__class__.__name__,
            details=None,
            request_id=request_id
        )


def register_exception_handlers(app: FastAPI) -> None:
    """
    Register all exception handlers with the FastAPI application.
    
    Args:
        app: FastAPI application instance
    """
    
    @app.exception_handler(InvalidImageError)
    async def invalid_image_handler(request: Request, exc: InvalidImageError):
        """Handle invalid image errors (400 Bad Request)."""
        logger.warning(f"Invalid image error: {exc.message}")
        return JSONResponse(
            status_code=400,
            content=create_error_response(exc, 400).model_dump()
        )
    
    @app.exception_handler(ValidationError)
    async def validation_error_handler(request: Request, exc: ValidationError):
        """Handle validation errors (422 Unprocessable Entity)."""
        logger.warning(f"Validation error: {exc.message}")
        return JSONResponse(
            status_code=422,
            content=create_error_response(exc, 422).model_dump()
        )
    
    @app.exception_handler(VisionAnalysisError)
    async def vision_error_handler(request: Request, exc: VisionAnalysisError):
        """Handle vision analysis errors (500 Internal Server Error)."""
        logger.error(f"Vision analysis error: {exc.message}")
        return JSONResponse(
            status_code=500,
            content=create_error_response(exc, 500).model_dump()
        )
    
    @app.exception_handler(TextExtractionError)
    async def text_extraction_error_handler(request: Request, exc: TextExtractionError):
        """Handle OCR errors (500 Internal Server Error)."""
        logger.error(f"Text extraction error: {exc.message}")
        return JSONResponse(
            status_code=500,
            content=create_error_response(exc, 500).model_dump()
        )
    
    @app.exception_handler(EntityExtractionError)
    async def entity_extraction_error_handler(request: Request, exc: EntityExtractionError):
        """Handle entity extraction errors (500 Internal Server Error)."""
        logger.error(f"Entity extraction error: {exc.message}")
        return JSONResponse(
            status_code=500,
            content=create_error_response(exc, 500).model_dump()
        )
    
    @app.exception_handler(KnowledgeRetrievalError)
    async def knowledge_retrieval_error_handler(request: Request, exc: KnowledgeRetrievalError):
        """Handle RAG errors (500 Internal Server Error)."""
        logger.error(f"Knowledge retrieval error: {exc.message}")
        return JSONResponse(
            status_code=500,
            content=create_error_response(exc, 500).model_dump()
        )
    
    @app.exception_handler(ResponseGenerationError)
    async def response_generation_error_handler(request: Request, exc: ResponseGenerationError):
        """Handle LLM errors (500 Internal Server Error)."""
        logger.error(f"Response generation error: {exc.message}")
        return JSONResponse(
            status_code=500,
            content=create_error_response(exc, 500).model_dump()
        )
    
    @app.exception_handler(PipelineConfigurationError)
    async def pipeline_config_error_handler(request: Request, exc: PipelineConfigurationError):
        """Handle pipeline configuration errors (500 Internal Server Error)."""
        logger.error(f"Pipeline configuration error: {exc.message}")
        return JSONResponse(
            status_code=500,
            content=create_error_response(exc, 500).model_dump()
        )
    
    @app.exception_handler(DomainException)
    async def domain_exception_handler(request: Request, exc: DomainException):
        """Handle any other domain exceptions (500 Internal Server Error)."""
        logger.error(f"Domain exception: {exc.message}")
        return JSONResponse(
            status_code=500,
            content=create_error_response(exc, 500).model_dump()
        )
    
    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        """Handle unexpected exceptions (500 Internal Server Error)."""
        logger.exception(f"Unexpected error: {exc}")
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                success=False,
                error="An unexpected error occurred",
                error_type="InternalServerError",
                details=None,
                request_id=None
            ).model_dump()
        )
