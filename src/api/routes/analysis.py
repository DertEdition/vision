"""
Analysis Routes

Main drug image analysis endpoints.
"""

import logging
from typing import Optional
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from fastapi.responses import JSONResponse

from ..models.requests import AnalyzeFromPathRequest, AnalyzeFromBase64Request
from ..models.responses import AnalysisResponse, DrugInfoResponse, StageTimingResponse
from ..dependencies import get_analysis_service
from ...application.services import DrugAnalysisService
from ...domain.entities.pipeline_result import PipelineResult


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analyze", tags=["Analysis"])


def _convert_result_to_response(result: PipelineResult) -> AnalysisResponse:
    """
    Convert PipelineResult to AnalysisResponse.
    
    Maps the internal domain result to the API response model.
    """
    user_response = result.get_user_response()
    
    # Extract drug info if available
    drug_info = None
    if drug_data := user_response.get("drug"):
        drug_info = DrugInfoResponse(
            name=drug_data.get("name"),
            active_ingredients=drug_data.get("active_ingredients", []),
            dosage_form=drug_data.get("dosage_form"),
            strength=drug_data.get("strength"),
            manufacturer=drug_data.get("manufacturer")
        )
    
    # Extract stage timings for debugging
    stage_timings = None
    if result.stage_statuses:
        stage_timings = [
            StageTimingResponse(
                stage=sr.stage.value,
                status=sr.status.value,
                duration_ms=sr.duration_ms
            )
            for sr in result.stage_statuses.values()
        ]
    
    # Extract errors if any
    errors = None
    if result.has_errors:
        errors = [error.to_dict() for error in result.errors]
    
    return AnalysisResponse(
        request_id=result.request_id or "unknown",
        success=result.is_successful,
        drug=drug_info,
        explanation=user_response.get("explanation"),
        confidence=user_response.get("confidence"),
        warnings=user_response.get("warnings", []),
        disclaimer=user_response.get("disclaimer", ""),
        processing_time_ms=result.total_processing_time_ms,
        stage_timings=stage_timings,
        errors=errors
    )


@router.post(
    "/upload",
    response_model=AnalysisResponse,
    summary="Analyze Drug Image from Upload",
    description="Upload a drug image file for analysis. Supports JPEG, PNG, WebP, and other common image formats."
)
async def analyze_upload(
    file: UploadFile = File(
        ...,
        description="Drug image file to analyze"
    ),
    service: DrugAnalysisService = Depends(get_analysis_service)
):
    """
    Analyze a drug image from file upload.
    
    This endpoint accepts multipart/form-data with an image file.
    The image is processed through the complete pipeline:
    VISION → OCR → ENTITY → RAG → LLM
    
    Returns:
        AnalysisResponse with drug identification and explanation
    """
    logger.info(f"Received upload request: {file.filename}")
    
    # Validate content type
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid content type: {file.content_type}. Expected image/*"
        )
    
    # Read file bytes
    try:
        image_bytes = await file.read()
    except Exception as e:
        logger.error(f"Failed to read uploaded file: {e}")
        raise HTTPException(
            status_code=400,
            detail="Failed to read uploaded file"
        )
    
    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty"
        )
    
    # Determine format from filename
    format_hint = None
    if file.filename:
        suffix = Path(file.filename).suffix.lower().lstrip(".")
        if suffix in {"jpg", "jpeg", "png", "webp", "bmp", "gif"}:
            format_hint = suffix
    
    # Run analysis
    logger.info(f"Processing image: {file.filename} ({len(image_bytes)} bytes)")
    result = service.analyze_from_bytes(image_bytes, format=format_hint)
    
    return _convert_result_to_response(result)


@router.post(
    "/base64",
    response_model=AnalysisResponse,
    summary="Analyze Drug Image from Base64",
    description="Analyze a drug image encoded as base64 string."
)
async def analyze_base64(
    request: AnalyzeFromBase64Request,
    service: DrugAnalysisService = Depends(get_analysis_service)
):
    """
    Analyze a drug image from base64-encoded data.
    
    This is useful for frontend integrations where images
    are captured and encoded in the browser.
    
    Returns:
        AnalysisResponse with drug identification and explanation
    """
    logger.info("Received base64 analysis request")
    
    # Run analysis
    result = service.analyze_from_base64(
        request.image_base64,
        format=request.format,
        options=request.options
    )
    
    return _convert_result_to_response(result)


@router.post(
    "/path",
    response_model=AnalysisResponse,
    summary="Analyze Drug Image from Server Path",
    description="Analyze a drug image from a file path on the server. Useful for batch processing or testing."
)
async def analyze_path(
    request: AnalyzeFromPathRequest,
    service: DrugAnalysisService = Depends(get_analysis_service)
):
    """
    Analyze a drug image from a file path on the server.
    
    This endpoint is useful for:
    - Testing with local test images
    - Batch processing scenarios
    - Integration with file-based workflows
    
    Returns:
        AnalysisResponse with drug identification and explanation
    """
    logger.info(f"Received path analysis request: {request.file_path}")
    
    # Run analysis
    result = service.analyze_from_file(
        request.file_path,
        options=request.options
    )
    
    return _convert_result_to_response(result)
