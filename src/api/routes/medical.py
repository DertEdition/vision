"""
Medical Analysis API Routes

REST API endpoints for medical image analysis (dermatology + chest X-ray).
"""

import logging
import time
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException

from ..models.requests import MedicalAnalyzeFromBase64Request, MedicalAnalyzeFromPathRequest
from ..models.responses import MedicalAnalysisResponse
from ..dependencies import get_medical_analysis_service
from ...application.services.medical_analysis_service import MedicalAnalysisService
from ...domain.entities.medical_diagnosis import MedicalDiagnosisResult

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analyze/medical", tags=["Medical Diagnosis"])


def _build_response(result: MedicalDiagnosisResult, processing_time_ms: float) -> MedicalAnalysisResponse:
    """Convert domain result to API response."""
    return MedicalAnalysisResponse(
        request_id=result.request_id or "",
        success=result.is_successful,
        diagnosis_type=result.diagnosis_type,
        dermatology={
            "malignancy": result.dermatology.malignancy,
            "malignancy_confidence": round(result.dermatology.malignancy_confidence, 4),
            "disease_type": result.dermatology.disease_type,
            "disease_type_confidence": round(result.dermatology.disease_type_confidence, 4),
            "super_class": result.dermatology.super_class,
            "main_class": result.dermatology.main_class,
            "recommendations": result.dermatology.recommendations,
        } if result.dermatology else None,
        chest_xray={
            "findings": result.chest_xray.findings,
            "finding_probabilities": {
                k: round(v, 4) for k, v in result.chest_xray.finding_probabilities.items()
            },
            "has_abnormality": result.chest_xray.has_abnormality,
            "primary_finding": result.chest_xray.primary_finding,
            "recommendations": result.chest_xray.recommendations,
        } if result.chest_xray else None,
        explanation=result.explanation,
        confidence=str(result.overall_confidence),
        warnings=result.warnings,
        disclaimer=result.disclaimer,
        processing_time_ms=round(processing_time_ms, 2),
        errors=result.errors if result.has_errors else None,
    )


# ============================================================
# Dermatology Endpoints
# ============================================================


@router.post("/dermatology/upload", response_model=MedicalAnalysisResponse)
async def analyze_dermatology_upload(
    file: UploadFile = File(...),
    service: MedicalAnalysisService = Depends(get_medical_analysis_service),
):
    """
    Analyze a skin lesion image for dermatological diagnosis.
    
    Upload a dermoscopic or clinical skin image to get:
    - Malignancy classification (benign/malignant/indeterminate)
    - Disease type classification
    - Confidence scores
    - Medical recommendations
    """
    start = time.time()
    
    # Validate file type
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Expected an image file.",
        )
    
    try:
        image_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {e}")
    
    # Determine format
    format = "jpeg"
    if file.filename:
        ext = file.filename.rsplit(".", 1)[-1].lower()
        format = "jpeg" if ext == "jpg" else ext
    
    result = service.analyze_dermatology(image_bytes, format=format)
    processing_time = (time.time() - start) * 1000
    
    return _build_response(result, processing_time)


@router.post("/dermatology/base64", response_model=MedicalAnalysisResponse)
async def analyze_dermatology_base64(
    request: MedicalAnalyzeFromBase64Request,
    service: MedicalAnalysisService = Depends(get_medical_analysis_service),
):
    """
    Analyze a base64-encoded skin lesion image for dermatological diagnosis.
    """
    start = time.time()
    
    result = service.analyze_from_base64(
        image_base64=request.image_base64,
        diagnosis_type="dermatology",
        format=request.format or "jpeg",
        options=request.options,
    )
    processing_time = (time.time() - start) * 1000
    
    return _build_response(result, processing_time)


# ============================================================
# Chest X-ray Endpoints
# ============================================================


@router.post("/chest-xray/upload", response_model=MedicalAnalysisResponse)
async def analyze_chest_xray_upload(
    file: UploadFile = File(...),
    service: MedicalAnalysisService = Depends(get_medical_analysis_service),
):
    """
    Analyze a chest X-ray image for thoracic disease diagnosis.
    
    Upload a chest X-ray image to get:
    - Multi-label disease detection (14 thoracic conditions)
    - Finding probabilities
    - Medical recommendations
    """
    start = time.time()
    
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Expected an image file.",
        )
    
    try:
        image_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {e}")
    
    format = "jpeg"
    if file.filename:
        ext = file.filename.rsplit(".", 1)[-1].lower()
        format = "jpeg" if ext == "jpg" else ext
    
    result = service.analyze_chest_xray(image_bytes, format=format)
    processing_time = (time.time() - start) * 1000
    
    return _build_response(result, processing_time)


@router.post("/chest-xray/base64", response_model=MedicalAnalysisResponse)
async def analyze_chest_xray_base64(
    request: MedicalAnalyzeFromBase64Request,
    service: MedicalAnalysisService = Depends(get_medical_analysis_service),
):
    """
    Analyze a base64-encoded chest X-ray image for thoracic disease diagnosis.
    """
    start = time.time()
    
    result = service.analyze_from_base64(
        image_base64=request.image_base64,
        diagnosis_type="chest_xray",
        format=request.format or "jpeg",
        options=request.options,
    )
    processing_time = (time.time() - start) * 1000
    
    return _build_response(result, processing_time)


# ============================================================
# General / Path-based Endpoint
# ============================================================


@router.post("/from-path", response_model=MedicalAnalysisResponse)
async def analyze_from_path(
    request: MedicalAnalyzeFromPathRequest,
    service: MedicalAnalysisService = Depends(get_medical_analysis_service),
):
    """
    Analyze a medical image from a file path on the server.
    
    Specify the diagnosis_type as 'dermatology' or 'chest_xray'.
    """
    start = time.time()
    
    if request.diagnosis_type not in ("dermatology", "chest_xray"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid diagnosis_type: {request.diagnosis_type}. "
                   "Must be 'dermatology' or 'chest_xray'.",
        )
    
    result = service.analyze_from_file(
        file_path=request.file_path,
        diagnosis_type=request.diagnosis_type,
        options=request.options,
    )
    processing_time = (time.time() - start) * 1000
    
    return _build_response(result, processing_time)
