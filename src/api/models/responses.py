"""
Pydantic Response Models

Response serialization schemas for the Drug Image Analysis API.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class DrugInfoResponse(BaseModel):
    """
    Drug identification result from the pipeline.
    """
    
    name: Optional[str] = Field(None, description="Identified drug name")
    active_ingredients: List[str] = Field(
        default_factory=list,
        description="List of active ingredients"
    )
    dosage_form: Optional[str] = Field(
        None,
        description="Drug form (tablet, capsule, syrup, etc.)"
    )
    strength: Optional[str] = Field(
        None,
        description="Drug strength/dosage (e.g., '500mg', '10mg/5ml')"
    )
    manufacturer: Optional[str] = Field(
        None,
        description="Pharmaceutical manufacturer"
    )


class StageTimingResponse(BaseModel):
    """
    Execution timing for a pipeline stage.
    """
    
    stage: str = Field(..., description="Pipeline stage name")
    status: str = Field(..., description="Execution status")
    duration_ms: float = Field(..., description="Stage execution time in milliseconds")


class AnalysisResponse(BaseModel):
    """
    Complete drug image analysis response.
    
    Contains the full result of the analysis pipeline including
    drug identification, explanation, warnings, and execution metadata.
    """
    
    request_id: str = Field(..., description="Unique request identifier for tracking")
    success: bool = Field(..., description="Whether analysis completed successfully")
    
    # Drug identification results
    drug: Optional[DrugInfoResponse] = Field(
        None,
        description="Identified drug information (null if identification failed)"
    )
    
    # LLM-generated content
    explanation: Optional[str] = Field(
        None,
        description="Natural language explanation of the drug and usage"
    )
    confidence: Optional[str] = Field(
        None,
        description="Confidence level of the identification (high/medium/low)"
    )
    
    # Safety information
    warnings: List[str] = Field(
        default_factory=list,
        description="Safety warnings and alerts"
    )
    disclaimer: str = Field(
        ...,
        description="Medical disclaimer (always present)"
    )
    
    # Execution metadata
    processing_time_ms: float = Field(
        ...,
        description="Total pipeline execution time in milliseconds"
    )
    stage_timings: Optional[List[StageTimingResponse]] = Field(
        None,
        description="Per-stage execution timings (for debugging)"
    )
    
    # Error information (only populated on failure)
    errors: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="List of errors if analysis failed"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "request_id": "550e8400-e29b-41d4-a716-446655440000",
                    "success": True,
                    "drug": {
                        "name": "Aspirin",
                        "active_ingredients": ["Acetylsalicylic acid"],
                        "dosage_form": "Tablet",
                        "strength": "500mg",
                        "manufacturer": "Bayer"
                    },
                    "explanation": "Aspirin (Acetylsalicylic acid) is a nonsteroidal anti-inflammatory drug...",
                    "confidence": "high",
                    "warnings": ["Do not use if allergic to NSAIDs"],
                    "disclaimer": "⚠️ This information is for educational purposes only...",
                    "processing_time_ms": 1234.56,
                    "stage_timings": None,
                    "errors": None
                }
            ]
        }
    }


class ErrorDetail(BaseModel):
    """
    Detailed error information.
    """
    
    stage: Optional[str] = Field(None, description="Pipeline stage where error occurred")
    error_type: str = Field(..., description="Error type/class name")
    message: str = Field(..., description="Human-readable error message")
    is_recoverable: bool = Field(True, description="Whether the error is recoverable")


class ErrorResponse(BaseModel):
    """
    Error response for failed requests.
    """
    
    success: bool = Field(False, description="Always false for error responses")
    error: str = Field(..., description="Main error message")
    error_type: str = Field(..., description="Error classification")
    details: Optional[List[ErrorDetail]] = Field(
        None,
        description="Detailed error information"
    )
    request_id: Optional[str] = Field(
        None,
        description="Request ID if available"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "success": False,
                    "error": "Image file not found",
                    "error_type": "InvalidImageError",
                    "details": None,
                    "request_id": None
                }
            ]
        }
    }


class HealthResponse(BaseModel):
    """
    Health check response.
    """
    
    status: str = Field(..., description="Health status", examples=["healthy", "unhealthy"])
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Server timestamp"
    )
    components: Optional[Dict[str, str]] = Field(
        None,
        description="Component health status"
    )
