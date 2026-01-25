"""
Pipeline Result Entity

Final output entity from the complete drug image analysis pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime

from .drug_info import DrugInfo
from .extraction_result import (
    VisionAnalysisResult,
    TextExtractionResult,
    EntityExtractionResult,
    KnowledgeRetrievalResult,
)
from ..value_objects.confidence_score import ConfidenceScore


class StageStatus(Enum):
    """Status of a pipeline stage execution."""
    
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    PARTIAL = "partial"  # Completed with some errors


class PipelineStage(Enum):
    """Enumeration of pipeline stages."""
    
    VISION_ANALYSIS = "vision_analysis"
    TEXT_EXTRACTION = "text_extraction"
    ENTITY_EXTRACTION = "entity_extraction"
    KNOWLEDGE_RETRIEVAL = "knowledge_retrieval"
    RESPONSE_GENERATION = "response_generation"
    SAFETY_CHECK = "safety_check"


@dataclass
class PipelineError:
    """
    Represents an error that occurred during pipeline execution.
    
    Attributes:
        stage: Pipeline stage where error occurred
        error_type: Type of error
        message: Human-readable error message
        details: Additional error details
        timestamp: When the error occurred
        is_recoverable: Whether pipeline can continue
    """
    
    stage: PipelineStage
    error_type: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)
    is_recoverable: bool = True
    
    def __str__(self) -> str:
        return f"[{self.stage.value}] {self.error_type}: {self.message}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "stage": self.stage.value,
            "error_type": self.error_type,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "is_recoverable": self.is_recoverable,
        }


@dataclass
class StageResult:
    """
    Metadata about a single stage execution.
    
    Attributes:
        stage: The pipeline stage
        status: Execution status
        start_time: When stage started
        end_time: When stage finished
        duration_ms: Execution duration in milliseconds
        error: Error if stage failed
    """
    
    stage: PipelineStage
    status: StageStatus = StageStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_ms: float = 0.0
    error: Optional[PipelineError] = None
    
    @property
    def is_successful(self) -> bool:
        """Check if stage completed successfully."""
        return self.status in {StageStatus.COMPLETED, StageStatus.PARTIAL}
    
    @property
    def is_failed(self) -> bool:
        """Check if stage failed."""
        return self.status == StageStatus.FAILED


# Default medical disclaimer
DEFAULT_DISCLAIMER = """
⚠️ IMPORTANT DISCLAIMER: This information is provided for educational purposes only 
and is NOT a substitute for professional medical advice, diagnosis, or treatment. 
Always consult a qualified healthcare provider or pharmacist before taking any medication. 
Do not start, stop, or change any treatment without professional medical advice.
""".strip()


@dataclass
class PipelineResult:
    """
    Final result from the complete drug image analysis pipeline.
    
    This entity encapsulates everything about the analysis:
    - Identified drug information
    - Generated user-friendly explanation
    - Safety warnings and disclaimers
    - Stage-by-stage results for debugging
    - Any errors that occurred
    
    Attributes:
        drug_info: Identified drug information
        explanation: Generated user-friendly explanation
        warnings: List of safety warnings
        disclaimer: Mandatory medical disclaimer
        stage_results: Results from each pipeline stage
        errors: Accumulated errors from all stages
        overall_confidence: Overall pipeline confidence
        request_id: Unique identifier for this request
        created_at: Timestamp of result creation
    """
    
    # Primary outputs
    drug_info: Optional[DrugInfo] = None
    explanation: str = ""
    warnings: List[str] = field(default_factory=list)
    disclaimer: str = DEFAULT_DISCLAIMER
    
    # Stage-specific results
    vision_result: Optional[VisionAnalysisResult] = None
    text_result: Optional[TextExtractionResult] = None
    entity_result: Optional[EntityExtractionResult] = None
    knowledge_result: Optional[KnowledgeRetrievalResult] = None
    
    # Pipeline metadata
    stage_statuses: Dict[PipelineStage, StageResult] = field(default_factory=dict)
    errors: List[PipelineError] = field(default_factory=list)
    overall_confidence: ConfidenceScore = field(default_factory=ConfidenceScore.zero)
    
    # Request tracking
    request_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    total_processing_time_ms: float = 0.0
    
    @property
    def is_successful(self) -> bool:
        """Check if pipeline completed successfully with drug identification."""
        return (
            self.drug_info is not None and
            self.drug_info.drug_name != "Unknown Drug" and
            len([e for e in self.errors if not e.is_recoverable]) == 0
        )
    
    @property
    def has_errors(self) -> bool:
        """Check if any errors occurred."""
        return len(self.errors) > 0
    
    @property
    def has_critical_errors(self) -> bool:
        """Check if any non-recoverable errors occurred."""
        return any(not e.is_recoverable for e in self.errors)
    
    @property
    def completed_stages(self) -> List[PipelineStage]:
        """Get list of successfully completed stages."""
        return [
            stage for stage, result in self.stage_statuses.items()
            if result.is_successful
        ]
    
    @property
    def failed_stages(self) -> List[PipelineStage]:
        """Get list of failed stages."""
        return [
            stage for stage, result in self.stage_statuses.items()
            if result.is_failed
        ]
    
    def add_error(self, error: PipelineError) -> None:
        """Add an error to the result."""
        self.errors.append(error)
    
    def add_warning(self, warning: str) -> None:
        """Add a safety warning."""
        if warning not in self.warnings:
            self.warnings.append(warning)
    
    def set_stage_status(
        self,
        stage: PipelineStage,
        status: StageStatus,
        duration_ms: float = 0.0,
        error: Optional[PipelineError] = None
    ) -> None:
        """Update status for a pipeline stage."""
        if stage not in self.stage_statuses:
            self.stage_statuses[stage] = StageResult(stage=stage)
        
        result = self.stage_statuses[stage]
        result.status = status
        result.duration_ms = duration_ms
        result.error = error
        
        if status == StageStatus.RUNNING:
            result.start_time = datetime.now()
        elif status in {StageStatus.COMPLETED, StageStatus.FAILED, StageStatus.PARTIAL}:
            result.end_time = datetime.now()
    
    def get_user_response(self) -> Dict[str, Any]:
        """
        Get formatted response for end user.
        
        Returns:
            Dictionary with user-facing information
        """
        response = {
            "success": self.is_successful,
            "disclaimer": self.disclaimer,
        }
        
        if self.drug_info:
            response["drug"] = {
                "name": self.drug_info.drug_name,
                "active_ingredients": self.drug_info.active_ingredients,
                "dosage_form": self.drug_info.dosage_form.value if self.drug_info.dosage_form else None,
                "strength": self.drug_info.strength,
                "manufacturer": self.drug_info.manufacturer,
            }
        
        if self.explanation:
            response["explanation"] = self.explanation
        
        if self.warnings:
            response["warnings"] = self.warnings
        
        if not self.is_successful:
            response["error_message"] = self._get_user_friendly_error()
        
        response["confidence"] = str(self.overall_confidence)
        
        return response
    
    def _get_user_friendly_error(self) -> str:
        """Generate user-friendly error message."""
        if not self.errors:
            return "Unable to analyze the image. Please try with a clearer image."
        
        # Prioritize most relevant error
        for error in self.errors:
            if error.stage == PipelineStage.VISION_ANALYSIS:
                return "Could not detect a drug package in the image. Please ensure the image clearly shows the medication packaging."
            elif error.stage == PipelineStage.TEXT_EXTRACTION:
                return "Could not read text from the image. Please try with a clearer, well-lit image."
            elif error.stage == PipelineStage.ENTITY_EXTRACTION:
                return "Could not identify the drug from the text. Please ensure the drug name is visible."
        
        return "An error occurred during analysis. Please try again with a different image."
    
    def get_debug_info(self) -> Dict[str, Any]:
        """
        Get detailed debug information.
        
        Returns:
            Dictionary with debugging information
        """
        return {
            "request_id": self.request_id,
            "created_at": self.created_at.isoformat(),
            "total_processing_time_ms": self.total_processing_time_ms,
            "overall_confidence": self.overall_confidence.value,
            "stages": {
                stage.value: {
                    "status": result.status.value,
                    "duration_ms": result.duration_ms,
                    "error": str(result.error) if result.error else None,
                }
                for stage, result in self.stage_statuses.items()
            },
            "errors": [e.to_dict() for e in self.errors],
            "completed_stages": [s.value for s in self.completed_stages],
            "failed_stages": [s.value for s in self.failed_stages],
        }
    
    def __str__(self) -> str:
        status = "Success" if self.is_successful else "Failed"
        drug_name = self.drug_info.drug_name if self.drug_info else "Unknown"
        return f"PipelineResult({status}: {drug_name}, confidence={self.overall_confidence})"
    
    @classmethod
    def create_error_result(
        cls,
        error_message: str,
        stage: PipelineStage = PipelineStage.VISION_ANALYSIS,
        request_id: Optional[str] = None
    ) -> "PipelineResult":
        """
        Create a result indicating pipeline failure.
        
        Args:
            error_message: Human-readable error message
            stage: Stage where failure occurred
            request_id: Request identifier
            
        Returns:
            PipelineResult indicating failure
        """
        result = cls(request_id=request_id)
        result.add_error(PipelineError(
            stage=stage,
            error_type="PipelineFailure",
            message=error_message,
            is_recoverable=False
        ))
        result.set_stage_status(stage, StageStatus.FAILED)
        return result
