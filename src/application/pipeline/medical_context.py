"""
Medical Pipeline Context

Context object for carrying state through medical diagnosis pipeline stages.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime
import uuid

from ...domain.value_objects.image_data import ImageData
from ...domain.entities.medical_diagnosis import (
    DermatologyDiagnosis,
    ChestXrayDiagnosis,
)


@dataclass
class MedicalPipelineContext:
    """
    Context for medical diagnosis pipelines.
    
    Carries state and results between pipeline stages.
    Similar to PipelineContext but tailored for medical image classification.
    
    Attributes:
        request_id: Unique identifier for this request
        image: Input image data
        diagnosis_type: Type of diagnosis (dermatology or chest_xray)
        options: Additional options for processing
        classification_result: Raw classification output from CNN
        generated_response: Generated text explanation
        dermatology_diagnosis: Structured dermatology result
        chest_xray_diagnosis: Structured chest X-ray result
        start_time: Pipeline start timestamp
        errors: Accumulated errors
    """
    
    # Input
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    image: Optional[ImageData] = None
    diagnosis_type: str = ""  # "dermatology" or "chest_xray"
    options: Dict[str, Any] = field(default_factory=dict)
    
    # Stage results
    classification_result: Optional[Dict[str, Any]] = None
    generated_response: Optional[str] = None
    
    # Diagnosis entities
    dermatology_diagnosis: Optional[DermatologyDiagnosis] = None
    chest_xray_diagnosis: Optional[ChestXrayDiagnosis] = None
    
    # Metadata
    start_time: datetime = field(default_factory=datetime.now)
    errors: list = field(default_factory=list)
    stage_timings: Dict[str, float] = field(default_factory=dict)
    
    @property
    def has_classification(self) -> bool:
        """Check if classification has been performed."""
        return self.classification_result is not None
    
    @property
    def has_errors(self) -> bool:
        """Check if any errors occurred."""
        return len(self.errors) > 0
    
    def add_error(self, stage: str, error: str) -> None:
        """Add an error from a specific stage."""
        self.errors.append({
            "stage": stage,
            "error": error,
            "timestamp": datetime.now().isoformat(),
        })
    
    def record_stage_timing(self, stage: str, duration_ms: float) -> None:
        """Record timing for a pipeline stage."""
        self.stage_timings[stage] = duration_ms
