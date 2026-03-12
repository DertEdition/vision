"""
Medical Diagnosis Entities

Domain entities for medical image diagnosis results (dermatology + chest X-ray).
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime

from ..value_objects.confidence_score import ConfidenceScore


@dataclass
class DermatologyDiagnosis:
    """
    Dermatology diagnosis result from CNN classification.
    
    Attributes:
        malignancy: Classification result (benign, malignant, indeterminate)
        malignancy_confidence: Confidence score for malignancy classification
        disease_type: Specific disease sub-class (e.g., melanoma, basal_cell_carcinoma)
        disease_type_confidence: Confidence for disease type classification
        super_class: Top-level class (melanocytic, nonmelanocytic)
        main_class: Main class (banal, dysplastic, melanoma, keratinocytic, vascular)
        recommendations: List of medical recommendations
    """
    
    malignancy: str = "unknown"  # benign, malignant, indeterminate
    malignancy_confidence: float = 0.0
    disease_type: str = "unknown"
    disease_type_confidence: float = 0.0
    super_class: str = "unknown"  # melanocytic, nonmelanocytic
    main_class: str = "unknown"
    recommendations: List[str] = field(default_factory=list)
    
    @property
    def is_malignant(self) -> bool:
        """Check if the diagnosis indicates malignancy."""
        return self.malignancy == "malignant"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "malignancy": self.malignancy,
            "malignancy_confidence": round(self.malignancy_confidence, 4),
            "disease_type": self.disease_type,
            "disease_type_confidence": round(self.disease_type_confidence, 4),
            "super_class": self.super_class,
            "main_class": self.main_class,
            "recommendations": self.recommendations,
        }


@dataclass
class ChestXrayDiagnosis:
    """
    Chest X-ray diagnosis result from CNN classification.
    
    Attributes:
        findings: List of detected conditions
        finding_probabilities: Mapping of condition name → probability
        has_abnormality: Whether any abnormality was detected
        primary_finding: Most likely finding
        recommendations: List of medical recommendations
    """
    
    findings: List[str] = field(default_factory=list)
    finding_probabilities: Dict[str, float] = field(default_factory=dict)
    has_abnormality: bool = False
    primary_finding: str = "No Finding"
    recommendations: List[str] = field(default_factory=list)
    
    # Standard NIH Chest X-ray 14 disease labels
    ALL_FINDINGS = [
        "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
        "Mass", "Nodule", "Pneumonia", "Pneumothorax",
        "Consolidation", "Edema", "Emphysema", "Fibrosis",
        "Pleural_Thickening", "Hernia"
    ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "findings": self.findings,
            "finding_probabilities": {
                k: round(v, 4) for k, v in self.finding_probabilities.items()
            },
            "has_abnormality": self.has_abnormality,
            "primary_finding": self.primary_finding,
            "recommendations": self.recommendations,
        }


# Default medical disclaimer for diagnosis results
MEDICAL_DIAGNOSIS_DISCLAIMER = """
⚠️ ÖNEMLİ UYARI: Bu teşhis bilgileri yalnızca eğitim amaçlıdır ve profesyonel tıbbi 
tavsiye, teşhis veya tedavinin yerine GEÇMEZ. Herhangi bir sağlık kararı vermeden önce 
mutlaka bir sağlık uzmanına danışın. Bu sistem yapay zeka destekli bir ön tarama aracıdır 
ve kesin teşhis için yeterli değildir.
""".strip()


@dataclass
class MedicalDiagnosisResult:
    """
    Unified result entity for medical image diagnosis pipelines.
    
    Encapsulates the complete output of either dermatology or
    chest X-ray diagnosis pipeline.
    
    Attributes:
        diagnosis_type: Type of diagnosis (dermatology or chest_xray)
        dermatology: Dermatology diagnosis if applicable
        chest_xray: Chest X-ray diagnosis if applicable
        explanation: Generated explanation text
        warnings: Safety warnings
        disclaimer: Medical disclaimer
        overall_confidence: Overall confidence score
        request_id: Unique request identifier
        created_at: Result creation timestamp
        total_processing_time_ms: Total pipeline execution time
        errors: List of errors if any
    """
    
    # Diagnosis type
    diagnosis_type: str = ""  # "dermatology" or "chest_xray"
    
    # Specific diagnosis results
    dermatology: Optional[DermatologyDiagnosis] = None
    chest_xray: Optional[ChestXrayDiagnosis] = None
    
    # Generated content
    explanation: str = ""
    warnings: List[str] = field(default_factory=list)
    disclaimer: str = MEDICAL_DIAGNOSIS_DISCLAIMER
    
    # Metadata
    overall_confidence: ConfidenceScore = field(default_factory=ConfidenceScore.zero)
    request_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    total_processing_time_ms: float = 0.0
    errors: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def is_successful(self) -> bool:
        """Check if diagnosis completed successfully."""
        if self.diagnosis_type == "dermatology":
            return self.dermatology is not None and self.dermatology.malignancy != "unknown"
        elif self.diagnosis_type == "chest_xray":
            return self.chest_xray is not None
        return False
    
    @property
    def has_errors(self) -> bool:
        """Check if any errors occurred."""
        return len(self.errors) > 0
    
    def get_user_response(self) -> Dict[str, Any]:
        """
        Get formatted response for end user.
        
        Returns:
            Dictionary with user-facing diagnosis information
        """
        response = {
            "success": self.is_successful,
            "diagnosis_type": self.diagnosis_type,
            "disclaimer": self.disclaimer,
        }
        
        if self.dermatology:
            response["dermatology"] = self.dermatology.to_dict()
        
        if self.chest_xray:
            response["chest_xray"] = self.chest_xray.to_dict()
        
        if self.explanation:
            response["explanation"] = self.explanation
        
        if self.warnings:
            response["warnings"] = self.warnings
        
        response["confidence"] = str(self.overall_confidence)
        
        return response
    
    def __str__(self) -> str:
        status = "Success" if self.is_successful else "Failed"
        return (
            f"MedicalDiagnosisResult({status}: {self.diagnosis_type}, "
            f"confidence={self.overall_confidence})"
        )
