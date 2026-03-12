"""
Medical Analysis Service

Application service for medical image analysis (dermatology + chest X-ray).
Analogous to DrugAnalysisService but for medical diagnosis pipelines.
"""

import logging
import base64
import time
from typing import Optional, Dict, Any
from pathlib import Path

from ..pipeline.medical_pipeline import MedicalPipelineOrchestrator
from ...domain.value_objects.image_data import ImageData
from ...domain.entities.medical_diagnosis import MedicalDiagnosisResult

logger = logging.getLogger(__name__)


class MedicalAnalysisService:
    """
    Service for medical image analysis.
    
    Provides a high-level interface for analyzing medical images
    using dermatology and chest X-ray pipelines.
    
    Attributes:
        _dermatology_pipeline: Pipeline for dermatology analysis
        _chest_xray_pipeline: Pipeline for chest X-ray analysis
    """
    
    def __init__(
        self,
        dermatology_pipeline: Optional[MedicalPipelineOrchestrator] = None,
        chest_xray_pipeline: Optional[MedicalPipelineOrchestrator] = None,
    ):
        """
        Initialize the medical analysis service.
        
        Args:
            dermatology_pipeline: Configured dermatology pipeline
            chest_xray_pipeline: Configured chest X-ray pipeline
        """
        self._dermatology_pipeline = dermatology_pipeline
        self._chest_xray_pipeline = chest_xray_pipeline
        
        enabled = []
        if dermatology_pipeline:
            enabled.append("dermatology")
        if chest_xray_pipeline:
            enabled.append("chest_xray")
        logger.info(f"MedicalAnalysisService initialized (enabled: {enabled})")
    
    def analyze_dermatology(
        self,
        image_bytes: bytes,
        format: str = "jpeg",
        options: Optional[Dict[str, Any]] = None
    ) -> MedicalDiagnosisResult:
        """
        Analyze a dermatology image.
        
        Args:
            image_bytes: Raw image bytes
            format: Image format
            options: Optional analysis configuration
            
        Returns:
            MedicalDiagnosisResult with dermatology diagnosis
        """
        if not self._dermatology_pipeline:
            return MedicalDiagnosisResult(
                diagnosis_type="dermatology",
                explanation="Dermatoloji analiz servisi yapılandırılmamış.",
                errors=[{"stage": "init", "error": "Dermatology pipeline not configured"}],
            )
        
        logger.info("Starting dermatology analysis")
        image_data = ImageData.from_bytes(image_bytes, format=format)
        return self._dermatology_pipeline.run(image_data, options)
    
    def analyze_chest_xray(
        self,
        image_bytes: bytes,
        format: str = "jpeg",
        options: Optional[Dict[str, Any]] = None
    ) -> MedicalDiagnosisResult:
        """
        Analyze a chest X-ray image.
        
        Args:
            image_bytes: Raw image bytes
            format: Image format
            options: Optional analysis configuration
            
        Returns:
            MedicalDiagnosisResult with chest X-ray diagnosis
        """
        if not self._chest_xray_pipeline:
            return MedicalDiagnosisResult(
                diagnosis_type="chest_xray",
                explanation="Göğüs röntgeni analiz servisi yapılandırılmamış.",
                errors=[{"stage": "init", "error": "Chest X-ray pipeline not configured"}],
            )
        
        logger.info("Starting chest X-ray analysis")
        image_data = ImageData.from_bytes(image_bytes, format=format)
        return self._chest_xray_pipeline.run(image_data, options)
    
    def analyze_from_base64(
        self,
        image_base64: str,
        diagnosis_type: str,
        format: str = "jpeg",
        options: Optional[Dict[str, Any]] = None
    ) -> MedicalDiagnosisResult:
        """
        Analyze an image from base64 encoding.
        
        Args:
            image_base64: Base64-encoded image
            diagnosis_type: "dermatology" or "chest_xray"
            format: Image format
            options: Optional analysis configuration
            
        Returns:
            MedicalDiagnosisResult
        """
        try:
            image_bytes = base64.b64decode(image_base64)
        except Exception as e:
            return MedicalDiagnosisResult(
                diagnosis_type=diagnosis_type,
                explanation="Geçersiz base64 görüntü verisi.",
                errors=[{"stage": "decode", "error": str(e)}],
            )
        
        if diagnosis_type == "dermatology":
            return self.analyze_dermatology(image_bytes, format, options)
        elif diagnosis_type == "chest_xray":
            return self.analyze_chest_xray(image_bytes, format, options)
        else:
            return MedicalDiagnosisResult(
                diagnosis_type=diagnosis_type,
                explanation=f"Bilinmeyen teşhis türü: {diagnosis_type}",
                errors=[{"stage": "routing", "error": f"Unknown diagnosis type: {diagnosis_type}"}],
            )
    
    def analyze_from_file(
        self,
        file_path: str,
        diagnosis_type: str,
        options: Optional[Dict[str, Any]] = None
    ) -> MedicalDiagnosisResult:
        """
        Analyze an image from a file path.
        
        Args:
            file_path: Path to the image file
            diagnosis_type: "dermatology" or "chest_xray"
            options: Optional analysis configuration
            
        Returns:
            MedicalDiagnosisResult
        """
        path = Path(file_path)
        
        if not path.exists():
            return MedicalDiagnosisResult(
                diagnosis_type=diagnosis_type,
                explanation=f"Dosya bulunamadı: {file_path}",
                errors=[{"stage": "file_read", "error": f"File not found: {file_path}"}],
            )
        
        try:
            image_bytes = path.read_bytes()
            format = path.suffix.lstrip(".").lower()
            if format == "jpg":
                format = "jpeg"
        except Exception as e:
            return MedicalDiagnosisResult(
                diagnosis_type=diagnosis_type,
                explanation=f"Dosya okunamadı: {file_path}",
                errors=[{"stage": "file_read", "error": str(e)}],
            )
        
        if diagnosis_type == "dermatology":
            return self.analyze_dermatology(image_bytes, format, options)
        elif diagnosis_type == "chest_xray":
            return self.analyze_chest_xray(image_bytes, format, options)
        else:
            return MedicalDiagnosisResult(
                diagnosis_type=diagnosis_type,
                explanation=f"Bilinmeyen teşhis türü: {diagnosis_type}",
                errors=[{"stage": "routing", "error": f"Unknown diagnosis type: {diagnosis_type}"}],
            )
