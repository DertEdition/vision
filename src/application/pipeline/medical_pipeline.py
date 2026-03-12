"""
Medical Pipeline Orchestrator

Pipeline orchestrators and stage executors for medical image diagnosis.
Supports dermatology and chest X-ray analysis pipelines.
"""

import logging
import time
from typing import Optional, Dict, Any, List
from datetime import datetime

from .medical_context import MedicalPipelineContext
from ...domain.ports.image_classifier import ImageClassifierPort
from ...domain.value_objects.image_data import ImageData
from ...domain.value_objects.confidence_score import ConfidenceScore
from ...domain.entities.medical_diagnosis import (
    DermatologyDiagnosis,
    ChestXrayDiagnosis,
    MedicalDiagnosisResult,
    MEDICAL_DIAGNOSIS_DISCLAIMER,
)
from ...domain.entities.pipeline_result import PipelineStage, StageStatus

logger = logging.getLogger(__name__)


# ============================================================
# Pipeline Stage Executors
# ============================================================


class ImageClassificationStage:
    """
    Pipeline stage that runs CNN model inference on input medical image.
    
    This stage delegates to an ImageClassifierPort implementation
    (DermatologyClassifier or ChestXrayClassifier).
    """
    
    stage = PipelineStage.IMAGE_CLASSIFICATION
    
    def __init__(self, classifier: ImageClassifierPort):
        self._classifier = classifier
    
    def execute(self, context: MedicalPipelineContext) -> None:
        """
        Execute image classification.
        
        Args:
            context: Medical pipeline context with input image
        """
        logger.info(f"Running image classification ({self._classifier.model_name})")
        start = time.time()
        
        try:
            result = self._classifier.classify(
                context.image,
                options=context.options
            )
            context.classification_result = result
            
            # Build structured diagnosis from raw results
            if context.diagnosis_type == "dermatology":
                self._build_dermatology_diagnosis(context, result)
            elif context.diagnosis_type == "chest_xray":
                self._build_chest_xray_diagnosis(context, result)
            
            duration = (time.time() - start) * 1000
            context.record_stage_timing("image_classification", duration)
            logger.info(
                f"Classification complete in {duration:.0f}ms "
                f"(model: {self._classifier.model_name})"
            )
        except Exception as e:
            duration = (time.time() - start) * 1000
            context.record_stage_timing("image_classification", duration)
            context.add_error("image_classification", str(e))
            logger.error(f"Classification failed: {e}", exc_info=True)
            raise
    
    def _build_dermatology_diagnosis(
        self, context: MedicalPipelineContext, result: Dict[str, Any]
    ) -> None:
        """Build DermatologyDiagnosis from raw classification result."""
        malignancy_data = result.get("malignancy", {})
        disease_data = result.get("disease_type", {})
        
        recommendations = []
        mal_prediction = malignancy_data.get("prediction", "unknown")
        if mal_prediction == "malignant":
            recommendations.extend([
                "Bu lezyon potansiyel olarak kötü huylu olarak sınıflandırılmıştır.",
                "Lütfen en kısa sürede bir dermatolog ile görüşün.",
                "Biyopsi önerilmektedir.",
            ])
        elif mal_prediction == "indeterminate":
            recommendations.extend([
                "Bu lezyon belirsiz olarak sınıflandırılmıştır.",
                "Takip muayenesi önerilmektedir.",
            ])
        else:
            recommendations.append(
                "Bu lezyon iyi huylu olarak sınıflandırılmıştır. "
                "Düzenli cilt kontrolleri önerilir."
            )
        
        context.dermatology_diagnosis = DermatologyDiagnosis(
            malignancy=mal_prediction,
            malignancy_confidence=malignancy_data.get("confidence", 0.0),
            disease_type=disease_data.get("prediction", "unknown"),
            disease_type_confidence=disease_data.get("confidence", 0.0),
            recommendations=recommendations,
        )
    
    def _build_chest_xray_diagnosis(
        self, context: MedicalPipelineContext, result: Dict[str, Any]
    ) -> None:
        """Build ChestXrayDiagnosis from raw classification result."""
        findings = result.get("findings", [])
        has_abnormality = result.get("has_abnormality", False)
        
        recommendations = []
        if has_abnormality:
            recommendations.extend([
                "Göğüs röntgeninizde anormal bulgular tespit edilmiştir.",
                "Lütfen bir göğüs hastalıkları uzmanına başvurun.",
                f"Tespit edilen bulgular: {', '.join(findings)}",
            ])
        else:
            recommendations.append(
                "Göğüs röntgeninde belirgin bir anormallik tespit edilmemiştir. "
                "Şikayetleriniz devam ederse doktorunuza başvurun."
            )
        
        context.chest_xray_diagnosis = ChestXrayDiagnosis(
            findings=findings,
            finding_probabilities=result.get("finding_probabilities", {}),
            has_abnormality=has_abnormality,
            primary_finding=result.get("primary_finding", "No Finding"),
            recommendations=recommendations,
        )


class MedicalResponseStage:
    """
    Pipeline stage that generates a user-friendly medical report
    from classification results.
    """
    
    stage = PipelineStage.MEDICAL_RESPONSE
    
    def execute(self, context: MedicalPipelineContext) -> None:
        """
        Generate a human-readable explanation from diagnosis results.
        
        Args:
            context: Medical pipeline context with classification results
        """
        logger.info("Generating medical response")
        start = time.time()
        
        try:
            if context.diagnosis_type == "dermatology":
                context.generated_response = self._generate_dermatology_response(
                    context.dermatology_diagnosis
                )
            elif context.diagnosis_type == "chest_xray":
                context.generated_response = self._generate_chest_xray_response(
                    context.chest_xray_diagnosis
                )
            else:
                context.generated_response = "Bilinmeyen teşhis türü."
            
            duration = (time.time() - start) * 1000
            context.record_stage_timing("medical_response", duration)
            logger.info(f"Response generated in {duration:.0f}ms")
        except Exception as e:
            duration = (time.time() - start) * 1000
            context.record_stage_timing("medical_response", duration)
            context.add_error("medical_response", str(e))
            logger.error(f"Response generation failed: {e}", exc_info=True)
    
    def _generate_dermatology_response(
        self, diagnosis: Optional[DermatologyDiagnosis]
    ) -> str:
        """Generate human-readable dermatology report."""
        if not diagnosis:
            return "Dermatoloji analizi tamamlanamadı."
        
        lines = [
            "## Dermatoloji Analiz Sonucu\n",
            f"**Malignite Değerlendirmesi:** {diagnosis.malignancy.capitalize()}",
            f"**Güven Skoru:** {diagnosis.malignancy_confidence:.1%}",
            f"**Hastalık Tipi:** {diagnosis.disease_type}",
            f"**Hastalık Tipi Güven Skoru:** {diagnosis.disease_type_confidence:.1%}",
            "",
            "### Öneriler:",
        ]
        for rec in diagnosis.recommendations:
            lines.append(f"- {rec}")
        
        return "\n".join(lines)
    
    def _generate_chest_xray_response(
        self, diagnosis: Optional[ChestXrayDiagnosis]
    ) -> str:
        """Generate human-readable chest X-ray report."""
        if not diagnosis:
            return "Göğüs röntgeni analizi tamamlanamadı."
        
        lines = [
            "## Göğüs Röntgeni Analiz Sonucu\n",
            f"**Anormallik Tespit Edildi:** {'Evet' if diagnosis.has_abnormality else 'Hayır'}",
            f"**Ana Bulgu:** {diagnosis.primary_finding}",
        ]
        
        if diagnosis.findings:
            lines.append("\n### Tespit Edilen Bulgular:")
            for finding in diagnosis.findings:
                prob = diagnosis.finding_probabilities.get(finding, 0)
                lines.append(f"- **{finding}:** {prob:.1%}")
        
        lines.append("\n### Öneriler:")
        for rec in diagnosis.recommendations:
            lines.append(f"- {rec}")
        
        return "\n".join(lines)


# ============================================================
# Pipeline Orchestrator
# ============================================================


class MedicalPipelineOrchestrator:
    """
    Orchestrator for medical image analysis pipelines.
    
    Executes a simplified two-stage pipeline:
    IMAGE_CLASSIFICATION → MEDICAL_RESPONSE
    
    Similar pattern to PipelineOrchestrator but for medical analysis.
    
    Usage:
        orchestrator = MedicalPipelineBuilder()\\
            .with_classifier(dermatology_classifier)\\
            .with_diagnosis_type("dermatology")\\
            .build()
        result = orchestrator.run(image_data)
    """
    
    def __init__(
        self,
        classifier: ImageClassifierPort,
        diagnosis_type: str,
        timeout_seconds: int = 120,
    ):
        """
        Initialize the medical pipeline orchestrator.
        
        Args:
            classifier: Image classification implementation
            diagnosis_type: Type of diagnosis (dermatology or chest_xray)
            timeout_seconds: Maximum execution time
        """
        self._classifier = classifier
        self._diagnosis_type = diagnosis_type
        self._timeout = timeout_seconds
        
        # Create pipeline stages
        self._classification_stage = ImageClassificationStage(classifier)
        self._response_stage = MedicalResponseStage()
    
    def run(
        self,
        image: ImageData,
        options: Optional[Dict[str, Any]] = None
    ) -> MedicalDiagnosisResult:
        """
        Execute the medical diagnosis pipeline.
        
        Args:
            image: Medical image to analyze
            options: Optional processing configuration
            
        Returns:
            MedicalDiagnosisResult with diagnosis and explanation
        """
        start_time = time.time()
        
        # Create context
        context = MedicalPipelineContext(
            image=image,
            diagnosis_type=self._diagnosis_type,
            options=options or {},
        )
        
        logger.info(
            f"Starting medical pipeline (type={self._diagnosis_type}, "
            f"request_id={context.request_id})"
        )
        
        # Stage 1: Classification
        try:
            self._classification_stage.execute(context)
        except Exception as e:
            logger.error(f"Classification stage failed: {e}")
            return self._create_error_result(
                context, str(e), start_time
            )
        
        # Stage 2: Response Generation
        try:
            self._response_stage.execute(context)
        except Exception as e:
            logger.warning(f"Response generation failed (non-critical): {e}")
        
        # Build final result
        total_time = (time.time() - start_time) * 1000
        
        result = MedicalDiagnosisResult(
            diagnosis_type=self._diagnosis_type,
            dermatology=context.dermatology_diagnosis,
            chest_xray=context.chest_xray_diagnosis,
            explanation=context.generated_response or "",
            warnings=self._generate_warnings(context),
            disclaimer=MEDICAL_DIAGNOSIS_DISCLAIMER,
            overall_confidence=self._calculate_confidence(context),
            request_id=context.request_id,
            total_processing_time_ms=total_time,
            errors=context.errors,
        )
        
        logger.info(
            f"Medical pipeline complete: {result} "
            f"(total: {total_time:.0f}ms)"
        )
        
        return result
    
    def _generate_warnings(self, context: MedicalPipelineContext) -> List[str]:
        """Generate safety warnings based on results."""
        warnings = [
            "Bu sonuçlar yapay zeka modeli tarafından üretilmiştir ve tıbbi teşhis yerine geçmez.",
        ]
        
        if context.diagnosis_type == "dermatology" and context.dermatology_diagnosis:
            if context.dermatology_diagnosis.is_malignant:
                warnings.append(
                    "⚠️ Malign (kötü huylu) bulgu tespit edilmiştir. "
                    "Acil tıbbi değerlendirme önerilir."
                )
        
        if context.diagnosis_type == "chest_xray" and context.chest_xray_diagnosis:
            if context.chest_xray_diagnosis.has_abnormality:
                warnings.append(
                    "⚠️ Göğüs röntgeninde anormal bulgular tespit edilmiştir. "
                    "Uzman değerlendirmesi önerilir."
                )
        
        return warnings
    
    def _calculate_confidence(self, context: MedicalPipelineContext) -> ConfidenceScore:
        """Calculate overall confidence from classification results."""
        confidence_value = 0.0
        
        if context.diagnosis_type == "dermatology" and context.dermatology_diagnosis:
            confidence_value = max(
                context.dermatology_diagnosis.malignancy_confidence,
                context.dermatology_diagnosis.disease_type_confidence,
            )
        elif context.diagnosis_type == "chest_xray" and context.classification_result:
            primary_conf = context.classification_result.get("primary_confidence", 0.0)
            confidence_value = primary_conf
        
        return ConfidenceScore(value=confidence_value)
    
    def _create_error_result(
        self,
        context: MedicalPipelineContext,
        error_message: str,
        start_time: float,
    ) -> MedicalDiagnosisResult:
        """Create an error result."""
        total_time = (time.time() - start_time) * 1000
        return MedicalDiagnosisResult(
            diagnosis_type=self._diagnosis_type,
            explanation=f"Analiz sırasında bir hata oluştu: {error_message}",
            warnings=["Analiz tamamlanamadı. Lütfen tekrar deneyin."],
            disclaimer=MEDICAL_DIAGNOSIS_DISCLAIMER,
            overall_confidence=ConfidenceScore.zero(),
            request_id=context.request_id,
            total_processing_time_ms=total_time,
            errors=context.errors,
        )


# ============================================================
# Pipeline Builder
# ============================================================


class MedicalPipelineBuilder:
    """
    Builder for constructing MedicalPipelineOrchestrator instances.
    
    Follows the same builder pattern as PipelineBuilder.
    
    Usage:
        pipeline = MedicalPipelineBuilder()\\
            .with_classifier(dermatology_classifier)\\
            .with_diagnosis_type("dermatology")\\
            .with_timeout(120)\\
            .build()
    """
    
    def __init__(self):
        self._classifier: Optional[ImageClassifierPort] = None
        self._diagnosis_type: str = ""
        self._timeout: int = 120
    
    def with_classifier(self, classifier: ImageClassifierPort) -> "MedicalPipelineBuilder":
        """Set the image classifier."""
        self._classifier = classifier
        return self
    
    def with_diagnosis_type(self, diagnosis_type: str) -> "MedicalPipelineBuilder":
        """Set the diagnosis type (dermatology or chest_xray)."""
        self._diagnosis_type = diagnosis_type
        return self
    
    def with_timeout(self, timeout_seconds: int) -> "MedicalPipelineBuilder":
        """Set pipeline timeout."""
        self._timeout = timeout_seconds
        return self
    
    def build(self) -> MedicalPipelineOrchestrator:
        """
        Build the medical pipeline orchestrator.
        
        Returns:
            Configured MedicalPipelineOrchestrator
            
        Raises:
            ValueError: If required components are missing
        """
        if self._classifier is None:
            raise ValueError("Classifier is required. Call with_classifier() first.")
        if not self._diagnosis_type:
            raise ValueError("Diagnosis type is required. Call with_diagnosis_type() first.")
        if self._diagnosis_type not in ("dermatology", "chest_xray"):
            raise ValueError(
                f"Invalid diagnosis type: {self._diagnosis_type}. "
                "Must be 'dermatology' or 'chest_xray'."
            )
        
        return MedicalPipelineOrchestrator(
            classifier=self._classifier,
            diagnosis_type=self._diagnosis_type,
            timeout_seconds=self._timeout,
        )
