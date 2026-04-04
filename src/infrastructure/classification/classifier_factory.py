"""
Classifier Factory

Factory for creating medical image classifiers.
"""

from enum import Enum
from typing import Optional

from ...domain.ports.image_classifier import ImageClassifierPort


class ClassifierType(Enum):
    """Available classifier types."""
    DERMATOLOGY = "dermatology"
    CHEST_XRAY = "chest_xray"


class ClassifierFactory:
    """
    Factory for creating medical image classifier instances.
    
    Follows the same factory pattern as VisionAnalyzerFactory, OCRFactory, etc.
    
    Usage:
        classifier = ClassifierFactory.create(
            ClassifierType.DERMATOLOGY,
            model_path="models/dermatology_model.pth",
            device="cuda"
        )
    """
    
    @staticmethod
    def create(
        classifier_type: ClassifierType,
        model_path: Optional[str] = None,
        device: str = "cuda",
        confidence_threshold: float = 0.5,
        abnormality_threshold: Optional[float] = None,
        image_size: int = 224,
        **kwargs
    ) -> ImageClassifierPort:
        """
        Create a classifier instance.
        
        Args:
            classifier_type: Type of classifier to create
            model_path: Path to trained model weights
            device: Computation device (cuda/cpu)
            confidence_threshold: Minimum confidence for predictions
            abnormality_threshold: Optional higher threshold for abnormality decision
            image_size: Input image size
            
        Returns:
            Configured ImageClassifierPort implementation
            
        Raises:
            ValueError: If classifier type is not supported
        """
        if classifier_type == ClassifierType.DERMATOLOGY:
            from .dermatology_classifier import DermatologyClassifier
            return DermatologyClassifier(
                model_path=model_path or "models/dermatology_model.pth",
                device=device,
                confidence_threshold=confidence_threshold,
                image_size=image_size,
                **kwargs,
            )
        
        elif classifier_type == ClassifierType.CHEST_XRAY:
            from .chest_xray_classifier import ChestXrayClassifier
            return ChestXrayClassifier(
                model_path=model_path or "models/chest_xray_model.pth",
                device=device,
                confidence_threshold=confidence_threshold,
                abnormality_threshold=abnormality_threshold,
                image_size=image_size,
                **kwargs,
            )
        
        else:
            raise ValueError(f"Unsupported classifier type: {classifier_type}")
