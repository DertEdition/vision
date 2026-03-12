"""
Image Classifier Port

Abstract interface for medical image classification implementations.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List

from ..value_objects.image_data import ImageData


class ImageClassifierPort(ABC):
    """
    Port (interface) for medical image classification implementations.
    
    Responsible for classifying medical images to:
    - Detect diseases or conditions
    - Provide confidence scores for predictions
    - Return class labels and probabilities
    
    Implementations may use:
    - ResNet for dermatology classification
    - DenseNet for chest X-ray classification
    - Other CNN architectures
    """
    
    @abstractmethod
    def classify(
        self,
        image: ImageData,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Classify a medical image.
        
        Args:
            image: Image data to classify
            options: Optional classification configuration
                - confidence_threshold: Minimum prediction confidence
                - top_k: Number of top predictions to return
                
        Returns:
            Dictionary containing classification results:
                - predictions: List of (label, probability) tuples
                - top_prediction: Most likely class label
                - confidence: Confidence of top prediction
                - all_probabilities: Dict of all class probabilities
                
        Raises:
            ClassificationError: If classification fails
        """
        pass
    
    @abstractmethod
    def get_class_labels(self) -> List[str]:
        """
        Get the list of class labels this classifier recognizes.
        
        Returns:
            List of string class labels
        """
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the name/identifier of the underlying model."""
        pass
    
    @property
    def supported_formats(self) -> list:
        """Get list of supported image formats."""
        return ["jpeg", "jpg", "png", "bmp", "webp"]
