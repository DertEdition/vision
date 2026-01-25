"""
Vision Analyzer Port

Abstract interface for vision analysis implementations.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

from ..value_objects.image_data import ImageData
from ..entities.extraction_result import VisionAnalysisResult


class VisionAnalyzerPort(ABC):
    """
    Port (interface) for vision analysis implementations.
    
    Responsible for analyzing pharmaceutical images to:
    - Detect drug packaging (boxes, blisters, leaflets)
    - Identify regions of interest for OCR
    - Assess image quality
    
    Implementations may use:
    - YOLOv8 for object detection
    - Vision Transformers for classification
    - Custom pharmaceutical detection models
    """
    
    @abstractmethod
    def analyze(
        self,
        image: ImageData,
        options: Optional[Dict[str, Any]] = None
    ) -> VisionAnalysisResult:
        """
        Analyze an image for pharmaceutical content.
        
        Args:
            image: Image data to analyze
            options: Optional analysis configuration
                - confidence_threshold: Minimum detection confidence
                - max_detections: Maximum number of detections
                - detect_text_regions: Whether to detect text areas
                
        Returns:
            VisionAnalysisResult containing detected objects and regions
            
        Raises:
            VisionAnalysisError: If analysis fails
        """
        pass
    
    @abstractmethod
    def is_pharmaceutical_image(self, image: ImageData) -> bool:
        """
        Quick check if image contains pharmaceutical content.
        
        Args:
            image: Image data to check
            
        Returns:
            True if image likely contains pharmaceutical packaging
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
