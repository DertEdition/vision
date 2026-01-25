"""
Vision Analyzer Factory

Factory for creating vision analyzer instances.
"""

from typing import Optional, Dict, Any
from enum import Enum

from ...domain.ports.vision_analyzer import VisionAnalyzerPort
from .yolo_analyzer import YOLOVisionAnalyzer, DummyVisionAnalyzer


class VisionAnalyzerType(Enum):
    """Available vision analyzer implementations."""
    
    YOLO = "yolo"
    YOLO_CUSTOM = "yolo_custom"
    DUMMY = "dummy"


class VisionAnalyzerFactory:
    """
    Factory for creating vision analyzer instances.
    
    Usage:
        # Create default YOLO analyzer
        analyzer = VisionAnalyzerFactory.create(VisionAnalyzerType.YOLO)
        
        # Create custom YOLO analyzer
        analyzer = VisionAnalyzerFactory.create(
            VisionAnalyzerType.YOLO_CUSTOM,
            model_path="path/to/weights.pt"
        )
    """
    
    @staticmethod
    def create(
        analyzer_type: VisionAnalyzerType,
        **kwargs
    ) -> VisionAnalyzerPort:
        """
        Create a vision analyzer instance.
        
        Args:
            analyzer_type: Type of analyzer to create
            **kwargs: Additional configuration options
                For YOLO:
                - model_path: Path to model weights
                - confidence_threshold: Detection threshold
                - device: Inference device ('cpu', 'cuda')
                
        Returns:
            VisionAnalyzerPort implementation
        """
        if analyzer_type == VisionAnalyzerType.YOLO:
            return YOLOVisionAnalyzer(
                use_pretrained=True,
                confidence_threshold=kwargs.get("confidence_threshold", 0.25),
                device=kwargs.get("device", "cpu")
            )
        
        elif analyzer_type == VisionAnalyzerType.YOLO_CUSTOM:
            model_path = kwargs.get("model_path")
            if not model_path:
                raise ValueError("model_path required for YOLO_CUSTOM analyzer")
            
            return YOLOVisionAnalyzer(
                model_path=model_path,
                confidence_threshold=kwargs.get("confidence_threshold", 0.25),
                device=kwargs.get("device", "cpu"),
                use_pretrained=False
            )
        
        elif analyzer_type == VisionAnalyzerType.DUMMY:
            return DummyVisionAnalyzer()
        
        else:
            raise ValueError(f"Unknown analyzer type: {analyzer_type}")
    
    @staticmethod
    def create_from_config(config: Dict[str, Any]) -> VisionAnalyzerPort:
        """
        Create analyzer from configuration dictionary.
        
        Args:
            config: Configuration dictionary with 'type' and other options
            
        Returns:
            VisionAnalyzerPort implementation
        """
        analyzer_type = VisionAnalyzerType(config.get("type", "yolo"))
        return VisionAnalyzerFactory.create(analyzer_type, **config)
