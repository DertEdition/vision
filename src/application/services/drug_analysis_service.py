"""
Drug Analysis Service

High-level application service for drug image analysis.
"""

from typing import Optional, Dict, Any
from pathlib import Path
import logging

from ..pipeline.orchestrator import PipelineOrchestrator
from ...domain.value_objects.image_data import ImageData
from ...domain.entities.pipeline_result import PipelineResult
from ...domain.exceptions import InvalidImageError, ValidationError


logger = logging.getLogger(__name__)


class DrugAnalysisService:
    """
    Application service for analyzing drug images.
    
    This is the main entry point for external consumers.
    It provides a simplified interface to the pipeline and handles:
    - Input validation
    - Image loading from various sources
    - Result formatting
    - Error handling
    
    Usage:
        service = DrugAnalysisService(pipeline)
        
        # From file path
        result = service.analyze_from_file("path/to/drug_image.jpg")
        
        # From bytes
        result = service.analyze_from_bytes(image_bytes)
        
        # From base64
        result = service.analyze_from_base64(base64_string)
    """
    
    def __init__(self, pipeline: PipelineOrchestrator):
        """
        Initialize the service.
        
        Args:
            pipeline: Configured pipeline orchestrator
        """
        self.pipeline = pipeline
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Validate pipeline
        pipeline.validate_configuration()
    
    def analyze(
        self,
        image: ImageData,
        options: Optional[Dict[str, Any]] = None
    ) -> PipelineResult:
        """
        Analyze a drug image.
        
        Args:
            image: Image data to analyze
            options: Optional analysis options
            
        Returns:
            PipelineResult with analysis results
        """
        self.logger.info(f"Starting drug image analysis from {image.source or 'bytes'}")
        
        # Run pipeline
        result = self.pipeline.run(image, options)
        
        # Log result summary
        if result.is_successful:
            self.logger.info(f"Analysis successful: {result.drug_info}")
        else:
            self.logger.warning(f"Analysis failed or incomplete: {len(result.errors)} errors")
        
        return result
    
    def analyze_from_file(
        self,
        file_path: str,
        options: Optional[Dict[str, Any]] = None
    ) -> PipelineResult:
        """
        Analyze a drug image from a file path.
        
        Args:
            file_path: Path to the image file
            options: Optional analysis options
            
        Returns:
            PipelineResult with analysis results
            
        Raises:
            InvalidImageError: If file doesn't exist or is invalid
        """
        path = Path(file_path)
        
        # Validate file exists
        if not path.exists():
            raise InvalidImageError(f"Image file not found: {file_path}")
        
        # Validate extension
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}
        if path.suffix.lower() not in valid_extensions:
            raise InvalidImageError(
                f"Unsupported image format: {path.suffix}. "
                f"Supported: {', '.join(valid_extensions)}"
            )
        
        # Load image
        try:
            image = ImageData.from_file(str(path))
        except Exception as e:
            raise InvalidImageError(f"Failed to load image: {e}")
        
        return self.analyze(image, options)
    
    def analyze_from_bytes(
        self,
        image_bytes: bytes,
        format: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> PipelineResult:
        """
        Analyze a drug image from raw bytes.
        
        Args:
            image_bytes: Raw image bytes
            format: Image format (jpeg, png, etc.)
            options: Optional analysis options
            
        Returns:
            PipelineResult with analysis results
        """
        if not image_bytes:
            raise InvalidImageError("Image bytes cannot be empty")
        
        image = ImageData.from_bytes(image_bytes, format=format)
        return self.analyze(image, options)
    
    def analyze_from_base64(
        self,
        base64_string: str,
        format: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> PipelineResult:
        """
        Analyze a drug image from base64 string.
        
        Args:
            base64_string: Base64 encoded image
            format: Image format (jpeg, png, etc.)
            options: Optional analysis options
            
        Returns:
            PipelineResult with analysis results
        """
        if not base64_string:
            raise InvalidImageError("Base64 string cannot be empty")
        
        image = ImageData.from_base64(base64_string, format=format)
        return self.analyze(image, options)
    
    def get_user_response(self, result: PipelineResult) -> Dict[str, Any]:
        """
        Get a formatted response suitable for end users.
        
        Args:
            result: Pipeline result
            
        Returns:
            Dictionary with user-friendly response
        """
        return result.get_user_response()
    
    def get_debug_info(self, result: PipelineResult) -> Dict[str, Any]:
        """
        Get detailed debug information about pipeline execution.
        
        Args:
            result: Pipeline result
            
        Returns:
            Dictionary with debug information
        """
        return result.get_debug_info()
