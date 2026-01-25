"""
Text Extractor Port

Abstract interface for OCR/text extraction implementations.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

from ..value_objects.image_data import ImageData
from ..value_objects.bounding_box import BoundingBox
from ..entities.extraction_result import TextExtractionResult


class TextExtractorPort(ABC):
    """
    Port (interface) for OCR/text extraction implementations.
    
    Responsible for extracting text from pharmaceutical images:
    - Drug names from packaging
    - Active ingredients lists
    - Dosage information
    - Leaflet/prospectus content
    
    Implementations may use:
    - PaddleOCR for general OCR with Turkish support
    - Tesseract as fallback
    - Cloud OCR services (Google Vision, Azure)
    - Document AI for layout-aware extraction
    """
    
    @abstractmethod
    def extract(
        self,
        image: ImageData,
        regions: Optional[List[BoundingBox]] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> TextExtractionResult:
        """
        Extract text from an image.
        
        Args:
            image: Image data to process
            regions: Optional list of regions to focus OCR on.
                     If None, processes entire image.
            options: Optional extraction configuration
                - language: Primary language hint (e.g., "tr" for Turkish)
                - detect_layout: Whether to preserve layout structure
                - confidence_threshold: Minimum OCR confidence
                
        Returns:
            TextExtractionResult containing extracted text blocks
            
        Raises:
            TextExtractionError: If extraction fails
        """
        pass
    
    @abstractmethod
    def extract_from_region(
        self,
        image: ImageData,
        region: BoundingBox
    ) -> str:
        """
        Extract text from a specific region of the image.
        
        Args:
            image: Image data to process
            region: Bounding box defining the region
            
        Returns:
            Extracted text string
        """
        pass
    
    @property
    @abstractmethod
    def supported_languages(self) -> List[str]:
        """Get list of supported language codes."""
        pass
    
    @property
    @abstractmethod
    def engine_name(self) -> str:
        """Get the name of the OCR engine."""
        pass
    
    def supports_language(self, language_code: str) -> bool:
        """Check if a language is supported."""
        return language_code.lower() in [l.lower() for l in self.supported_languages]
