"""
OCR Factory

Factory for creating OCR extractor instances.
"""

from typing import Optional, Dict, Any
from enum import Enum

from ...domain.ports.text_extractor import TextExtractorPort
from .paddle_ocr import PaddleOCRExtractor, DummyOCRExtractor
from .tesseract_ocr import TesseractOCRExtractor


class OCRType(Enum):
    """Available OCR implementations."""
    
    PADDLE = "paddle"
    TESSERACT = "tesseract"
    DUMMY = "dummy"


class OCRFactory:
    """
    Factory for creating OCR extractor instances.
    
    Usage:
        # Create PaddleOCR extractor
        ocr = OCRFactory.create(OCRType.PADDLE, lang="tr")
        
        # Create Tesseract fallback
        ocr = OCRFactory.create(OCRType.TESSERACT, lang="tur+eng")
    """
    
    @staticmethod
    def create(
        ocr_type: OCRType,
        **kwargs
    ) -> TextExtractorPort:
        """
        Create an OCR extractor instance.
        
        Args:
            ocr_type: Type of OCR to create
            **kwargs: Configuration options
                For PADDLE:
                - lang: Language code (default: "tr")
                - use_angle_cls: Use angle classification
                - use_gpu: Use GPU
                For TESSERACT:
                - lang: Tesseract language code (default: "tur+eng")
                - psm: Page segmentation mode
                
        Returns:
            TextExtractorPort implementation
        """
        if ocr_type == OCRType.PADDLE:
            return PaddleOCRExtractor(
                lang=kwargs.get("lang", "tr"),
                use_angle_cls=kwargs.get("use_angle_cls", True),
                use_gpu=kwargs.get("use_gpu", False),
                show_log=kwargs.get("show_log", False)
            )
        
        elif ocr_type == OCRType.TESSERACT:
            return TesseractOCRExtractor(
                lang=kwargs.get("lang", "tur+eng"),
                config=kwargs.get("config", ""),
                oem=kwargs.get("oem", 3),
                psm=kwargs.get("psm", 3)
            )
        
        elif ocr_type == OCRType.DUMMY:
            return DummyOCRExtractor(
                preset_text=kwargs.get("preset_text", "SAMPLE DRUG 500mg")
            )
        
        else:
            raise ValueError(f"Unknown OCR type: {ocr_type}")
    
    @staticmethod
    def create_with_fallback(
        primary_type: OCRType = OCRType.PADDLE,
        fallback_type: OCRType = OCRType.TESSERACT,
        **kwargs
    ) -> TextExtractorPort:
        """
        Create OCR with automatic fallback.
        
        Tries to create primary OCR, falls back if initialization fails.
        
        Args:
            primary_type: Primary OCR type to try
            fallback_type: Fallback OCR if primary fails
            **kwargs: Configuration options
            
        Returns:
            TextExtractorPort implementation
        """
        try:
            return OCRFactory.create(primary_type, **kwargs)
        except Exception as e:
            import logging
            logging.warning(f"Primary OCR ({primary_type}) failed: {e}, using fallback")
            return OCRFactory.create(fallback_type, **kwargs)
    
    @staticmethod
    def create_from_config(config: Dict[str, Any]) -> TextExtractorPort:
        """
        Create OCR from configuration dictionary.
        
        Args:
            config: Configuration with 'type' and other options
            
        Returns:
            TextExtractorPort implementation
        """
        ocr_type = OCRType(config.get("type", "paddle"))
        return OCRFactory.create(ocr_type, **config)
