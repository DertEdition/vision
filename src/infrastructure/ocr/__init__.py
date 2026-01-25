"""
OCR (Text Extraction) Adapters

Implementations of TextExtractorPort using various OCR engines.
"""

from .paddle_ocr import PaddleOCRExtractor
from .tesseract_ocr import TesseractOCRExtractor
from .factory import OCRFactory, OCRType

__all__ = [
    "PaddleOCRExtractor",
    "TesseractOCRExtractor",
    "OCRFactory",
    "OCRType",
]
