"""
Tesseract OCR Text Extractor

OCR fallback implementation using Tesseract.
"""

from typing import Optional, Dict, Any, List
import logging
import time
from io import BytesIO

from ...domain.ports.text_extractor import TextExtractorPort
from ...domain.value_objects.image_data import ImageData
from ...domain.value_objects.bounding_box import BoundingBox
from ...domain.value_objects.confidence_score import ConfidenceScore
from ...domain.entities.extraction_result import TextExtractionResult, TextBlock
from ...domain.exceptions import (
    TextExtractionError,
    OCREngineError,
)


logger = logging.getLogger(__name__)


class TesseractOCRExtractor(TextExtractorPort):
    """
    Text extractor implementation using Tesseract OCR.
    
    Serves as a fallback when PaddleOCR is unavailable.
    Requires tesseract to be installed on the system.
    
    Attributes:
        lang: Tesseract language code(s)
        config: Additional tesseract configuration
    """
    
    # Tesseract language codes
    SUPPORTED_LANGUAGES = [
        "tur",  # Turkish
        "eng",  # English
        "deu",  # German
        "fra",  # French
    ]
    
    # Language code mapping (our codes to tesseract codes)
    LANG_MAPPING = {
        "tr": "tur",
        "en": "eng",
        "de": "deu",
        "fr": "fra",
    }
    
    def __init__(
        self,
        lang: str = "tur+eng",
        config: str = "",
        oem: int = 3,
        psm: int = 3
    ):
        """
        Initialize Tesseract OCR extractor.
        
        Args:
            lang: Tesseract language(s), e.g., "tur+eng"
            config: Additional tesseract configuration
            oem: OCR Engine Mode (3 = default)
            psm: Page Segmentation Mode (3 = auto)
        """
        self._lang = lang
        self._config = config
        self._oem = oem
        self._psm = psm
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Configure Tesseract path for Windows if not in PATH
        self._configure_tesseract_path()
    
    def _configure_tesseract_path(self):
        """Configure Tesseract executable path for Windows."""
        import sys
        import os
        from pathlib import Path
        
        # Only needed on Windows
        if sys.platform != 'win32':
            return
        
        try:
            import pytesseract
            
            # Common Tesseract installation paths on Windows
            possible_paths = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            ]
            
            # Check if tesseract is already accessible
            try:
                import subprocess
                subprocess.run(["tesseract", "--version"], 
                             capture_output=True, check=True, timeout=2)
                # Tesseract is in PATH, no need to configure
                return
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                pass
            
            # Try to find Tesseract in common locations
            for path in possible_paths:
                if Path(path).exists():
                    pytesseract.pytesseract.tesseract_cmd = path
                    self.logger.info(f"Configured Tesseract path: {path}")
                    return
            
            self.logger.warning(
                "Tesseract not found in PATH or common installation locations. "
                "Please ensure Tesseract is installed."
            )
        except ImportError:
            pass
    
    def _get_tesseract_config(self) -> str:
        """Build tesseract configuration string."""
        config_parts = [
            f"--oem {self._oem}",
            f"--psm {self._psm}",
        ]
        if self._config:
            config_parts.append(self._config)
        return " ".join(config_parts)
    
    def _load_image(self, image: ImageData):
        """Load and preprocess image for OCR using OpenCV."""
        try:
            from ..utils.image_processing import (
                bytes_to_cv2, 
                resize_image, 
                enhance_for_ocr
            )
            
            # Decode image bytes to OpenCV format
            img = bytes_to_cv2(image.bytes)
            
            # Resize if needed
            img, _ = resize_image(img, max_dimension=1800)
            
            # Enhance for OCR (CLAHE + bilateral filter + sharpening)
            enhanced = enhance_for_ocr(img)
            
            return enhanced
            
        except Exception as e:
            raise TextExtractionError(f"Failed to load image for OCR: {e}")
    
    def extract(
        self,
        image: ImageData,
        regions: Optional[List[BoundingBox]] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> TextExtractionResult:
        """
        Extract text from an image using Tesseract.
        
        Args:
            image: Image data to process
            regions: Optional regions (currently uses full image)
            options: Optional configuration
                - lang: Override language
                
        Returns:
            TextExtractionResult with extracted text
        """
        start_time = time.time()
        options = options or {}
        
        try:
            import pytesseract
        except ImportError:
            raise OCREngineError(
                "pytesseract not installed. Install with: pip install pytesseract",
                engine_name="Tesseract"
            )
        
        # Load image (returns OpenCV numpy array)
        cv_image = self._load_image(image)
        img_height, img_width = cv_image.shape[:2]  # OpenCV: (height, width)
        
        # Get language
        lang = options.get("lang", self._lang)
        
        # Run OCR with detailed output
        try:
            data = pytesseract.image_to_data(
                cv_image,
                lang=lang,
                config=self._get_tesseract_config(),
                output_type=pytesseract.Output.DICT
            )
        except Exception as e:
            raise OCREngineError(f"Tesseract OCR failed: {e}", engine_name="Tesseract")
        
        # Parse results
        text_blocks: List[TextBlock] = []
        
        n_boxes = len(data['text'])
        for i in range(n_boxes):
            text = data['text'][i].strip()
            conf = data['conf'][i]
            
            if text and conf > 0:  # Filter empty and low confidence
                # Get bounding box
                x = data['left'][i]
                y = data['top'][i]
                w = data['width'][i]
                h = data['height'][i]
                
                bbox = BoundingBox(
                    x_min=x / img_width,
                    y_min=y / img_height,
                    x_max=(x + w) / img_width,
                    y_max=(y + h) / img_height,
                    is_normalized=True
                )
                
                text_blocks.append(TextBlock(
                    text=text,
                    bounding_box=bbox,
                    confidence=ConfidenceScore(value=conf / 100.0, source="tesseract"),
                    language=lang.split("+")[0]  # Primary language
                ))
        
        # Get full text
        try:
            full_text = pytesseract.image_to_string(
                pil_image,
                lang=lang,
                config=self._get_tesseract_config()
            )
        except Exception:
            full_text = " ".join(block.text for block in text_blocks)
        
        processing_time = (time.time() - start_time) * 1000
        
        return TextExtractionResult(
            text_blocks=text_blocks,
            full_text=full_text.strip(),
            primary_language=lang.split("+")[0],
            processing_time_ms=processing_time,
            raw_output={"engine": "Tesseract", "num_blocks": len(text_blocks)}
        )
    
    def extract_from_region(
        self,
        image: ImageData,
        region: BoundingBox
    ) -> str:
        """
        Extract text from a specific region using Tesseract.
        """
        try:
            from PIL import Image as PILImage
            import pytesseract
            
            # Load and crop image
            pil_image = self._load_image(image)
            img_width, img_height = pil_image.size
            
            # Convert normalized bbox to pixels
            abs_bbox = region.to_absolute(img_width, img_height)
            
            # Crop
            cropped = pil_image.crop((
                int(abs_bbox.x_min),
                int(abs_bbox.y_min),
                int(abs_bbox.x_max),
                int(abs_bbox.y_max)
            ))
            
            # OCR
            text = pytesseract.image_to_string(
                cropped,
                lang=self._lang,
                config=self._get_tesseract_config()
            )
            
            return text.strip()
            
        except Exception as e:
            self.logger.warning(f"Failed to extract from region: {e}")
            # Fall back to full image extraction
            result = self.extract(image)
            return result.full_text
    
    @property
    def supported_languages(self) -> List[str]:
        return list(self.LANG_MAPPING.keys())
    
    @property
    def engine_name(self) -> str:
        return "Tesseract"
