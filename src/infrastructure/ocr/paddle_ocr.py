"""
PaddleOCR Text Extractor

OCR implementation using PaddleOCR with Turkish language support.
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
    NoTextFoundError,
)


logger = logging.getLogger(__name__)


class PaddleOCRExtractor(TextExtractorPort):
    """
    Text extractor implementation using PaddleOCR.
    
    PaddleOCR provides excellent multilingual support including Turkish,
    and offers layout-aware text extraction.
    
    Attributes:
        lang: Language code for OCR
        use_angle_cls: Whether to use angle classification
        show_log: Whether to show PaddleOCR logs
    """
    
    # Supported languages in PaddleOCR
    SUPPORTED_LANGUAGES = [
        "tr",  # Turkish
        "en",  # English
        "latin",  # Latin languages
        "ch",  # Chinese
        "ar",  # Arabic
    ]
    
    def __init__(
        self,
        lang: str = "tr",
        use_angle_cls: bool = True,
        use_gpu: bool = False,
        show_log: bool = False
    ):
        """
        Initialize PaddleOCR extractor.
        
        Args:
            lang: OCR language (default: Turkish)
            use_angle_cls: Use angle classification for rotated text
            use_gpu: Use GPU acceleration
            show_log: Show PaddleOCR logs
        """
        self._lang = lang
        self._use_angle_cls = use_angle_cls
        self._use_gpu = use_gpu
        self._show_log = show_log
        self._ocr = None
        self._initialized = False
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def _initialize(self) -> None:
        """Lazy initialization of PaddleOCR."""
        if self._initialized:
            return
        
        try:
            from paddleocr import PaddleOCR
            
            self.logger.info(f"Initializing PaddleOCR with lang={self._lang}")
            
            self._ocr = PaddleOCR(
                lang=self._lang,
                use_angle_cls=self._use_angle_cls,
                use_gpu=self._use_gpu,
                show_log=self._show_log
            )
            
            self._initialized = True
            self.logger.info("PaddleOCR initialized successfully")
            
        except ImportError:
            raise OCREngineError(
                "paddleocr not installed. Install with: pip install paddleocr",
                engine_name="PaddleOCR"
            )
        except Exception as e:
            raise OCREngineError(
                f"Failed to initialize PaddleOCR: {e}",
                engine_name="PaddleOCR"
            )
    
    def _load_image(self, image: ImageData):
        """Convert ImageData to numpy array for PaddleOCR."""
        try:
            from PIL import Image as PILImage
            import numpy as np
            
            image_bytes = image.bytes
            pil_image = PILImage.open(BytesIO(image_bytes))
            
            # Convert to RGB
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")
            
            return np.array(pil_image)
            
        except Exception as e:
            raise TextExtractionError(f"Failed to load image for OCR: {e}")
    
    def extract(
        self,
        image: ImageData,
        regions: Optional[List[BoundingBox]] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> TextExtractionResult:
        """
        Extract text from an image using PaddleOCR.
        
        Args:
            image: Image data to process
            regions: Optional regions to focus on (currently extracts from full image)
            options: Optional configuration
            
        Returns:
            TextExtractionResult with extracted text blocks
        """
        start_time = time.time()
        options = options or {}
        
        # Initialize OCR engine
        self._initialize()
        
        # Load image
        img_array = self._load_image(image)
        img_height, img_width = img_array.shape[:2]
        
        # Run OCR
        try:
            results = self._ocr.ocr(img_array, cls=self._use_angle_cls)
        except Exception as e:
            raise OCREngineError(f"PaddleOCR inference failed: {e}", engine_name="PaddleOCR")
        
        # Parse results
        text_blocks: List[TextBlock] = []
        
        if results and results[0]:
            for line in results[0]:
                bbox_points = line[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                text_info = line[1]    # (text, confidence)
                
                text = text_info[0]
                confidence = text_info[1]
                
                # Convert polygon to bounding box (normalized)
                x_coords = [p[0] for p in bbox_points]
                y_coords = [p[1] for p in bbox_points]
                
                bbox = BoundingBox(
                    x_min=min(x_coords) / img_width,
                    y_min=min(y_coords) / img_height,
                    x_max=max(x_coords) / img_width,
                    y_max=max(y_coords) / img_height,
                    is_normalized=True
                )
                
                text_blocks.append(TextBlock(
                    text=text,
                    bounding_box=bbox,
                    confidence=ConfidenceScore(value=confidence, source="paddleocr"),
                    language=self._lang
                ))
        
        # Build full text
        full_text = "\n".join(block.text for block in text_blocks)
        
        processing_time = (time.time() - start_time) * 1000
        
        return TextExtractionResult(
            text_blocks=text_blocks,
            full_text=full_text,
            primary_language=self._lang,
            processing_time_ms=processing_time,
            raw_output={"engine": "PaddleOCR", "num_blocks": len(text_blocks)}
        )
    
    def extract_from_region(
        self,
        image: ImageData,
        region: BoundingBox
    ) -> str:
        """
        Extract text from a specific region.
        
        Currently extracts from full image and filters by region.
        """
        result = self.extract(image)
        
        # Filter text blocks that fall within the region
        texts = []
        for block in result.text_blocks:
            if block.bounding_box:
                # Simple check: if block center is within region
                bcx, bcy = block.bounding_box.center
                if (region.x_min <= bcx <= region.x_max and
                    region.y_min <= bcy <= region.y_max):
                    texts.append(block.text)
        
        return "\n".join(texts) if texts else result.full_text
    
    @property
    def supported_languages(self) -> List[str]:
        return self.SUPPORTED_LANGUAGES.copy()
    
    @property
    def engine_name(self) -> str:
        return "PaddleOCR"


class DummyOCRExtractor(TextExtractorPort):
    """
    Dummy OCR extractor for testing without PaddleOCR.
    
    Returns preset text for testing purposes.
    """
    
    def __init__(self, preset_text: str = "SAMPLE DRUG 500mg Tablet"):
        self._preset_text = preset_text
    
    def extract(
        self,
        image: ImageData,
        regions: Optional[List[BoundingBox]] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> TextExtractionResult:
        return TextExtractionResult(
            text_blocks=[
                TextBlock(
                    text=self._preset_text,
                    confidence=ConfidenceScore(value=0.95, source="dummy")
                )
            ],
            full_text=self._preset_text,
            primary_language="tr",
            processing_time_ms=1.0
        )
    
    def extract_from_region(self, image: ImageData, region: BoundingBox) -> str:
        return self._preset_text
    
    @property
    def supported_languages(self) -> List[str]:
        return ["tr", "en"]
    
    @property
    def engine_name(self) -> str:
        return "DummyOCR"
