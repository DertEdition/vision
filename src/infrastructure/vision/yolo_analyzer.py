"""
YOLO Vision Analyzer

Vision analysis implementation using YOLOv8 for pharmaceutical package detection.
"""

from typing import Optional, Dict, Any, List
import logging
import time
from io import BytesIO

from ...domain.ports.vision_analyzer import VisionAnalyzerPort
from ...domain.value_objects.image_data import ImageData
from ...domain.value_objects.bounding_box import BoundingBox
from ...domain.value_objects.confidence_score import ConfidenceScore
from ...domain.entities.extraction_result import (
    VisionAnalysisResult,
    DetectedObject,
    DetectionClass,
)
from ...domain.exceptions import (
    VisionAnalysisError,
    ImageLoadError,
    ModelLoadError,
)


logger = logging.getLogger(__name__)


class YOLOVisionAnalyzer(VisionAnalyzerPort):
    """
    Vision analyzer implementation using YOLOv8.
    
    For pharmaceutical package detection, this implementation can either:
    1. Use a pre-trained YOLO model with general object detection
    2. Use a custom fine-tuned model for drug packages
    
    Attributes:
        model_path: Path to YOLO weights file
        confidence_threshold: Minimum detection confidence
        device: Device to run inference on ('cpu', 'cuda', 'mps')
    """
    
    # Mapping from YOLO class names to our detection classes
    CLASS_MAPPING = {
        # If using general YOLO model, map relevant classes
        "box": DetectionClass.DRUG_BOX,
        "package": DetectionClass.DRUG_BOX,
        "bottle": DetectionClass.DRUG_BOX,
        "label": DetectionClass.LABEL,
        "book": DetectionClass.LEAFLET,  # Leaflets sometimes detected as books
        "paper": DetectionClass.LEAFLET,
        # Custom pharmaceutical classes (if using fine-tuned model)
        "drug_box": DetectionClass.DRUG_BOX,
        "blister_pack": DetectionClass.BLISTER_PACK,
        "leaflet": DetectionClass.LEAFLET,
        "barcode": DetectionClass.BARCODE,
    }
    
    # Classes that indicate pharmaceutical content
    PHARMACEUTICAL_CLASSES = {
        DetectionClass.DRUG_BOX,
        DetectionClass.BLISTER_PACK,
        DetectionClass.LEAFLET,
        DetectionClass.LABEL,
    }
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.25,
        device: str = "cpu",
        use_pretrained: bool = True
    ):
        """
        Initialize the YOLO analyzer.
        
        Args:
            model_path: Path to custom YOLO weights (optional)
            confidence_threshold: Minimum detection confidence
            device: Inference device
            use_pretrained: Use pretrained YOLOv8 if no custom model
        """
        self._model_path = model_path
        self._confidence_threshold = confidence_threshold
        self._device = device
        self._use_pretrained = use_pretrained
        self._model = None
        self._model_loaded = False
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def _load_model(self) -> None:
        """Lazy load the YOLO model."""
        if self._model_loaded:
            return
        
        try:
            from ultralytics import YOLO
            
            if self._model_path:
                self.logger.info(f"Loading custom YOLO model from {self._model_path}")
                self._model = YOLO(self._model_path)
            elif self._use_pretrained:
                self.logger.info("Loading pretrained YOLOv8n model")
                self._model = YOLO("yolov8n.pt")
            else:
                raise ModelLoadError("No model path provided and pretrained disabled")
            
            self._model_loaded = True
            self.logger.info(f"YOLO model loaded successfully on {self._device}")
            
        except ImportError:
            raise ModelLoadError(
                "ultralytics package not installed. Install with: pip install ultralytics"
            )
        except Exception as e:
            raise ModelLoadError(f"Failed to load YOLO model: {e}")
    
    def _load_image(self, image: ImageData):
        """Load image for YOLO processing using OpenCV."""
        try:
            import cv2
            import numpy as np
            from ..utils.image_processing import bytes_to_cv2, cv2_to_rgb, resize_image
            
            # Decode image bytes to OpenCV format (BGR)
            img = bytes_to_cv2(image.bytes)
            
            # Resize if very large (YOLO works well with ~640-1280px)
            if max(img.shape[:2]) > 2000:
                img, _ = resize_image(img, max_dimension=2000)
            
            # Convert BGR to RGB (YOLO expects RGB)
            rgb_image = cv2_to_rgb(img)
            
            return rgb_image
            
        except Exception as e:
            raise ImageLoadError(f"Failed to load image: {e}")
    
    def analyze(
        self,
        image: ImageData,
        options: Optional[Dict[str, Any]] = None
    ) -> VisionAnalysisResult:
        """
        Analyze an image for pharmaceutical content.
        
        Args:
            image: Image data to analyze
            options: Optional configuration
                - confidence_threshold: Override default threshold
                - max_detections: Maximum number of detections
                - detect_all: Return all detections, not just pharmaceutical
                
        Returns:
            VisionAnalysisResult with detected objects
        """
        start_time = time.time()
        options = options or {}
        
        # Load model if not loaded
        self._load_model()
        
        # Load image (now returns numpy array)
        cv_image = self._load_image(image)
        img_height, img_width = cv_image.shape[:2]  # OpenCV: (height, width, channels)
        
        # Run inference
        confidence_threshold = options.get("confidence_threshold", self._confidence_threshold)
        
        try:
            results = self._model(
                cv_image,
                conf=confidence_threshold,
                device=self._device,
                verbose=False
            )
        except Exception as e:
            raise VisionAnalysisError(f"YOLO inference failed: {e}")
        
        # Parse results
        detected_objects: List[DetectedObject] = []
        
        if results and len(results) > 0:
            result = results[0]
            
            if result.boxes is not None:
                for i, box in enumerate(result.boxes):
                    # Get box coordinates (xyxy format)
                    xyxy = box.xyxy[0].cpu().numpy()
                    x_min, y_min, x_max, y_max = xyxy
                    
                    # Normalize coordinates
                    bbox = BoundingBox(
                        x_min=x_min / img_width,
                        y_min=y_min / img_height,
                        x_max=x_max / img_width,
                        y_max=y_max / img_height,
                        is_normalized=True
                    )
                    
                    # Get class and confidence
                    class_id = int(box.cls[0].cpu().numpy())
                    confidence = float(box.conf[0].cpu().numpy())
                    
                    # Get class name
                    class_name = result.names.get(class_id, "unknown").lower()
                    
                    # Map to our detection class
                    detection_class = self.CLASS_MAPPING.get(
                        class_name,
                        DetectionClass.UNKNOWN
                    )
                    
                    detected_objects.append(DetectedObject(
                        detection_class=detection_class,
                        bounding_box=bbox,
                        confidence=ConfidenceScore(value=confidence, source="yolo"),
                        attributes={"original_class": class_name}
                    ))
        
        # Check for pharmaceutical content
        is_pharmaceutical = any(
            obj.detection_class in self.PHARMACEUTICAL_CLASSES
            for obj in detected_objects
        )
        
        # If no pharmaceutical detected, add the full image as a text region for OCR
        # This is the fallback: allow OCR to try even if YOLO doesn't detect drug packaging
        if not is_pharmaceutical:
            # Add full image as potential text region
            detected_objects.append(DetectedObject(
                detection_class=DetectionClass.TEXT_REGION,
                bounding_box=BoundingBox(0, 0, 1, 1, is_normalized=True),
                confidence=ConfidenceScore(value=0.5, source="fallback"),
                attributes={"fallback": True}
            ))
            # Assume it might be pharmaceutical for OCR processing
            # Let subsequent stages (OCR, entity extraction) determine if it's actually a drug
            is_pharmaceutical = True
            self.logger.info("No pharmaceutical package detected by YOLO, using full image fallback for OCR")
        
        processing_time = (time.time() - start_time) * 1000
        
        return VisionAnalysisResult(
            detected_objects=detected_objects,
            is_pharmaceutical_image=is_pharmaceutical,
            processing_time_ms=processing_time,
            raw_output={"model": self.model_name, "num_detections": len(detected_objects)}
        )
    
    def is_pharmaceutical_image(self, image: ImageData) -> bool:
        """
        Quick check if image contains pharmaceutical content.
        
        Args:
            image: Image data to check
            
        Returns:
            True if pharmaceutical content detected
        """
        result = self.analyze(image, options={"confidence_threshold": 0.3})
        return result.is_pharmaceutical_image
    
    @property
    def model_name(self) -> str:
        """Get the model name."""
        if self._model_path:
            return f"YOLO-custom:{self._model_path}"
        return "YOLOv8n-pretrained"


class DummyVisionAnalyzer(VisionAnalyzerPort):
    """
    Dummy vision analyzer for testing without YOLO.
    
    Always returns the full image as a text region.
    """
    
    def analyze(
        self,
        image: ImageData,
        options: Optional[Dict[str, Any]] = None
    ) -> VisionAnalysisResult:
        """Return full image as detection region."""
        return VisionAnalysisResult(
            detected_objects=[
                DetectedObject(
                    detection_class=DetectionClass.DRUG_BOX,
                    bounding_box=BoundingBox(0, 0, 1, 1, is_normalized=True),
                    confidence=ConfidenceScore(value=0.8, source="dummy"),
                )
            ],
            is_pharmaceutical_image=True,
            processing_time_ms=1.0,
        )
    
    def is_pharmaceutical_image(self, image: ImageData) -> bool:
        """Always returns True for testing."""
        return True
    
    @property
    def model_name(self) -> str:
        return "DummyVisionAnalyzer"
