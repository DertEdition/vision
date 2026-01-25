"""
OpenCV-based Image Processing Utilities

Optimized image processing functions using OpenCV for better performance.
OpenCV uses C++ backend which is significantly faster than PIL.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def bytes_to_cv2(image_bytes: bytes) -> np.ndarray:
    """
    Convert image bytes to OpenCV BGR format.
    
    Args:
        image_bytes: Raw image bytes (JPEG, PNG, etc.)
        
    Returns:
        OpenCV image in BGR format (numpy array)
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Failed to decode image from bytes")
    
    return img


def cv2_to_rgb(img: np.ndarray) -> np.ndarray:
    """Convert BGR (OpenCV default) to RGB."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def cv2_to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert BGR to grayscale."""
    if len(img.shape) == 2:
        return img  # Already grayscale
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def resize_image(
    img: np.ndarray,
    max_dimension: int = 1800,
    interpolation: int = cv2.INTER_AREA
) -> Tuple[np.ndarray, float]:
    """
    Resize image if larger than max_dimension while preserving aspect ratio.
    
    Args:
        img: Input image
        max_dimension: Maximum allowed dimension
        interpolation: OpenCV interpolation method
            - INTER_AREA: Best for shrinking (default)
            - INTER_LINEAR: Fast bilinear
            - INTER_CUBIC: Slower but better quality
            
    Returns:
        Tuple of (resized_image, scale_factor)
    """
    height, width = img.shape[:2]
    
    if max(height, width) <= max_dimension:
        return img, 1.0
    
    scale = max_dimension / max(height, width)
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    resized = cv2.resize(img, (new_width, new_height), interpolation=interpolation)
    
    logger.debug(f"Resized image from {width}x{height} to {new_width}x{new_height}")
    
    return resized, scale


def enhance_for_ocr(img: np.ndarray) -> np.ndarray:
    """
    Enhance image for better OCR accuracy.
    
    Uses adaptive techniques that work well for drug package text:
    1. Convert to grayscale
    2. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    3. Denoise while preserving edges
    4. Sharpen text edges
    
    Args:
        img: Input image (BGR or grayscale)
        
    Returns:
        Enhanced grayscale image optimized for OCR
    """
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Apply CLAHE for adaptive contrast enhancement
    # This works much better than global histogram equalization
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Denoise while preserving edges (bilateral filter)
    denoised = cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)
    
    # Sharpen using unsharp masking
    gaussian = cv2.GaussianBlur(denoised, (0, 0), 3)
    sharpened = cv2.addWeighted(denoised, 1.5, gaussian, -0.5, 0)
    
    return sharpened


def enhance_for_ocr_binarized(img: np.ndarray) -> np.ndarray:
    """
    Enhance and binarize image for OCR (better for printed text).
    
    Uses adaptive thresholding which handles varying lighting conditions
    better than global thresholding.
    
    Args:
        img: Input image (BGR or grayscale)
        
    Returns:
        Binarized image optimized for OCR
    """
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Apply CLAHE first
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
    
    # Adaptive thresholding - works better for varying lighting
    binary = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=2
    )
    
    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return cleaned


def preprocess_for_yolo(
    img: np.ndarray,
    target_size: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Preprocess image for YOLO inference.
    
    Args:
        img: Input BGR image
        target_size: Optional target size (width, height)
        
    Returns:
        Preprocessed RGB image ready for YOLO
    """
    # YOLO expects RGB
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if target_size:
        rgb = cv2.resize(rgb, target_size, interpolation=cv2.INTER_LINEAR)
    
    return rgb


def detect_text_regions(img: np.ndarray) -> list:
    """
    Detect potential text regions using morphological operations.
    
    This is a fast pre-filter before OCR to focus on text areas.
    
    Args:
        img: Input image (BGR)
        
    Returns:
        List of bounding boxes [(x, y, w, h), ...]
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    
    # Apply morphological gradient to detect edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    
    # Binarize
    _, binary = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Connect text characters together
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    connected = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter and return bounding boxes
    text_regions = []
    height, width = gray.shape[:2]
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter by size (text regions have specific aspect ratios)
        aspect_ratio = w / h if h > 0 else 0
        area_ratio = (w * h) / (width * height)
        
        if 0.1 < aspect_ratio < 15 and 0.001 < area_ratio < 0.5:
            text_regions.append((x, y, w, h))
    
    return text_regions


def auto_rotate(img: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Automatically detect and correct image rotation (deskew).
    
    Useful for drug packages photographed at an angle.
    
    Args:
        img: Input image
        
    Returns:
        Tuple of (rotated_image, rotation_angle)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    
    # Detect edges
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Detect lines using Hough transform
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=100,
        minLineLength=100,
        maxLineGap=10
    )
    
    if lines is None:
        return img, 0.0
    
    # Calculate average angle
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # Only consider small angles (text should be roughly horizontal)
        if -30 < angle < 30:
            angles.append(angle)
    
    if not angles:
        return img, 0.0
    
    median_angle = np.median(angles)
    
    # Only rotate if angle is significant
    if abs(median_angle) < 1.0:
        return img, 0.0
    
    # Rotate image
    height, width = img.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(
        img, rotation_matrix, (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )
    
    logger.debug(f"Auto-rotated image by {median_angle:.2f} degrees")
    
    return rotated, median_angle


def crop_to_content(img: np.ndarray, padding: int = 10) -> np.ndarray:
    """
    Crop image to content area (remove excess background).
    
    Args:
        img: Input image
        padding: Padding around detected content
        
    Returns:
        Cropped image
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    
    # Threshold to find content
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find content bounds
    coords = cv2.findNonZero(binary)
    if coords is None:
        return img
    
    x, y, w, h = cv2.boundingRect(coords)
    
    # Add padding
    height, width = img.shape[:2]
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(width - x, w + 2 * padding)
    h = min(height - y, h + 2 * padding)
    
    return img[y:y+h, x:x+w]


class ImagePreprocessor:
    """
    Configurable image preprocessing pipeline.
    
    Provides a clean interface for chaining preprocessing operations.
    """
    
    def __init__(
        self,
        max_dimension: int = 1800,
        enhance_contrast: bool = True,
        denoise: bool = True,
        auto_deskew: bool = False,
        binarize: bool = False
    ):
        self.max_dimension = max_dimension
        self.enhance_contrast = enhance_contrast
        self.denoise = denoise
        self.auto_deskew = auto_deskew
        self.binarize = binarize
        
        self.logger = logging.getLogger(f"{__name__}.ImagePreprocessor")
    
    def process(self, image_bytes: bytes) -> np.ndarray:
        """
        Process image bytes through the preprocessing pipeline.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Processed image as numpy array
        """
        # Load image
        img = bytes_to_cv2(image_bytes)
        
        # Resize if needed
        img, scale = resize_image(img, self.max_dimension)
        
        # Auto deskew if enabled
        if self.auto_deskew:
            img, _ = auto_rotate(img)
        
        # Enhance for OCR
        if self.binarize:
            img = enhance_for_ocr_binarized(img)
        elif self.enhance_contrast:
            img = enhance_for_ocr(img)
        
        return img
    
    def process_for_yolo(self, image_bytes: bytes) -> np.ndarray:
        """
        Process image for YOLO inference.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            RGB image ready for YOLO
        """
        img = bytes_to_cv2(image_bytes)
        
        # Resize if very large
        if max(img.shape[:2]) > 2000:
            img, _ = resize_image(img, 2000)
        
        # Convert to RGB for YOLO
        return cv2_to_rgb(img)
