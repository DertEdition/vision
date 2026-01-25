"""
Utility modules for the infrastructure layer.
"""

from .image_processing import (
    bytes_to_cv2,
    cv2_to_rgb,
    cv2_to_grayscale,
    resize_image,
    enhance_for_ocr,
    enhance_for_ocr_binarized,
    preprocess_for_yolo,
    detect_text_regions,
    auto_rotate,
    crop_to_content,
    ImagePreprocessor,
)

__all__ = [
    "bytes_to_cv2",
    "cv2_to_rgb",
    "cv2_to_grayscale",
    "resize_image",
    "enhance_for_ocr",
    "enhance_for_ocr_binarized",
    "preprocess_for_yolo",
    "detect_text_regions",
    "auto_rotate",
    "crop_to_content",
    "ImagePreprocessor",
]
