"""
Input Validation

Validation utilities for pipeline inputs.
"""

from typing import Optional, List, Tuple
from pathlib import Path

from ..domain.value_objects.image_data import ImageData
from ..domain.exceptions import InvalidImageError, InvalidInputError


# Supported image formats
SUPPORTED_FORMATS = {"jpeg", "jpg", "png", "bmp", "webp", "gif"}

# Maximum image dimensions
MAX_IMAGE_DIMENSION = 8192

# Maximum file size (10 MB)
MAX_FILE_SIZE = 10 * 1024 * 1024


def validate_image(image: ImageData) -> Tuple[bool, Optional[str]]:
    """
    Validate image data.
    
    Args:
        image: ImageData to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Check if we can get bytes
        image_bytes = image.bytes
        
        # Check file size
        if len(image_bytes) > MAX_FILE_SIZE:
            return False, f"Image size exceeds maximum ({MAX_FILE_SIZE / 1024 / 1024:.1f} MB)"
        
        # Check if it's a valid image
        try:
            from PIL import Image as PILImage
            from io import BytesIO
            
            pil_image = PILImage.open(BytesIO(image_bytes))
            pil_image.verify()
            
            # Reopen because verify() can only be called once
            pil_image = PILImage.open(BytesIO(image_bytes))
            
            # Check dimensions
            width, height = pil_image.size
            if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
                return False, f"Image dimensions exceed maximum ({MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION})"
            
            # Check format
            img_format = pil_image.format.lower() if pil_image.format else "unknown"
            if img_format not in SUPPORTED_FORMATS:
                return False, f"Unsupported image format: {img_format}"
            
        except Exception as e:
            return False, f"Invalid image data: {e}"
        
        return True, None
        
    except Exception as e:
        return False, f"Failed to read image: {e}"


def validate_image_file(file_path: str) -> Tuple[bool, Optional[str]]:
    """
    Validate an image file path.
    
    Args:
        file_path: Path to image file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    path = Path(file_path)
    
    if not path.exists():
        return False, f"File not found: {file_path}"
    
    if not path.is_file():
        return False, f"Not a file: {file_path}"
    
    suffix = path.suffix.lower().lstrip(".")
    if suffix not in SUPPORTED_FORMATS:
        return False, f"Unsupported file format: {suffix}"
    
    if path.stat().st_size > MAX_FILE_SIZE:
        return False, f"File size exceeds maximum ({MAX_FILE_SIZE / 1024 / 1024:.1f} MB)"
    
    return True, None


def validate_text(text: str, min_length: int = 1, max_length: int = 100000) -> Tuple[bool, Optional[str]]:
    """
    Validate text input.
    
    Args:
        text: Text to validate
        min_length: Minimum text length
        max_length: Maximum text length
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not text:
        return False, "Text cannot be empty"
    
    if len(text) < min_length:
        return False, f"Text too short (minimum {min_length} characters)"
    
    if len(text) > max_length:
        return False, f"Text too long (maximum {max_length} characters)"
    
    return True, None


def validate_options(options: dict, allowed_keys: List[str]) -> Tuple[bool, Optional[str]]:
    """
    Validate options dictionary.
    
    Args:
        options: Options to validate
        allowed_keys: List of allowed option keys
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(options, dict):
        return False, "Options must be a dictionary"
    
    unknown_keys = set(options.keys()) - set(allowed_keys)
    if unknown_keys:
        return False, f"Unknown options: {', '.join(unknown_keys)}"
    
    return True, None
