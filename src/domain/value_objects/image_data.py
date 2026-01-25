"""
Image Data Value Object

Represents image data passed through the pipeline.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple
from pathlib import Path
import base64


@dataclass(frozen=True)
class ImageData:
    """
    Immutable value object representing image data.
    
    Can be constructed from file path, bytes, or base64 string.
    Provides lazy loading capabilities for memory efficiency.
    
    Attributes:
        source: Original source identifier (file path or URL)
        width: Image width in pixels (if known)
        height: Image height in pixels (if known)
        format: Image format (e.g., "jpeg", "png")
        _bytes: Raw image bytes (internal)
        _base64: Base64 encoded image (internal)
    """
    
    source: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    format: Optional[str] = None
    _bytes: Optional[bytes] = field(default=None, repr=False)
    _base64: Optional[str] = field(default=None, repr=False)
    
    def __post_init__(self) -> None:
        """Validate that at least one data source is provided."""
        if self._bytes is None and self._base64 is None and self.source is None:
            raise ValueError("ImageData must have at least one of: bytes, base64, or source path")
    
    @property
    def bytes(self) -> bytes:
        """
        Get raw image bytes, loading from source if necessary.
        
        Returns:
            Raw bytes of the image
            
        Raises:
            ValueError: If no data source is available
        """
        if self._bytes is not None:
            return self._bytes
        
        if self._base64 is not None:
            return base64.b64decode(self._base64)
        
        if self.source is not None:
            path = Path(self.source)
            if path.exists():
                return path.read_bytes()
        
        raise ValueError("Cannot load image bytes: no valid source available")
    
    @property
    def base64_string(self) -> str:
        """
        Get base64 encoded image string.
        
        Returns:
            Base64 encoded string of the image
        """
        if self._base64 is not None:
            return self._base64
        
        return base64.b64encode(self.bytes).decode("utf-8")
    
    @property
    def size(self) -> Optional[Tuple[int, int]]:
        """Get image dimensions as (width, height) tuple."""
        if self.width is not None and self.height is not None:
            return (self.width, self.height)
        return None
    
    @property
    def aspect_ratio(self) -> Optional[float]:
        """Calculate aspect ratio (width / height)."""
        if self.width and self.height:
            return self.width / self.height
        return None
    
    def __len__(self) -> int:
        """Return size of image data in bytes."""
        return len(self.bytes)
    
    def __str__(self) -> str:
        size_str = f"{self.width}x{self.height}" if self.size else "unknown size"
        format_str = self.format or "unknown format"
        return f"ImageData({size_str}, {format_str})"
    
    @classmethod
    def from_file(cls, file_path: str) -> "ImageData":
        """
        Create ImageData from a file path.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            ImageData instance
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {file_path}")
        
        # Determine format from extension
        format_map = {
            ".jpg": "jpeg",
            ".jpeg": "jpeg",
            ".png": "png",
            ".gif": "gif",
            ".bmp": "bmp",
            ".webp": "webp",
        }
        file_format = format_map.get(path.suffix.lower())
        
        return cls(
            source=str(path.absolute()),
            format=file_format,
            _bytes=path.read_bytes()
        )
    
    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        format: Optional[str] = None,
        source: Optional[str] = None
    ) -> "ImageData":
        """
        Create ImageData from raw bytes.
        
        Args:
            data: Raw image bytes
            format: Image format (e.g., "jpeg", "png")
            source: Optional source identifier
            
        Returns:
            ImageData instance
        """
        return cls(
            source=source,
            format=format,
            _bytes=data
        )
    
    @classmethod
    def from_base64(
        cls,
        base64_string: str,
        format: Optional[str] = None,
        source: Optional[str] = None
    ) -> "ImageData":
        """
        Create ImageData from base64 encoded string.
        
        Args:
            base64_string: Base64 encoded image data
            format: Image format (e.g., "jpeg", "png")
            source: Optional source identifier
            
        Returns:
            ImageData instance
        """
        # Handle data URL format
        if base64_string.startswith("data:"):
            # Extract format from data URL
            header, base64_data = base64_string.split(",", 1)
            if "image/" in header:
                format = header.split("image/")[1].split(";")[0]
            base64_string = base64_data
        
        return cls(
            source=source,
            format=format,
            _base64=base64_string
        )
