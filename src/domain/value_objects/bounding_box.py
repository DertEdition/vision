"""
Bounding Box Value Object

Represents a rectangular region in an image.
"""

from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass(frozen=True)
class BoundingBox:
    """
    Immutable value object representing a bounding box in an image.
    
    Uses normalized coordinates (0.0 to 1.0) relative to image dimensions,
    or absolute pixel coordinates based on the is_normalized flag.
    
    Attributes:
        x_min: Left edge coordinate
        y_min: Top edge coordinate
        x_max: Right edge coordinate
        y_max: Bottom edge coordinate
        is_normalized: Whether coordinates are normalized (0-1) or absolute pixels
        label: Optional label for the detected region
    """
    
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    is_normalized: bool = True
    label: Optional[str] = None
    
    def __post_init__(self) -> None:
        """Validate bounding box coordinates."""
        if self.x_min > self.x_max:
            raise ValueError(f"x_min ({self.x_min}) must be <= x_max ({self.x_max})")
        if self.y_min > self.y_max:
            raise ValueError(f"y_min ({self.y_min}) must be <= y_max ({self.y_max})")
        
        if self.is_normalized:
            for coord_name, coord_value in [
                ("x_min", self.x_min), ("y_min", self.y_min),
                ("x_max", self.x_max), ("y_max", self.y_max)
            ]:
                if not 0.0 <= coord_value <= 1.0:
                    raise ValueError(
                        f"Normalized {coord_name} must be between 0.0 and 1.0, got {coord_value}"
                    )
    
    @property
    def width(self) -> float:
        """Calculate box width."""
        return self.x_max - self.x_min
    
    @property
    def height(self) -> float:
        """Calculate box height."""
        return self.y_max - self.y_min
    
    @property
    def area(self) -> float:
        """Calculate box area."""
        return self.width * self.height
    
    @property
    def center(self) -> Tuple[float, float]:
        """Get center point coordinates."""
        return (
            (self.x_min + self.x_max) / 2,
            (self.y_min + self.y_max) / 2
        )
    
    def to_absolute(self, image_width: int, image_height: int) -> "BoundingBox":
        """
        Convert normalized coordinates to absolute pixel coordinates.
        
        Args:
            image_width: Width of the image in pixels
            image_height: Height of the image in pixels
            
        Returns:
            New BoundingBox with absolute coordinates
        """
        if not self.is_normalized:
            return self
        
        return BoundingBox(
            x_min=self.x_min * image_width,
            y_min=self.y_min * image_height,
            x_max=self.x_max * image_width,
            y_max=self.y_max * image_height,
            is_normalized=False,
            label=self.label
        )
    
    def to_normalized(self, image_width: int, image_height: int) -> "BoundingBox":
        """
        Convert absolute pixel coordinates to normalized coordinates.
        
        Args:
            image_width: Width of the image in pixels
            image_height: Height of the image in pixels
            
        Returns:
            New BoundingBox with normalized coordinates
        """
        if self.is_normalized:
            return self
        
        return BoundingBox(
            x_min=self.x_min / image_width,
            y_min=self.y_min / image_height,
            x_max=self.x_max / image_width,
            y_max=self.y_max / image_height,
            is_normalized=True,
            label=self.label
        )
    
    def to_xyxy(self) -> Tuple[float, float, float, float]:
        """Return coordinates as (x_min, y_min, x_max, y_max) tuple."""
        return (self.x_min, self.y_min, self.x_max, self.y_max)
    
    def to_xywh(self) -> Tuple[float, float, float, float]:
        """Return as (x_center, y_center, width, height) tuple."""
        cx, cy = self.center
        return (cx, cy, self.width, self.height)
    
    def expand(self, factor: float) -> "BoundingBox":
        """
        Expand the bounding box by a factor around its center.
        
        Args:
            factor: Expansion factor (1.0 = no change, 1.5 = 50% larger)
            
        Returns:
            New expanded BoundingBox
        """
        cx, cy = self.center
        new_width = self.width * factor
        new_height = self.height * factor
        
        return BoundingBox(
            x_min=cx - new_width / 2,
            y_min=cy - new_height / 2,
            x_max=cx + new_width / 2,
            y_max=cy + new_height / 2,
            is_normalized=self.is_normalized,
            label=self.label
        )
    
    def __str__(self) -> str:
        coord_type = "norm" if self.is_normalized else "abs"
        return f"BoundingBox({coord_type}: [{self.x_min:.3f}, {self.y_min:.3f}, {self.x_max:.3f}, {self.y_max:.3f}])"
    
    @classmethod
    def from_xyxy(
        cls,
        x_min: float,
        y_min: float,
        x_max: float,
        y_max: float,
        is_normalized: bool = True,
        label: Optional[str] = None
    ) -> "BoundingBox":
        """Create from (x_min, y_min, x_max, y_max) coordinates."""
        return cls(
            x_min=x_min,
            y_min=y_min,
            x_max=x_max,
            y_max=y_max,
            is_normalized=is_normalized,
            label=label
        )
    
    @classmethod
    def from_xywh(
        cls,
        x_center: float,
        y_center: float,
        width: float,
        height: float,
        is_normalized: bool = True,
        label: Optional[str] = None
    ) -> "BoundingBox":
        """Create from (x_center, y_center, width, height) coordinates."""
        return cls(
            x_min=x_center - width / 2,
            y_min=y_center - height / 2,
            x_max=x_center + width / 2,
            y_max=y_center + height / 2,
            is_normalized=is_normalized,
            label=label
        )
