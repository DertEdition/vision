"""
Confidence Score Value Object

Represents confidence levels for predictions and extractions.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ConfidenceLevel(Enum):
    """Categorical confidence levels for user-facing decisions."""
    
    VERY_LOW = "very_low"      # < 0.3: Unreliable, should warn user
    LOW = "low"                 # 0.3 - 0.5: Low confidence, proceed with caution
    MEDIUM = "medium"           # 0.5 - 0.7: Moderate confidence
    HIGH = "high"               # 0.7 - 0.9: Good confidence
    VERY_HIGH = "very_high"     # > 0.9: Excellent confidence


@dataclass(frozen=True)
class ConfidenceScore:
    """
    Immutable value object representing a confidence score.
    
    Attributes:
        value: Float between 0.0 and 1.0 representing confidence
        source: Optional identifier for what produced this score
    """
    
    value: float
    source: Optional[str] = None
    
    def __post_init__(self) -> None:
        """Validate confidence score is within valid range."""
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(f"Confidence score must be between 0.0 and 1.0, got {self.value}")
    
    @property
    def level(self) -> ConfidenceLevel:
        """Get categorical confidence level."""
        if self.value < 0.3:
            return ConfidenceLevel.VERY_LOW
        elif self.value < 0.5:
            return ConfidenceLevel.LOW
        elif self.value < 0.7:
            return ConfidenceLevel.MEDIUM
        elif self.value < 0.9:
            return ConfidenceLevel.HIGH
        else:
            return ConfidenceLevel.VERY_HIGH
    
    @property
    def is_reliable(self) -> bool:
        """Check if confidence is at least medium level."""
        return self.value >= 0.5
    
    @property
    def requires_warning(self) -> bool:
        """Check if confidence is too low and requires user warning."""
        return self.value < 0.5
    
    def __str__(self) -> str:
        return f"{self.value:.2%}"
    
    def __repr__(self) -> str:
        return f"ConfidenceScore(value={self.value:.4f}, level={self.level.value})"
    
    @classmethod
    def zero(cls) -> "ConfidenceScore":
        """Create a zero confidence score."""
        return cls(value=0.0, source="default")
    
    @classmethod
    def full(cls) -> "ConfidenceScore":
        """Create a full confidence score."""
        return cls(value=1.0, source="default")
    
    @classmethod
    def from_percentage(cls, percentage: float, source: Optional[str] = None) -> "ConfidenceScore":
        """Create from a percentage value (0-100)."""
        return cls(value=percentage / 100.0, source=source)
