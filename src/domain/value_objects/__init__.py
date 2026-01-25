"""
Value Objects

Immutable objects that represent domain concepts with no identity.
"""

from .confidence_score import ConfidenceScore, ConfidenceLevel
from .bounding_box import BoundingBox
from .dosage_info import DosageForm, DosageInfo
from .image_data import ImageData

__all__ = [
    "ConfidenceScore",
    "ConfidenceLevel",
    "BoundingBox",
    "DosageForm",
    "DosageInfo",
    "ImageData",
]
