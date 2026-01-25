"""
Domain Entities

Core business entities representing the drug image understanding domain.
"""

from .drug_info import DrugInfo
from .extraction_result import (
    VisionAnalysisResult,
    TextExtractionResult,
    EntityExtractionResult,
    KnowledgeRetrievalResult,
    DetectedObject,
    TextBlock,
    ExtractedEntity,
    KnowledgeChunk,
)
from .pipeline_result import PipelineResult, PipelineError, StageStatus

__all__ = [
    "DrugInfo",
    "VisionAnalysisResult",
    "TextExtractionResult",
    "EntityExtractionResult",
    "KnowledgeRetrievalResult",
    "DetectedObject",
    "TextBlock",
    "ExtractedEntity",
    "KnowledgeChunk",
    "PipelineResult",
    "PipelineError",
    "StageStatus",
]
