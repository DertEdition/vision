"""
Ports (Interfaces)

Abstract interfaces defining the contracts for infrastructure adapters.
Following Hexagonal Architecture / Ports & Adapters pattern.
"""

from .vision_analyzer import VisionAnalyzerPort
from .text_extractor import TextExtractorPort
from .entity_extractor import EntityExtractorPort
from .knowledge_retriever import KnowledgeRetrieverPort
from .response_generator import ResponseGeneratorPort

__all__ = [
    "VisionAnalyzerPort",
    "TextExtractorPort",
    "EntityExtractorPort",
    "KnowledgeRetrieverPort",
    "ResponseGeneratorPort",
]
