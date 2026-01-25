"""
Infrastructure Layer

Concrete implementations of domain ports (adapters).
Contains integrations with external services and libraries.
"""

from .vision import YOLOVisionAnalyzer, VisionAnalyzerFactory
from .ocr import PaddleOCRExtractor, TesseractOCRExtractor, OCRFactory
from .entity_extraction import HybridEntityExtractor, EntityExtractorFactory
from .rag import ChromaKnowledgeRetriever, KnowledgeRetrieverFactory
from .llm import OpenAIResponseGenerator, LLMFactory

__all__ = [
    # Vision
    "YOLOVisionAnalyzer",
    "VisionAnalyzerFactory",
    # OCR
    "PaddleOCRExtractor",
    "TesseractOCRExtractor",
    "OCRFactory",
    # Entity Extraction
    "HybridEntityExtractor",
    "EntityExtractorFactory",
    # RAG
    "ChromaKnowledgeRetriever",
    "KnowledgeRetrieverFactory",
    # LLM
    "OpenAIResponseGenerator",
    "LLMFactory",
]
