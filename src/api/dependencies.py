"""
Dependency Injection

FastAPI dependencies for the Drug Image Analysis API.
Manages the lifecycle of shared resources like the pipeline and service.
"""

import logging
from functools import lru_cache
from typing import Generator

from ..application.services import DrugAnalysisService
from ..application.services.medical_analysis_service import MedicalAnalysisService
from ..application.pipeline import PipelineBuilder
from ..application.pipeline.medical_pipeline import MedicalPipelineBuilder
from ..infrastructure.vision import VisionAnalyzerFactory, VisionAnalyzerType
from ..infrastructure.ocr import OCRFactory, OCRType
from ..infrastructure.entity_extraction import EntityExtractorFactory, EntityExtractorType
from ..infrastructure.rag import KnowledgeRetrieverFactory, KnowledgeRetrieverType
from ..infrastructure.llm import LLMFactory, LLMType
from ..infrastructure.classification import (
    ClassifierFactory,
    ClassifierType,
)


logger = logging.getLogger(__name__)


def _create_pipeline(config):
    """
    Create a configured pipeline from configuration.
    
    This mirrors the logic from main.py but is decoupled from CLI.
    """
    builder = PipelineBuilder()
    
    builder.with_vision_analyzer(
        VisionAnalyzerFactory.create(
            VisionAnalyzerType(config.vision.type),
            model_path=config.vision.model_path,
            confidence_threshold=config.vision.confidence_threshold,
            device=config.vision.device
        )
    )
    
    builder.with_text_extractor(
        OCRFactory.create(
            OCRType(config.ocr.type),
            lang=config.ocr.language,
            use_gpu=config.ocr.use_gpu
        )
    )
    
    builder.with_entity_extractor(
        EntityExtractorFactory.create(
            EntityExtractorType(config.entity_extraction.type),
            use_llm_refinement=config.entity_extraction.use_llm_refinement
        )
    )
    
    builder.with_knowledge_retriever(
        KnowledgeRetrieverFactory.create(
            KnowledgeRetrieverType(config.rag.type),
            persist_directory=config.rag.persist_directory,
            collection_name=config.rag.collection_name
        )
    )
    
    builder.with_response_generator(
        LLMFactory.create(
            LLMType(config.llm.type),
            api_key=config.llm.api_key,
            model=config.llm.model,
            temperature=config.llm.temperature,
            language=config.llm.language
        )
    )
    
    return builder.build()


@lru_cache()
def get_config():
    """
    Get application configuration as a singleton.
    
    Uses lru_cache to ensure config is loaded only once.
    """
    # Import here to avoid circular imports
    import sys
    sys.path.insert(0, '.')
    from config import get_default_config
    
    config = get_default_config()
    logger.info("Configuration loaded")
    return config


@lru_cache()
def get_pipeline():
    """
    Get pipeline orchestrator as a singleton.
    
    The pipeline is expensive to create (loads ML models),
    so we cache it for the lifetime of the application.
    """
    config = get_config()
    logger.info("Initializing pipeline...")
    pipeline = _create_pipeline(config)
    logger.info("Pipeline initialized successfully")
    return pipeline


@lru_cache()
def get_analysis_service() -> DrugAnalysisService:
    """
    Get DrugAnalysisService as a singleton.
    
    This is the main dependency injected into route handlers.
    
    Returns:
        Configured DrugAnalysisService instance
    """
    pipeline = get_pipeline()
    service = DrugAnalysisService(pipeline)
    logger.info("DrugAnalysisService initialized")
    return service


def reset_dependencies():
    """
    Clear cached dependencies.
    
    Useful for testing or reconfiguration at runtime.
    """
    get_config.cache_clear()
    get_pipeline.cache_clear()
    get_analysis_service.cache_clear()
    get_medical_analysis_service.cache_clear()
    logger.info("Dependencies reset")


def _create_medical_pipelines(config):
    """
    Create configured medical analysis pipelines from configuration.
    
    Returns:
        Tuple of (dermatology_pipeline, chest_xray_pipeline)
    """
    derma_pipeline = None
    chest_pipeline = None
    
    if config.dermatology.enabled:
        try:
            derma_classifier = ClassifierFactory.create(
                ClassifierType.DERMATOLOGY,
                model_path=config.dermatology.model_path,
                device=config.dermatology.device,
                confidence_threshold=config.dermatology.confidence_threshold,
                image_size=config.dermatology.image_size,
            )
            derma_pipeline = MedicalPipelineBuilder()\
                .with_classifier(derma_classifier)\
                .with_diagnosis_type("dermatology")\
                .with_timeout(config.pipeline.timeout_seconds)\
                .build()
            logger.info("Dermatology pipeline created")
        except Exception as e:
            logger.warning(f"Failed to create dermatology pipeline: {e}")
    
    if config.chest_xray.enabled:
        try:
            chest_classifier = ClassifierFactory.create(
                ClassifierType.CHEST_XRAY,
                model_path=config.chest_xray.model_path,
                device=config.chest_xray.device,
                confidence_threshold=config.chest_xray.confidence_threshold,
                image_size=config.chest_xray.image_size,
            )
            chest_pipeline = MedicalPipelineBuilder()\
                .with_classifier(chest_classifier)\
                .with_diagnosis_type("chest_xray")\
                .with_timeout(config.pipeline.timeout_seconds)\
                .build()
            logger.info("Chest X-ray pipeline created")
        except Exception as e:
            logger.warning(f"Failed to create chest X-ray pipeline: {e}")
    
    return derma_pipeline, chest_pipeline


@lru_cache()
def get_medical_analysis_service() -> MedicalAnalysisService:
    """
    Get MedicalAnalysisService as a singleton.
    
    Creates and caches dermatology and chest X-ray pipelines.
    
    Returns:
        Configured MedicalAnalysisService instance
    """
    config = get_config()
    logger.info("Initializing medical analysis service...")
    
    derma_pipeline, chest_pipeline = _create_medical_pipelines(config)
    
    service = MedicalAnalysisService(
        dermatology_pipeline=derma_pipeline,
        chest_xray_pipeline=chest_pipeline,
    )
    logger.info("MedicalAnalysisService initialized")
    return service
