"""
Pipeline Orchestrator

Main orchestration logic for the drug image analysis pipeline.
Implements Chain of Responsibility pattern.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import logging
import time

from .context import PipelineContext
from .stages import (
    PipelineStageExecutor,
    StageConfig,
    VisionAnalysisStage,
    TextExtractionStage,
    EntityExtractionStage,
    KnowledgeRetrievalStage,
    ResponseGenerationStage,
)
from ...domain.value_objects.image_data import ImageData
from ...domain.entities.pipeline_result import PipelineResult, PipelineStage
from ...domain.ports.vision_analyzer import VisionAnalyzerPort
from ...domain.ports.text_extractor import TextExtractorPort
from ...domain.ports.entity_extractor import EntityExtractorPort
from ...domain.ports.knowledge_retriever import KnowledgeRetrieverPort
from ...domain.ports.response_generator import ResponseGeneratorPort
from ...domain.exceptions import PipelineConfigurationError


logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """
    Configuration for the pipeline orchestrator.
    
    Attributes:
        timeout_seconds: Overall pipeline timeout
        fail_fast: If True, stop on first stage failure
        stages: Per-stage configurations
    """
    
    timeout_seconds: float = 120.0
    fail_fast: bool = False
    stages: Dict[PipelineStage, StageConfig] = field(default_factory=dict)
    
    def get_stage_config(self, stage: PipelineStage) -> StageConfig:
        """Get configuration for a specific stage."""
        return self.stages.get(stage, StageConfig())


class PipelineOrchestrator:
    """
    Main pipeline orchestrator for drug image analysis.
    
    Orchestrates the flow: VISION → OCR → ENTITY → RAG → LLM
    
    Features:
    - Sequential stage execution
    - Error accumulation without crash
    - Partial output on stage failure
    - Configurable fail-fast or fail-soft behavior
    - Comprehensive logging and metrics
    
    Usage:
        orchestrator = PipelineOrchestrator(
            vision_analyzer=yolo_analyzer,
            text_extractor=paddle_ocr,
            entity_extractor=hybrid_extractor,
            knowledge_retriever=chroma_retriever,
            response_generator=openai_generator
        )
        
        result = orchestrator.run(image_data)
    """
    
    def __init__(
        self,
        vision_analyzer: VisionAnalyzerPort,
        text_extractor: TextExtractorPort,
        entity_extractor: EntityExtractorPort,
        knowledge_retriever: KnowledgeRetrieverPort,
        response_generator: ResponseGeneratorPort,
        config: Optional[PipelineConfig] = None
    ):
        """
        Initialize the pipeline orchestrator.
        
        Args:
            vision_analyzer: Vision analysis implementation
            text_extractor: OCR implementation
            entity_extractor: Entity extraction implementation
            knowledge_retriever: RAG implementation
            response_generator: LLM implementation
            config: Pipeline configuration
        """
        self.config = config or PipelineConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Store adapters
        self._vision_analyzer = vision_analyzer
        self._text_extractor = text_extractor
        self._entity_extractor = entity_extractor
        self._knowledge_retriever = knowledge_retriever
        self._response_generator = response_generator
        
        # Build stage chain
        self._stages = self._build_stages()
        
        self.logger.info(f"Pipeline initialized with {len(self._stages)} stages")
    
    def _build_stages(self) -> List[PipelineStageExecutor]:
        """Build the ordered list of pipeline stages."""
        return [
            VisionAnalysisStage(
                analyzer=self._vision_analyzer,
                config=self.config.get_stage_config(PipelineStage.VISION_ANALYSIS)
            ),
            TextExtractionStage(
                extractor=self._text_extractor,
                config=self.config.get_stage_config(PipelineStage.TEXT_EXTRACTION)
            ),
            EntityExtractionStage(
                extractor=self._entity_extractor,
                config=self.config.get_stage_config(PipelineStage.ENTITY_EXTRACTION)
            ),
            KnowledgeRetrievalStage(
                retriever=self._knowledge_retriever,
                config=self.config.get_stage_config(PipelineStage.KNOWLEDGE_RETRIEVAL)
            ),
            ResponseGenerationStage(
                generator=self._response_generator,
                config=self.config.get_stage_config(PipelineStage.RESPONSE_GENERATION)
            ),
        ]
    
    def run(
        self,
        image: ImageData,
        options: Optional[Dict[str, Any]] = None
    ) -> PipelineResult:
        """
        Run the complete pipeline on an image.
        
        Args:
            image: Input image to analyze
            options: Optional configuration options
                - vision: Vision stage options
                - ocr: OCR stage options
                - entity: Entity extraction options
                - rag: Knowledge retrieval options
                - llm: Response generation options
                
        Returns:
            PipelineResult containing analysis results
        """
        start_time = time.time()
        
        # Create context
        context = PipelineContext.create(image=image, options=options or {})
        self.logger.info(f"Starting pipeline execution (request_id={context.request_id})")
        
        # Execute stages
        stages_completed = 0
        stages_failed = 0
        
        for stage_executor in self._stages:
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > self.config.timeout_seconds:
                self.logger.error(f"Pipeline timeout after {elapsed:.2f}s")
                context.add_error(
                    stage=stage_executor.stage,
                    error_type="PipelineTimeout",
                    message=f"Pipeline timed out after {self.config.timeout_seconds} seconds",
                    is_recoverable=False
                )
                break
            
            # Check abort flag
            if context.should_abort:
                self.logger.warning(f"Pipeline aborted: {context.abort_reason}")
                break
            
            # Execute stage
            success = stage_executor.run(context)
            
            if success:
                stages_completed += 1
            else:
                stages_failed += 1
                
                if self.config.fail_fast:
                    self.logger.warning(f"Fail-fast enabled, stopping pipeline after {stage_executor.name} failure")
                    break
        
        # Build result
        result = context.to_pipeline_result()
        
        elapsed_total = (time.time() - start_time) * 1000
        result.total_processing_time_ms = elapsed_total
        
        self.logger.info(
            f"Pipeline completed: {stages_completed} succeeded, {stages_failed} failed, "
            f"total time: {elapsed_total:.2f}ms"
        )
        
        return result
    
    def run_partial(
        self,
        image: ImageData,
        until_stage: PipelineStage,
        options: Optional[Dict[str, Any]] = None
    ) -> PipelineContext:
        """
        Run pipeline up to a specific stage (for testing/debugging).
        
        Args:
            image: Input image to analyze
            until_stage: Stop after this stage
            options: Optional configuration options
            
        Returns:
            PipelineContext with partial results
        """
        context = PipelineContext.create(image=image, options=options or {})
        
        for stage_executor in self._stages:
            if context.should_abort:
                break
            
            stage_executor.run(context)
            
            if stage_executor.stage == until_stage:
                break
        
        return context
    
    def validate_configuration(self) -> bool:
        """
        Validate that the pipeline is properly configured.
        
        Returns:
            True if configuration is valid
            
        Raises:
            PipelineConfigurationError: If configuration is invalid
        """
        missing = []
        
        if self._vision_analyzer is None:
            missing.append("vision_analyzer")
        if self._text_extractor is None:
            missing.append("text_extractor")
        if self._entity_extractor is None:
            missing.append("entity_extractor")
        if self._knowledge_retriever is None:
            missing.append("knowledge_retriever")
        if self._response_generator is None:
            missing.append("response_generator")
        
        if missing:
            raise PipelineConfigurationError(
                message=f"Pipeline is missing required components: {', '.join(missing)}",
                missing_components=missing
            )
        
        return True
    
    @property
    def stage_count(self) -> int:
        """Get the number of stages in the pipeline."""
        return len(self._stages)
    
    @property
    def stage_names(self) -> List[str]:
        """Get the names of all stages."""
        return [s.name for s in self._stages]


class PipelineBuilder:
    """
    Builder for constructing pipeline orchestrators.
    
    Usage:
        pipeline = (
            PipelineBuilder()
            .with_vision_analyzer(yolo_analyzer)
            .with_text_extractor(paddle_ocr)
            .with_entity_extractor(hybrid_extractor)
            .with_knowledge_retriever(chroma_retriever)
            .with_response_generator(openai_generator)
            .with_config(pipeline_config)
            .build()
        )
    """
    
    def __init__(self):
        self._vision_analyzer: Optional[VisionAnalyzerPort] = None
        self._text_extractor: Optional[TextExtractorPort] = None
        self._entity_extractor: Optional[EntityExtractorPort] = None
        self._knowledge_retriever: Optional[KnowledgeRetrieverPort] = None
        self._response_generator: Optional[ResponseGeneratorPort] = None
        self._config: Optional[PipelineConfig] = None
    
    def with_vision_analyzer(self, analyzer: VisionAnalyzerPort) -> "PipelineBuilder":
        """Set the vision analyzer."""
        self._vision_analyzer = analyzer
        return self
    
    def with_text_extractor(self, extractor: TextExtractorPort) -> "PipelineBuilder":
        """Set the text extractor."""
        self._text_extractor = extractor
        return self
    
    def with_entity_extractor(self, extractor: EntityExtractorPort) -> "PipelineBuilder":
        """Set the entity extractor."""
        self._entity_extractor = extractor
        return self
    
    def with_knowledge_retriever(self, retriever: KnowledgeRetrieverPort) -> "PipelineBuilder":
        """Set the knowledge retriever."""
        self._knowledge_retriever = retriever
        return self
    
    def with_response_generator(self, generator: ResponseGeneratorPort) -> "PipelineBuilder":
        """Set the response generator."""
        self._response_generator = generator
        return self
    
    def with_config(self, config: PipelineConfig) -> "PipelineBuilder":
        """Set the pipeline configuration."""
        self._config = config
        return self
    
    def build(self) -> PipelineOrchestrator:
        """
        Build the pipeline orchestrator.
        
        Returns:
            Configured PipelineOrchestrator
            
        Raises:
            PipelineConfigurationError: If required components are missing
        """
        missing = []
        
        if self._vision_analyzer is None:
            missing.append("vision_analyzer")
        if self._text_extractor is None:
            missing.append("text_extractor")
        if self._entity_extractor is None:
            missing.append("entity_extractor")
        if self._knowledge_retriever is None:
            missing.append("knowledge_retriever")
        if self._response_generator is None:
            missing.append("response_generator")
        
        if missing:
            raise PipelineConfigurationError(
                message=f"Cannot build pipeline, missing: {', '.join(missing)}",
                missing_components=missing
            )
        
        return PipelineOrchestrator(
            vision_analyzer=self._vision_analyzer,
            text_extractor=self._text_extractor,
            entity_extractor=self._entity_extractor,
            knowledge_retriever=self._knowledge_retriever,
            response_generator=self._response_generator,
            config=self._config
        )
