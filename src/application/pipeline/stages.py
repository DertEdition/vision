"""
Pipeline Stage Definitions

Defines individual pipeline stages and their execution logic.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Callable
from enum import Enum
import logging

from .context import PipelineContext
from ...domain.entities.pipeline_result import PipelineStage, PipelineError
from ...domain.exceptions import DomainException


logger = logging.getLogger(__name__)


@dataclass
class StageConfig:
    """
    Configuration for a pipeline stage.
    
    Attributes:
        enabled: Whether the stage is enabled
        timeout_seconds: Maximum execution time
        retry_count: Number of retries on failure
        retry_delay_seconds: Delay between retries
        fail_soft: If True, continue pipeline on failure
        options: Stage-specific options
    """
    
    enabled: bool = True
    timeout_seconds: float = 30.0
    retry_count: int = 2
    retry_delay_seconds: float = 1.0
    fail_soft: bool = True
    options: Dict[str, Any] = field(default_factory=dict)


class PipelineStageExecutor(ABC):
    """
    Abstract base class for pipeline stage executors.
    
    Each stage in the pipeline implements this interface.
    Stages are responsible for:
    - Reading required data from context
    - Executing their specific logic
    - Writing results back to context
    - Handling errors appropriately
    """
    
    def __init__(self, config: Optional[StageConfig] = None):
        """
        Initialize the stage executor.
        
        Args:
            config: Stage configuration
        """
        self.config = config or StageConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @property
    @abstractmethod
    def stage(self) -> PipelineStage:
        """Get the pipeline stage this executor handles."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get human-readable stage name."""
        pass
    
    @abstractmethod
    def execute(self, context: PipelineContext) -> None:
        """
        Execute the stage logic.
        
        Args:
            context: Pipeline context to read from and write to
            
        Note:
            Implementations should:
            - Check prerequisites from context
            - Execute main logic
            - Store results in context
            - Handle errors and add them to context
        """
        pass
    
    def can_execute(self, context: PipelineContext) -> bool:
        """
        Check if this stage can execute given the current context.
        
        Args:
            context: Current pipeline context
            
        Returns:
            True if prerequisites are met
        """
        return not context.should_abort
    
    def run(self, context: PipelineContext) -> bool:
        """
        Run the stage with error handling and retries.
        
        Args:
            context: Pipeline context
            
        Returns:
            True if stage completed successfully
        """
        if not self.config.enabled:
            self.logger.info(f"Stage {self.name} is disabled, skipping")
            return True
        
        if not self.can_execute(context):
            self.logger.warning(f"Stage {self.name} cannot execute, prerequisites not met")
            return False
        
        context.start_stage(self.stage)
        
        attempts = 0
        last_error: Optional[Exception] = None
        
        while attempts <= self.config.retry_count:
            try:
                self.logger.info(f"Executing stage {self.name} (attempt {attempts + 1})")
                self.execute(context)
                context.finish_stage(self.stage)
                self.logger.info(f"Stage {self.name} completed successfully")
                return True
                
            except DomainException as e:
                last_error = e
                self.logger.warning(f"Stage {self.name} failed: {e}")
                
                if not e.is_recoverable:
                    context.add_error(
                        stage=self.stage,
                        error_type=e.__class__.__name__,
                        message=str(e),
                        is_recoverable=False,
                        details=e.details
                    )
                    context.finish_stage(self.stage)
                    return False
                
                attempts += 1
                if attempts <= self.config.retry_count:
                    self.logger.info(f"Retrying stage {self.name} after {self.config.retry_delay_seconds}s")
                    import time
                    time.sleep(self.config.retry_delay_seconds)
                    
            except Exception as e:
                last_error = e
                self.logger.error(f"Unexpected error in stage {self.name}: {e}", exc_info=True)
                attempts += 1
                if attempts <= self.config.retry_count:
                    import time
                    time.sleep(self.config.retry_delay_seconds)
        
        # All retries exhausted
        context.add_error(
            stage=self.stage,
            error_type=last_error.__class__.__name__ if last_error else "UnknownError",
            message=str(last_error) if last_error else "Stage failed with unknown error",
            is_recoverable=self.config.fail_soft,
            details={"attempts": attempts}
        )
        context.finish_stage(self.stage)
        
        if not self.config.fail_soft:
            context.should_abort = True
            context.abort_reason = f"Stage {self.name} failed after {attempts} attempts"
        
        return False


# =============================================================================
# Concrete Stage Executors
# =============================================================================

class VisionAnalysisStage(PipelineStageExecutor):
    """
    Vision Analysis Stage Executor.
    
    Analyzes the input image to detect pharmaceutical content
    and identify regions of interest.
    """
    
    def __init__(
        self,
        analyzer,  # VisionAnalyzerPort
        config: Optional[StageConfig] = None
    ):
        super().__init__(config)
        self.analyzer = analyzer
    
    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.VISION_ANALYSIS
    
    @property
    def name(self) -> str:
        return "Vision Analysis"
    
    def can_execute(self, context: PipelineContext) -> bool:
        return super().can_execute(context) and context.image is not None
    
    def execute(self, context: PipelineContext) -> None:
        from ...domain.exceptions import NoPharmaceuticalContentError
        
        result = self.analyzer.analyze(
            image=context.image,
            options=context.options.get("vision", {})
        )
        
        context.vision_result = result
        
        if not result.is_pharmaceutical_image:
            raise NoPharmaceuticalContentError()
        
        if result.image_quality_score and result.image_quality_score.value < 0.3:
            context.add_warning(
                "Image quality is low, which may affect accuracy of results."
            )


class TextExtractionStage(PipelineStageExecutor):
    """
    Text Extraction (OCR) Stage Executor.
    
    Extracts text from the image using OCR.
    """
    
    def __init__(
        self,
        extractor,  # TextExtractorPort
        config: Optional[StageConfig] = None
    ):
        super().__init__(config)
        self.extractor = extractor
    
    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.TEXT_EXTRACTION
    
    @property
    def name(self) -> str:
        return "Text Extraction"
    
    def can_execute(self, context: PipelineContext) -> bool:
        return super().can_execute(context) and context.image is not None
    
    def execute(self, context: PipelineContext) -> None:
        from ...domain.exceptions import NoTextFoundError
        
        # Get regions from vision analysis if available
        regions = None
        if context.vision_result:
            regions = context.vision_result.text_regions
        
        result = self.extractor.extract(
            image=context.image,
            regions=regions,
            options=context.options.get("ocr", {})
        )
        
        context.text_result = result
        
        if not result.has_text:
            raise NoTextFoundError()
        
        if result.overall_confidence.value < 0.5:
            context.add_warning(
                "Text recognition confidence is low, results may be inaccurate."
            )


class EntityExtractionStage(PipelineStageExecutor):
    """
    Entity Extraction Stage Executor.
    
    Extracts pharmaceutical entities from the OCR text.
    """
    
    def __init__(
        self,
        extractor,  # EntityExtractorPort
        config: Optional[StageConfig] = None
    ):
        super().__init__(config)
        self.extractor = extractor
    
    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.ENTITY_EXTRACTION
    
    @property
    def name(self) -> str:
        return "Entity Extraction"
    
    def can_execute(self, context: PipelineContext) -> bool:
        return super().can_execute(context) and context.has_text_result
    
    def execute(self, context: PipelineContext) -> None:
        from ...domain.exceptions import DrugNameNotFoundError
        from ...domain.entities.drug_info import DrugInfo
        from ...domain.value_objects.dosage_info import DosageForm, DosageInfo
        
        result = self.extractor.extract(
            text=context.extracted_text,
            options=context.options.get("entity", {})
        )
        
        context.entity_result = result
        
        if not result.has_drug_name:
            raise DrugNameNotFoundError(extracted_text=context.extracted_text)
        
        # Construct DrugInfo from extracted entities
        dosage_info = None
        if result.dosage_form:
            dosage_info = DosageInfo(
                form=DosageForm.from_string(result.dosage_form),
                strength=result.strength
            )
        
        context.drug_info = DrugInfo(
            drug_name=result.drug_name,
            active_ingredients=result.active_ingredients,
            dosage_info=dosage_info,
            manufacturer=result.manufacturer,
            confidence=result.overall_confidence,
            source_text=context.extracted_text
        )


class KnowledgeRetrievalStage(PipelineStageExecutor):
    """
    Knowledge Retrieval (RAG) Stage Executor.
    
    Retrieves relevant pharmaceutical knowledge.
    """
    
    def __init__(
        self,
        retriever,  # KnowledgeRetrieverPort
        config: Optional[StageConfig] = None
    ):
        super().__init__(config)
        self.retriever = retriever
    
    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.KNOWLEDGE_RETRIEVAL
    
    @property
    def name(self) -> str:
        return "Knowledge Retrieval"
    
    def can_execute(self, context: PipelineContext) -> bool:
        return super().can_execute(context) and context.has_entity_result
    
    def execute(self, context: PipelineContext) -> None:
        result = self.retriever.retrieve(
            entities=context.entity_result,
            options=context.options.get("rag", {})
        )
        
        context.knowledge_result = result
        
        if not result.has_knowledge:
            context.add_warning(
                "Limited information available for this drug in our knowledge base."
            )


class ResponseGenerationStage(PipelineStageExecutor):
    """
    Response Generation (LLM) Stage Executor.
    
    Generates user-friendly response using LLM.
    """
    
    def __init__(
        self,
        generator,  # ResponseGeneratorPort
        config: Optional[StageConfig] = None
    ):
        super().__init__(config)
        self.generator = generator
    
    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.RESPONSE_GENERATION
    
    @property
    def name(self) -> str:
        return "Response Generation"
    
    def can_execute(self, context: PipelineContext) -> bool:
        return (
            super().can_execute(context) and
            context.drug_info is not None
        )
    
    def execute(self, context: PipelineContext) -> None:
        from ...domain.exceptions import UnsafeResponseError
        
        # Use default knowledge result if retrieval failed
        knowledge = context.knowledge_result
        if knowledge is None:
            from ...domain.entities.extraction_result import KnowledgeRetrievalResult
            knowledge = KnowledgeRetrievalResult()
        
        response = self.generator.generate(
            drug_info=context.drug_info,
            knowledge=knowledge,
            options=context.options.get("llm", {})
        )
        
        # Validate response safety
        if not self.generator.validate_response(response):
            raise UnsafeResponseError(
                violations=["Response failed safety validation"],
                message="Generated response contains potentially unsafe content"
            )
        
        context.generated_response = response
