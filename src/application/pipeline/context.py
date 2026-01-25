"""
Pipeline Context

Carries state through the pipeline stages.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime
import uuid

from ...domain.value_objects.image_data import ImageData
from ...domain.entities.extraction_result import (
    VisionAnalysisResult,
    TextExtractionResult,
    EntityExtractionResult,
    KnowledgeRetrievalResult,
)
from ...domain.entities.drug_info import DrugInfo
from ...domain.entities.pipeline_result import (
    PipelineResult,
    PipelineError,
    PipelineStage,
    StageStatus,
)


@dataclass
class StageMetrics:
    """
    Metrics for a single pipeline stage execution.
    
    Attributes:
        stage: The pipeline stage
        start_time: When execution started
        end_time: When execution completed
        duration_ms: Total execution time in milliseconds
        memory_usage_mb: Memory used during execution
        retries: Number of retry attempts
    """
    
    stage: PipelineStage
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_ms: float = 0.0
    memory_usage_mb: float = 0.0
    retries: int = 0
    
    def start(self) -> None:
        """Mark stage as started."""
        self.start_time = datetime.now()
    
    def finish(self) -> None:
        """Mark stage as finished and calculate duration."""
        self.end_time = datetime.now()
        if self.start_time:
            delta = self.end_time - self.start_time
            self.duration_ms = delta.total_seconds() * 1000


@dataclass
class PipelineContext:
    """
    Context object that carries state through the pipeline.
    
    This is the central data structure passed between pipeline stages.
    Each stage reads what it needs and writes its results here.
    
    Attributes:
        request_id: Unique identifier for this pipeline execution
        image: Input image data
        options: User-provided options for the pipeline
        
        # Stage results (populated as pipeline progresses)
        vision_result: Result from vision analysis
        text_result: Result from text extraction
        entity_result: Result from entity extraction
        knowledge_result: Result from knowledge retrieval
        drug_info: Constructed drug information
        generated_response: LLM-generated response
        
        # Error tracking
        errors: List of errors from all stages
        warnings: List of warnings to include in output
        
        # Metadata
        created_at: When context was created
        stage_metrics: Execution metrics per stage
    """
    
    # Input
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    image: Optional[ImageData] = None
    options: Dict[str, Any] = field(default_factory=dict)
    
    # Stage results
    vision_result: Optional[VisionAnalysisResult] = None
    text_result: Optional[TextExtractionResult] = None
    entity_result: Optional[EntityExtractionResult] = None
    knowledge_result: Optional[KnowledgeRetrievalResult] = None
    drug_info: Optional[DrugInfo] = None
    generated_response: Optional[str] = None
    
    # Error and warning tracking
    errors: List[PipelineError] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Execution metadata
    created_at: datetime = field(default_factory=datetime.now)
    stage_metrics: Dict[PipelineStage, StageMetrics] = field(default_factory=dict)
    current_stage: Optional[PipelineStage] = None
    
    # Control flags
    should_abort: bool = False
    abort_reason: Optional[str] = None
    
    def add_error(
        self,
        stage: PipelineStage,
        error_type: str,
        message: str,
        is_recoverable: bool = True,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add an error to the context.
        
        Args:
            stage: Pipeline stage where error occurred
            error_type: Type/category of error
            message: Human-readable error message
            is_recoverable: Whether pipeline can continue
            details: Additional error details
        """
        error = PipelineError(
            stage=stage,
            error_type=error_type,
            message=message,
            is_recoverable=is_recoverable,
            details=details
        )
        self.errors.append(error)
        
        # Abort on non-recoverable errors
        if not is_recoverable:
            self.should_abort = True
            self.abort_reason = message
    
    def add_warning(self, warning: str) -> None:
        """Add a warning to include in the final output."""
        if warning not in self.warnings:
            self.warnings.append(warning)
    
    def start_stage(self, stage: PipelineStage) -> None:
        """Mark a stage as started."""
        self.current_stage = stage
        self.stage_metrics[stage] = StageMetrics(stage=stage)
        self.stage_metrics[stage].start()
    
    def finish_stage(self, stage: PipelineStage) -> None:
        """Mark a stage as finished."""
        if stage in self.stage_metrics:
            self.stage_metrics[stage].finish()
    
    def get_stage_duration(self, stage: PipelineStage) -> float:
        """Get execution time for a stage in milliseconds."""
        if stage in self.stage_metrics:
            return self.stage_metrics[stage].duration_ms
        return 0.0
    
    @property
    def total_duration_ms(self) -> float:
        """Get total pipeline execution time in milliseconds."""
        return sum(m.duration_ms for m in self.stage_metrics.values())
    
    @property
    def has_errors(self) -> bool:
        """Check if any errors occurred."""
        return len(self.errors) > 0
    
    @property
    def has_critical_errors(self) -> bool:
        """Check if any non-recoverable errors occurred."""
        return any(not e.is_recoverable for e in self.errors)
    
    @property
    def has_vision_result(self) -> bool:
        """Check if vision analysis completed."""
        return self.vision_result is not None
    
    @property
    def has_text_result(self) -> bool:
        """Check if text extraction completed."""
        return self.text_result is not None
    
    @property
    def has_entity_result(self) -> bool:
        """Check if entity extraction completed."""
        return self.entity_result is not None
    
    @property
    def has_knowledge_result(self) -> bool:
        """Check if knowledge retrieval completed."""
        return self.knowledge_result is not None
    
    @property
    def extracted_text(self) -> str:
        """Get the full extracted text, if available."""
        if self.text_result:
            return self.text_result.full_text
        return ""
    
    def to_pipeline_result(self) -> PipelineResult:
        """
        Convert context to a PipelineResult.
        
        Returns:
            PipelineResult containing all gathered information
        """
        from ...domain.value_objects.confidence_score import ConfidenceScore
        
        # Calculate overall confidence
        confidences = []
        if self.vision_result:
            confidences.append(self.vision_result.overall_confidence.value)
        if self.text_result:
            confidences.append(self.text_result.overall_confidence.value)
        if self.entity_result:
            confidences.append(self.entity_result.overall_confidence.value)
        
        overall_confidence = ConfidenceScore(
            value=sum(confidences) / len(confidences) if confidences else 0.0,
            source="pipeline"
        )
        
        result = PipelineResult(
            drug_info=self.drug_info,
            explanation=self.generated_response or "",
            warnings=self.warnings.copy(),
            vision_result=self.vision_result,
            text_result=self.text_result,
            entity_result=self.entity_result,
            knowledge_result=self.knowledge_result,
            errors=self.errors.copy(),
            overall_confidence=overall_confidence,
            request_id=self.request_id,
            created_at=self.created_at,
            total_processing_time_ms=self.total_duration_ms,
        )
        
        # Set stage statuses
        for stage, metrics in self.stage_metrics.items():
            has_error = any(e.stage == stage for e in self.errors)
            status = StageStatus.FAILED if has_error else StageStatus.COMPLETED
            result.set_stage_status(stage, status, metrics.duration_ms)
        
        return result
    
    def __str__(self) -> str:
        stages_done = len(self.stage_metrics)
        errors = len(self.errors)
        return f"PipelineContext(id={self.request_id[:8]}..., stages={stages_done}, errors={errors})"
    
    @classmethod
    def create(
        cls,
        image: ImageData,
        options: Optional[Dict[str, Any]] = None
    ) -> "PipelineContext":
        """
        Create a new pipeline context.
        
        Args:
            image: Input image to process
            options: Optional configuration options
            
        Returns:
            Initialized PipelineContext
        """
        return cls(
            image=image,
            options=options or {}
        )
