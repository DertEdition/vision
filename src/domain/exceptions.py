"""
Domain Exceptions

Custom exceptions for the drug image understanding domain.
All exceptions are organized by pipeline stage for clear error handling.
"""

from typing import Optional, Dict, Any


class DomainException(Exception):
    """
    Base exception for all domain-level errors.
    
    Attributes:
        message: Human-readable error message
        details: Additional error details
        is_recoverable: Whether the operation can be retried
    """
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        is_recoverable: bool = True
    ):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.is_recoverable = is_recoverable
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}: {self.message}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for serialization."""
        return {
            "type": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
            "is_recoverable": self.is_recoverable,
        }


# =============================================================================
# Vision Analysis Exceptions
# =============================================================================

class VisionAnalysisError(DomainException):
    """Base exception for vision analysis errors."""
    pass


class ImageLoadError(VisionAnalysisError):
    """Failed to load or decode the input image."""
    
    def __init__(self, message: str = "Failed to load image", **kwargs):
        super().__init__(message, **kwargs)


class ImageQualityError(VisionAnalysisError):
    """Image quality is too low for reliable analysis."""
    
    def __init__(
        self,
        message: str = "Image quality is insufficient for analysis",
        quality_score: Optional[float] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.quality_score = quality_score
        if quality_score:
            self.details["quality_score"] = quality_score


class NoPharmaceuticalContentError(VisionAnalysisError):
    """No pharmaceutical content detected in the image."""
    
    def __init__(
        self,
        message: str = "No pharmaceutical packaging detected in the image",
        **kwargs
    ):
        super().__init__(message, is_recoverable=False, **kwargs)


class ModelLoadError(VisionAnalysisError):
    """Failed to load the vision model."""
    
    def __init__(self, message: str = "Failed to load vision model", **kwargs):
        super().__init__(message, is_recoverable=False, **kwargs)


# =============================================================================
# Text Extraction (OCR) Exceptions
# =============================================================================

class TextExtractionError(DomainException):
    """Base exception for text extraction errors."""
    pass


class OCREngineError(TextExtractionError):
    """OCR engine encountered an error."""
    
    def __init__(
        self,
        message: str = "OCR engine error",
        engine_name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        if engine_name:
            self.details["engine"] = engine_name


class NoTextFoundError(TextExtractionError):
    """No text could be extracted from the image."""
    
    def __init__(
        self,
        message: str = "No readable text found in the image",
        **kwargs
    ):
        super().__init__(message, is_recoverable=False, **kwargs)


class LanguageNotSupportedError(TextExtractionError):
    """The detected language is not supported."""
    
    def __init__(
        self,
        language: str,
        supported_languages: Optional[list] = None,
        **kwargs
    ):
        message = f"Language '{language}' is not supported"
        super().__init__(message, is_recoverable=False, **kwargs)
        self.details["language"] = language
        if supported_languages:
            self.details["supported_languages"] = supported_languages


# =============================================================================
# Entity Extraction Exceptions
# =============================================================================

class EntityExtractionError(DomainException):
    """Base exception for entity extraction errors."""
    pass


class DrugNameNotFoundError(EntityExtractionError):
    """Could not identify a drug name from the text."""
    
    def __init__(
        self,
        message: str = "Could not identify drug name from extracted text",
        extracted_text: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        if extracted_text:
            self.details["extracted_text_preview"] = extracted_text[:200]


class AmbiguousDrugNameError(EntityExtractionError):
    """Multiple possible drug names detected, cannot determine primary."""
    
    def __init__(
        self,
        candidates: list,
        message: str = "Multiple possible drug names detected",
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.candidates = candidates
        self.details["candidates"] = candidates


class InvalidEntityError(EntityExtractionError):
    """Extracted entity failed validation."""
    
    def __init__(
        self,
        entity_type: str,
        value: str,
        reason: str,
        **kwargs
    ):
        message = f"Invalid {entity_type}: {value} - {reason}"
        super().__init__(message, **kwargs)
        self.details["entity_type"] = entity_type
        self.details["value"] = value
        self.details["reason"] = reason


# =============================================================================
# Knowledge Retrieval (RAG) Exceptions
# =============================================================================

class KnowledgeRetrievalError(DomainException):
    """Base exception for knowledge retrieval errors."""
    pass


class KnowledgeBaseConnectionError(KnowledgeRetrievalError):
    """Failed to connect to the knowledge base."""
    
    def __init__(
        self,
        message: str = "Failed to connect to knowledge base",
        **kwargs
    ):
        super().__init__(message, **kwargs)


class NoRelevantKnowledgeError(KnowledgeRetrievalError):
    """No relevant knowledge found for the drug."""
    
    def __init__(
        self,
        drug_name: str,
        message: Optional[str] = None,
        **kwargs
    ):
        message = message or f"No relevant knowledge found for drug: {drug_name}"
        super().__init__(message, **kwargs)
        self.details["drug_name"] = drug_name


class KnowledgeBaseEmptyError(KnowledgeRetrievalError):
    """Knowledge base is empty or not initialized."""
    
    def __init__(
        self,
        message: str = "Knowledge base is empty or not initialized",
        **kwargs
    ):
        super().__init__(message, is_recoverable=False, **kwargs)


# =============================================================================
# Response Generation (LLM) Exceptions
# =============================================================================

class ResponseGenerationError(DomainException):
    """Base exception for response generation errors."""
    pass


class LLMConnectionError(ResponseGenerationError):
    """Failed to connect to the LLM service."""
    
    def __init__(
        self,
        message: str = "Failed to connect to LLM service",
        provider: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        if provider:
            self.details["provider"] = provider


class LLMRateLimitError(ResponseGenerationError):
    """LLM rate limit exceeded."""
    
    def __init__(
        self,
        retry_after: Optional[int] = None,
        message: str = "LLM rate limit exceeded",
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after
        if retry_after:
            self.details["retry_after_seconds"] = retry_after


class UnsafeResponseError(ResponseGenerationError):
    """Generated response failed safety validation."""
    
    def __init__(
        self,
        violations: list,
        message: str = "Generated response contains unsafe content",
        **kwargs
    ):
        super().__init__(message, is_recoverable=True, **kwargs)
        self.violations = violations
        self.details["violations"] = violations


class ContextTooLongError(ResponseGenerationError):
    """Input context exceeds LLM maximum length."""
    
    def __init__(
        self,
        context_length: int,
        max_length: int,
        message: Optional[str] = None,
        **kwargs
    ):
        message = message or f"Context length ({context_length}) exceeds maximum ({max_length})"
        super().__init__(message, **kwargs)
        self.details["context_length"] = context_length
        self.details["max_length"] = max_length


# =============================================================================
# Pipeline Exceptions
# =============================================================================

class PipelineError(DomainException):
    """Base exception for pipeline-level errors."""
    pass


class PipelineConfigurationError(PipelineError):
    """Pipeline is not properly configured."""
    
    def __init__(
        self,
        message: str = "Pipeline is not properly configured",
        missing_components: Optional[list] = None,
        **kwargs
    ):
        super().__init__(message, is_recoverable=False, **kwargs)
        if missing_components:
            self.details["missing_components"] = missing_components


class PipelineTimeoutError(PipelineError):
    """Pipeline execution timed out."""
    
    def __init__(
        self,
        timeout_seconds: float,
        stage: Optional[str] = None,
        message: Optional[str] = None,
        **kwargs
    ):
        message = message or f"Pipeline timed out after {timeout_seconds} seconds"
        super().__init__(message, **kwargs)
        self.details["timeout_seconds"] = timeout_seconds
        if stage:
            self.details["stage"] = stage


class StageExecutionError(PipelineError):
    """A pipeline stage failed to execute."""
    
    def __init__(
        self,
        stage_name: str,
        original_error: Exception,
        **kwargs
    ):
        message = f"Stage '{stage_name}' failed: {str(original_error)}"
        super().__init__(message, **kwargs)
        self.stage_name = stage_name
        self.original_error = original_error
        self.details["stage"] = stage_name
        self.details["original_error"] = str(original_error)


# =============================================================================
# Validation Exceptions
# =============================================================================

class ValidationError(DomainException):
    """Base exception for validation errors."""
    pass


class InvalidImageError(ValidationError):
    """Input image is invalid or corrupted."""
    
    def __init__(
        self,
        message: str = "Invalid or corrupted image",
        **kwargs
    ):
        super().__init__(message, is_recoverable=False, **kwargs)


class InvalidInputError(ValidationError):
    """Invalid input provided to a function or method."""
    
    def __init__(
        self,
        field: str,
        reason: str,
        **kwargs
    ):
        message = f"Invalid input for '{field}': {reason}"
        super().__init__(message, is_recoverable=False, **kwargs)
        self.details["field"] = field
        self.details["reason"] = reason
