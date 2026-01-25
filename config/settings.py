"""
Application Configuration

Settings and configuration management for the drug image pipeline.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path
import os


@dataclass
class VisionConfig:
    """Vision analysis configuration."""
    
    type: str = "yolo"
    model_path: Optional[str] = None
    confidence_threshold: float = 0.25
    device: str = "cuda"  # cpu, cuda, mps


@dataclass
class OCRConfig:
    """OCR configuration."""
    
    type: str = "tesseract"  # paddle, tesseract
    language: str = "tur+eng"  # Tesseract language codes
    use_gpu: bool = False
    fallback_enabled: bool = True

 
@dataclass
class EntityExtractionConfig:
    """Entity extraction configuration."""
    
    type: str = "hybrid"
    use_llm_refinement: bool = False
    min_drug_name_length: int = 3


@dataclass
class RAGConfig:
    """RAG configuration."""
    
    type: str = "chroma"
    persist_directory: Optional[str] = "./data/chroma_db"  # Set default path
    collection_name: str = "drug_knowledge"
    top_k: int = 5
    min_relevance: float = 0.5


@dataclass
class LLMConfig:
    """LLM configuration."""
    
    type: str = "ollama"  # openai, ollama, dummy
    model: str = "gemma3:4b"  # Ollama model or OpenAI model
    api_key: Optional[str] = None  # Only for OpenAI
    ollama_base_url: str = "http://localhost:11434"  # Ollama API URL
    temperature: float = 0.3
    max_tokens: int = 1000
    language: str = "tr"
    timeout: int = 300  # Request timeout in seconds (5 minutes for large models)


@dataclass
class PipelineConfig:
    """Pipeline orchestration configuration."""
    
    timeout_seconds: float = 120.0
    fail_fast: bool = False
    stage_retry_count: int = 2
    stage_retry_delay: float = 1.0


@dataclass
class SafetyConfig:
    """Safety configuration."""
    
    confidence_threshold: float = 0.5
    strict_mode: bool = True
    inject_disclaimer: bool = True
    disclaimer_language: str = "tr"


@dataclass
class LoggingConfig:
    """Logging configuration."""
    
    level: str = "INFO"
    log_file: Optional[str] = None
    format: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


@dataclass
class AppConfig:
    """
    Main application configuration.
    
    Aggregates all component configurations.
    """
    
    vision: VisionConfig = field(default_factory=VisionConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    entity_extraction: EntityExtractionConfig = field(default_factory=EntityExtractionConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Data paths
    data_dir: str = "./data"
    knowledge_base_dir: str = "./data/drug_knowledge_base"
    
    @classmethod
    def from_env(cls) -> "AppConfig":
        """
        Create configuration from environment variables.
        
        Environment variables:
            DRUG_PIPELINE_VISION_DEVICE: Vision device (cpu/cuda)
            DRUG_PIPELINE_OCR_TYPE: OCR type (paddle/tesseract)
            DRUG_PIPELINE_OCR_LANGUAGE: OCR language
            DRUG_PIPELINE_LLM_API_KEY: OpenAI API key
            DRUG_PIPELINE_LLM_MODEL: LLM model name
            DRUG_PIPELINE_DATA_DIR: Data directory path
            DRUG_PIPELINE_LOG_LEVEL: Logging level
        """
        config = cls()
        
        # Vision
        if device := os.getenv("DRUG_PIPELINE_VISION_DEVICE"):
            config.vision.device = device
        
        # OCR
        if ocr_type := os.getenv("DRUG_PIPELINE_OCR_TYPE"):
            config.ocr.type = ocr_type
        if ocr_lang := os.getenv("DRUG_PIPELINE_OCR_LANGUAGE"):
            config.ocr.language = ocr_lang
        
        # LLM
        if api_key := os.getenv("DRUG_PIPELINE_LLM_API_KEY"):
            config.llm.api_key = api_key
        elif api_key := os.getenv("OPENAI_API_KEY"):
            config.llm.api_key = api_key
        if model := os.getenv("DRUG_PIPELINE_LLM_MODEL"):
            config.llm.model = model
        
        # Data
        if data_dir := os.getenv("DRUG_PIPELINE_DATA_DIR"):
            config.data_dir = data_dir
            config.knowledge_base_dir = str(Path(data_dir) / "drug_knowledge_base")
        
        # Logging
        if log_level := os.getenv("DRUG_PIPELINE_LOG_LEVEL"):
            config.logging.level = log_level
        
        return config
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppConfig":
        """Create configuration from dictionary."""
        config = cls()
        
        if "vision" in data:
            for key, value in data["vision"].items():
                if hasattr(config.vision, key):
                    setattr(config.vision, key, value)
        
        if "ocr" in data:
            for key, value in data["ocr"].items():
                if hasattr(config.ocr, key):
                    setattr(config.ocr, key, value)
        
        if "entity_extraction" in data:
            for key, value in data["entity_extraction"].items():
                if hasattr(config.entity_extraction, key):
                    setattr(config.entity_extraction, key, value)
        
        if "rag" in data:
            for key, value in data["rag"].items():
                if hasattr(config.rag, key):
                    setattr(config.rag, key, value)
        
        if "llm" in data:
            for key, value in data["llm"].items():
                if hasattr(config.llm, key):
                    setattr(config.llm, key, value)
        
        if "pipeline" in data:
            for key, value in data["pipeline"].items():
                if hasattr(config.pipeline, key):
                    setattr(config.pipeline, key, value)
        
        if "safety" in data:
            for key, value in data["safety"].items():
                if hasattr(config.safety, key):
                    setattr(config.safety, key, value)
        
        if "data_dir" in data:
            config.data_dir = data["data_dir"]
        if "knowledge_base_dir" in data:
            config.knowledge_base_dir = data["knowledge_base_dir"]
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "vision": {
                "type": self.vision.type,
                "model_path": self.vision.model_path,
                "confidence_threshold": self.vision.confidence_threshold,
                "device": self.vision.device,
            },
            "ocr": {
                "type": self.ocr.type,
                "language": self.ocr.language,
                "use_gpu": self.ocr.use_gpu,
                "fallback_enabled": self.ocr.fallback_enabled,
            },
            "entity_extraction": {
                "type": self.entity_extraction.type,
                "use_llm_refinement": self.entity_extraction.use_llm_refinement,
            },
            "rag": {
                "type": self.rag.type,
                "persist_directory": self.rag.persist_directory,
                "collection_name": self.rag.collection_name,
                "top_k": self.rag.top_k,
            },
            "llm": {
                "type": self.llm.type,
                "model": self.llm.model,
                "temperature": self.llm.temperature,
                "max_tokens": self.llm.max_tokens,
                "language": self.llm.language,
            },
            "pipeline": {
                "timeout_seconds": self.pipeline.timeout_seconds,
                "fail_fast": self.pipeline.fail_fast,
            },
            "safety": {
                "confidence_threshold": self.safety.confidence_threshold,
                "strict_mode": self.safety.strict_mode,
            },
            "data_dir": self.data_dir,
            "knowledge_base_dir": self.knowledge_base_dir,
        }


def get_default_config() -> AppConfig:
    """Get default application configuration."""
    return AppConfig.from_env()
