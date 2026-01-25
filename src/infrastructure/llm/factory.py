"""
LLM Factory

Factory for creating LLM response generator instances.
Supports both cloud (OpenAI) and local (Ollama) models.
"""

from typing import Optional, Dict, Any
from enum import Enum

from ...domain.ports.response_generator import ResponseGeneratorPort
from .openai_generator import OpenAIResponseGenerator, DummyResponseGenerator


class LLMType(Enum):
    """Available LLM implementations."""
    
    OPENAI = "openai"
    OPENAI_GPT4 = "openai_gpt4"
    OPENAI_GPT35 = "openai_gpt35"
    OLLAMA = "ollama"
    OLLAMA_QWEN3 = "ollama_qwen3"
    OLLAMA_GEMMA = "ollama_gemma"
    DUMMY = "dummy"


class LLMFactory:
    """
    Factory for creating LLM response generator instances.
    
    Usage:
        # Cloud LLM (OpenAI)
        generator = LLMFactory.create(
            LLMType.OPENAI_GPT4,
            api_key="your-api-key"
        )
        
        # Local LLM (Ollama)
        generator = LLMFactory.create(
            LLMType.OLLAMA,
            model="qwen3:4b"
        )
    """
    
    @staticmethod
    def create(
        llm_type: LLMType,
        **kwargs
    ) -> ResponseGeneratorPort:
        """
        Create an LLM response generator instance.
        
        Args:
            llm_type: Type of LLM to create
            **kwargs: Configuration options
                For OpenAI:
                - api_key: API key for the service
                For Ollama:
                - base_url: Ollama API URL (default: http://localhost:11434)
                - model: Model name (e.g., "qwen3:4b")
                Common:
                - temperature: Response temperature
                - max_tokens: Maximum response length
                - language: Response language (tr/en)
                
        Returns:
            ResponseGeneratorPort implementation
        """
        if llm_type == LLMType.OPENAI:
            return OpenAIResponseGenerator(
                api_key=kwargs.get("api_key"),
                model=kwargs.get("model", "gpt-4"),
                temperature=kwargs.get("temperature", 0.3),
                max_tokens=kwargs.get("max_tokens", 1000),
                language=kwargs.get("language", "tr")
            )
        
        elif llm_type == LLMType.OPENAI_GPT4:
            return OpenAIResponseGenerator(
                api_key=kwargs.get("api_key"),
                model="gpt-4",
                temperature=kwargs.get("temperature", 0.3),
                max_tokens=kwargs.get("max_tokens", 1000),
                language=kwargs.get("language", "tr")
            )
        
        elif llm_type == LLMType.OPENAI_GPT35:
            return OpenAIResponseGenerator(
                api_key=kwargs.get("api_key"),
                model="gpt-3.5-turbo",
                temperature=kwargs.get("temperature", 0.3),
                max_tokens=kwargs.get("max_tokens", 1000),
                language=kwargs.get("language", "tr")
            )
        
        elif llm_type in (LLMType.OLLAMA, LLMType.OLLAMA_QWEN3, LLMType.OLLAMA_GEMMA):
            # Import here to avoid circular imports
            from .ollama_llm import OllamaResponseGenerator
            
            # Default models for specific types
            default_models = {
                LLMType.OLLAMA: "qwen3:4b",
                LLMType.OLLAMA_QWEN3: "qwen3:4b",
                LLMType.OLLAMA_GEMMA: "gemma3:4b",
            }
            
            return OllamaResponseGenerator(
                base_url=kwargs.get("base_url", "http://localhost:11434"),
                model=kwargs.get("model", default_models.get(llm_type, "qwen3:4b")),
                temperature=kwargs.get("temperature", 0.3),
                max_tokens=kwargs.get("max_tokens", 1000),
                language=kwargs.get("language", "tr"),
                timeout=kwargs.get("timeout", 120)
            )
        
        elif llm_type == LLMType.DUMMY:
            return DummyResponseGenerator()
        
        else:
            raise ValueError(f"Unknown LLM type: {llm_type}")
    
    @staticmethod
    def create_from_config(config: Dict[str, Any]) -> ResponseGeneratorPort:
        """Create generator from configuration dictionary."""
        llm_type_str = config.get("type", "ollama")
        
        # Map string to enum
        try:
            llm_type = LLMType(llm_type_str)
        except ValueError:
            # Try with prefix for backward compatibility
            if llm_type_str in ("openai", "gpt4", "gpt-4"):
                llm_type = LLMType.OPENAI
            elif llm_type_str in ("ollama", "local"):
                llm_type = LLMType.OLLAMA
            else:
                llm_type = LLMType.DUMMY
        
        return LLMFactory.create(llm_type, **config)

