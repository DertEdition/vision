"""
Response Generator Port

Abstract interface for LLM response generation implementations.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

from ..entities.drug_info import DrugInfo
from ..entities.extraction_result import KnowledgeRetrievalResult


class ResponseGeneratorPort(ABC):
    """
    Port (interface) for LLM response generation implementations.
    
    Responsible for generating user-friendly explanations:
    - Based ONLY on retrieved knowledge (no hallucination)
    - Clear, simple language for general users
    - Appropriate safety warnings
    - Mandatory disclaimers
    
    ABSOLUTE PROHIBITIONS:
    - No diagnosis
    - No treatment recommendations
    - No dosage prescriptions
    - No personalized medical advice
    - No hallucinated or unverified facts
    
    Implementations may use:
    - OpenAI GPT-4
    - Anthropic Claude
    - Google Gemini
    - Local LLMs (Llama, Mistral)
    """
    
    @abstractmethod
    def generate(
        self,
        drug_info: DrugInfo,
        knowledge: KnowledgeRetrievalResult,
        options: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate user-friendly explanation about a drug.
        
        Args:
            drug_info: Identified drug information
            knowledge: Retrieved pharmaceutical knowledge
            options: Optional generation configuration
                - max_length: Maximum response length
                - language: Output language (default: Turkish)
                - include_warnings: Whether to include warnings
                - formality: Response formality level
                
        Returns:
            Generated explanation text
            
        Raises:
            ResponseGenerationError: If generation fails
        """
        pass
    
    @abstractmethod
    def generate_with_template(
        self,
        drug_info: DrugInfo,
        knowledge: KnowledgeRetrievalResult,
        template_name: str
    ) -> str:
        """
        Generate response using a specific template.
        
        Args:
            drug_info: Identified drug information
            knowledge: Retrieved pharmaceutical knowledge
            template_name: Name of the template to use
            
        Returns:
            Generated explanation text
        """
        pass
    
    @abstractmethod
    def validate_response(self, response: str) -> bool:
        """
        Validate that generated response meets safety requirements.
        
        Checks for:
        - Absence of diagnostic language
        - Absence of treatment recommendations
        - Absence of dosage prescriptions
        - Presence of appropriate disclaimers
        
        Args:
            response: Generated response to validate
            
        Returns:
            True if response passes safety validation
        """
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the name of the LLM model."""
        pass
    
    @property
    @abstractmethod
    def max_context_length(self) -> int:
        """Get the maximum context length supported."""
        pass
    
    @property
    def available_templates(self) -> list:
        """Get list of available response templates."""
        return [
            "default",
            "detailed",
            "brief",
            "patient_friendly",
        ]
