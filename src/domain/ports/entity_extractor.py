"""
Entity Extractor Port

Abstract interface for pharmaceutical entity extraction implementations.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List

from ..entities.extraction_result import EntityExtractionResult, ExtractedEntity


class EntityExtractorPort(ABC):
    """
    Port (interface) for entity extraction implementations.
    
    Responsible for extracting structured pharmaceutical entities
    from raw OCR text:
    - Drug names (brand names)
    - Active ingredients (generic names)
    - Dosage forms (tablet, capsule, syrup)
    - Strength/concentration
    - Manufacturer information
    - Batch numbers, expiry dates
    
    Implementations may use:
    - Rule-based pattern matching
    - Named Entity Recognition (NER) models
    - LLM-based extraction
    - Hybrid approaches
    """
    
    @abstractmethod
    def extract(
        self,
        text: str,
        options: Optional[Dict[str, Any]] = None
    ) -> EntityExtractionResult:
        """
        Extract pharmaceutical entities from text.
        
        Args:
            text: Raw text from OCR extraction
            options: Optional extraction configuration
                - language: Text language hint
                - use_llm_refinement: Whether to use LLM for refinement
                - confidence_threshold: Minimum entity confidence
                
        Returns:
            EntityExtractionResult containing extracted entities
            
        Raises:
            EntityExtractionError: If extraction fails
        """
        pass
    
    @abstractmethod
    def extract_drug_name(self, text: str) -> Optional[ExtractedEntity]:
        """
        Extract the primary drug name from text.
        
        Args:
            text: Raw text to process
            
        Returns:
            ExtractedEntity for drug name, or None if not found
        """
        pass
    
    @abstractmethod
    def extract_active_ingredients(self, text: str) -> List[ExtractedEntity]:
        """
        Extract active ingredients from text.
        
        Args:
            text: Raw text to process
            
        Returns:
            List of ExtractedEntity for each ingredient found
        """
        pass
    
    @property
    @abstractmethod
    def extractor_name(self) -> str:
        """Get the name of the extractor implementation."""
        pass
    
    @property
    def supported_entity_types(self) -> List[str]:
        """Get list of entity types this extractor can identify."""
        return [
            "drug_name",
            "active_ingredient",
            "dosage_form",
            "strength",
            "manufacturer",
            "barcode",
            "batch_number",
            "expiry_date",
        ]
