"""
Entity Extractor Factory

Factory for creating entity extractor instances.
"""

from typing import Optional, Dict, Any
from enum import Enum

from ...domain.ports.entity_extractor import EntityExtractorPort
from .hybrid_extractor import HybridEntityExtractor, DummyEntityExtractor


class EntityExtractorType(Enum):
    """Available entity extractor implementations."""
    
    HYBRID = "hybrid"
    RULE_BASED = "rule_based"  # Alias for hybrid without LLM
    DUMMY = "dummy"


class EntityExtractorFactory:
    """
    Factory for creating entity extractor instances.
    
    Usage:
        extractor = EntityExtractorFactory.create(EntityExtractorType.HYBRID)
    """
    
    @staticmethod
    def create(
        extractor_type: EntityExtractorType,
        **kwargs
    ) -> EntityExtractorPort:
        """
        Create an entity extractor instance.
        
        Args:
            extractor_type: Type of extractor to create
            **kwargs: Configuration options
            
        Returns:
            EntityExtractorPort implementation
        """
        if extractor_type in {EntityExtractorType.HYBRID, EntityExtractorType.RULE_BASED}:
            return HybridEntityExtractor(
                use_llm_refinement=kwargs.get("use_llm_refinement", False),
                confidence_boost_per_match=kwargs.get("confidence_boost", 0.1),
                min_drug_name_length=kwargs.get("min_drug_name_length", 3)
            )
        
        elif extractor_type == EntityExtractorType.DUMMY:
            return DummyEntityExtractor(
                drug_name=kwargs.get("drug_name", "Sample Drug"),
                ingredients=kwargs.get("ingredients")
            )
        
        else:
            raise ValueError(f"Unknown extractor type: {extractor_type}")
    
    @staticmethod
    def create_from_config(config: Dict[str, Any]) -> EntityExtractorPort:
        """Create extractor from configuration dictionary."""
        extractor_type = EntityExtractorType(config.get("type", "hybrid"))
        return EntityExtractorFactory.create(extractor_type, **config)
