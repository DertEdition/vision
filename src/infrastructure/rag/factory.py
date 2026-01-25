"""
Knowledge Retriever Factory

Factory for creating knowledge retriever instances.
"""

from typing import Optional, Dict, Any
from enum import Enum

from ...domain.ports.knowledge_retriever import KnowledgeRetrieverPort
from .chroma_retriever import ChromaKnowledgeRetriever, DummyKnowledgeRetriever


class KnowledgeRetrieverType(Enum):
    """Available knowledge retriever implementations."""
    
    CHROMA = "chroma"
    DUMMY = "dummy"


class KnowledgeRetrieverFactory:
    """
    Factory for creating knowledge retriever instances.
    
    Usage:
        retriever = KnowledgeRetrieverFactory.create(
            KnowledgeRetrieverType.CHROMA,
            persist_directory="./data/chroma"
        )
    """
    
    @staticmethod
    def create(
        retriever_type: KnowledgeRetrieverType,
        **kwargs
    ) -> KnowledgeRetrieverPort:
        """
        Create a knowledge retriever instance.
        
        Args:
            retriever_type: Type of retriever to create
            **kwargs: Configuration options
                For CHROMA:
                - persist_directory: Directory for persistent storage
                - collection_name: ChromaDB collection name
                - embedding_model: Sentence transformer model
                
        Returns:
            KnowledgeRetrieverPort implementation
        """
        if retriever_type == KnowledgeRetrieverType.CHROMA:
            return ChromaKnowledgeRetriever(
                persist_directory=kwargs.get("persist_directory"),
                collection_name=kwargs.get("collection_name", "drug_knowledge"),
                embedding_model=kwargs.get("embedding_model", "all-MiniLM-L6-v2")
            )
        
        elif retriever_type == KnowledgeRetrieverType.DUMMY:
            return DummyKnowledgeRetriever(
                knowledge_text=kwargs.get("knowledge_text")
            )
        
        else:
            raise ValueError(f"Unknown retriever type: {retriever_type}")
    
    @staticmethod
    def create_from_config(config: Dict[str, Any]) -> KnowledgeRetrieverPort:
        """Create retriever from configuration dictionary."""
        retriever_type = KnowledgeRetrieverType(config.get("type", "chroma"))
        return KnowledgeRetrieverFactory.create(retriever_type, **config)
