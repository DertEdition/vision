"""
Knowledge Retriever Port

Abstract interface for RAG knowledge retrieval implementations.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List

from ..entities.extraction_result import EntityExtractionResult, KnowledgeRetrievalResult
from ..entities.drug_info import DrugInfo


class KnowledgeRetrieverPort(ABC):
    """
    Port (interface) for RAG knowledge retrieval implementations.
    
    Responsible for retrieving verified pharmaceutical knowledge:
    - Drug indications and uses
    - Contraindications and warnings
    - Side effects
    - Drug interactions
    - Dosage guidelines
    
    Knowledge sources should be:
    - Official pharmaceutical documentation
    - Regulatory agency databases
    - Verified medical references
    
    Implementations may use:
    - ChromaDB for local vector storage
    - Pinecone for cloud vector storage
    - Elasticsearch with vector search
    - Custom pharmaceutical databases
    """
    
    @abstractmethod
    def retrieve(
        self,
        entities: EntityExtractionResult,
        options: Optional[Dict[str, Any]] = None
    ) -> KnowledgeRetrievalResult:
        """
        Retrieve knowledge based on extracted entities.
        
        Args:
            entities: Extracted pharmaceutical entities
            options: Optional retrieval configuration
                - top_k: Number of chunks to retrieve
                - min_relevance: Minimum relevance score
                - sources: Specific sources to query
                
        Returns:
            KnowledgeRetrievalResult containing relevant knowledge chunks
            
        Raises:
            KnowledgeRetrievalError: If retrieval fails
        """
        pass
    
    @abstractmethod
    def retrieve_by_drug_name(
        self,
        drug_name: str,
        top_k: int = 5
    ) -> KnowledgeRetrievalResult:
        """
        Retrieve knowledge for a specific drug name.
        
        Args:
            drug_name: Name of the drug to look up
            top_k: Maximum number of chunks to retrieve
            
        Returns:
            KnowledgeRetrievalResult with drug-specific knowledge
        """
        pass
    
    @abstractmethod
    def retrieve_by_ingredient(
        self,
        ingredient: str,
        top_k: int = 5
    ) -> KnowledgeRetrievalResult:
        """
        Retrieve knowledge for an active ingredient.
        
        Args:
            ingredient: Active ingredient name
            top_k: Maximum number of chunks to retrieve
            
        Returns:
            KnowledgeRetrievalResult with ingredient-specific knowledge
        """
        pass
    
    @abstractmethod
    def index_document(
        self,
        content: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """
        Add a document to the knowledge base.
        
        Args:
            content: Document text content
            metadata: Document metadata (source, drug_name, etc.)
            
        Returns:
            True if indexing was successful
        """
        pass
    
    @property
    @abstractmethod
    def retriever_name(self) -> str:
        """Get the name of the retriever implementation."""
        pass
    
    @property
    @abstractmethod
    def knowledge_base_size(self) -> int:
        """Get the number of documents in the knowledge base."""
        pass
