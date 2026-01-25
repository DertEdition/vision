"""
ChromaDB Knowledge Retriever

RAG implementation using ChromaDB for pharmaceutical knowledge retrieval.
"""

from typing import Optional, Dict, Any, List
import logging
import time
from pathlib import Path

from ...domain.ports.knowledge_retriever import KnowledgeRetrieverPort
from ...domain.entities.extraction_result import (
    EntityExtractionResult,
    KnowledgeRetrievalResult,
    KnowledgeChunk,
)
from ...domain.entities.drug_info import DrugInfo
from ...domain.exceptions import (
    KnowledgeRetrievalError,
    KnowledgeBaseConnectionError,
    NoRelevantKnowledgeError,
    KnowledgeBaseEmptyError,
)


logger = logging.getLogger(__name__)


class ChromaKnowledgeRetriever(KnowledgeRetrieverPort):
    """
    Knowledge retriever implementation using ChromaDB.
    
    ChromaDB provides a simple, local vector database for
    semantic search over pharmaceutical knowledge.
    
    Attributes:
        persist_directory: Directory for persistent storage
        collection_name: Name of the ChromaDB collection
        embedding_model: Model for generating embeddings
    """
    
    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: str = "drug_knowledge",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize ChromaDB retriever.
        
        Args:
            persist_directory: Directory for persistent storage
            collection_name: ChromaDB collection name
            embedding_model: Sentence transformer model for embeddings
        """
        self._persist_directory = persist_directory
        self._collection_name = collection_name
        self._embedding_model = embedding_model
        self._client = None
        self._collection = None
        self._initialized = False
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def _initialize(self) -> None:
        """Lazy initialization of ChromaDB client."""
        if self._initialized:
            return
        
        try:
            import chromadb
            
            # Initialize client with new API
            if self._persist_directory:
                self.logger.info(f"Initializing ChromaDB with persist_directory={self._persist_directory}")
                # Use PersistentClient for new API
                self._client = chromadb.PersistentClient(path=self._persist_directory)
            else:
                self.logger.info("Initializing ChromaDB in-memory")
                # Use EphemeralClient for in-memory
                self._client = chromadb.EphemeralClient()
            
            # Get or create collection
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"description": "Pharmaceutical drug knowledge base"}
            )
            
            self._initialized = True
            self.logger.info(f"ChromaDB initialized with collection '{self._collection_name}'")
            
        except ImportError:
            raise KnowledgeBaseConnectionError(
                "chromadb not installed. Install with: pip install chromadb"
            )
        except Exception as e:
            raise KnowledgeBaseConnectionError(f"Failed to initialize ChromaDB: {e}")
    
    def retrieve(
        self,
        entities: EntityExtractionResult,
        options: Optional[Dict[str, Any]] = None
    ) -> KnowledgeRetrievalResult:
        """
        Retrieve knowledge based on extracted entities.
        
        Args:
            entities: Extracted pharmaceutical entities
            options: Optional configuration
                - top_k: Number of results (default: 5)
                - min_relevance: Minimum relevance score
                
        Returns:
            KnowledgeRetrievalResult with relevant chunks
        """
        start_time = time.time()
        options = options or {}
        
        # Initialize if needed
        self._initialize()
        
        top_k = options.get("top_k", 5)
        min_relevance = options.get("min_relevance", 0.5)
        
        # Build query from entities
        query_parts = []
        
        if entities.drug_name:
            query_parts.append(entities.drug_name)
        
        for ingredient in entities.active_ingredients:
            query_parts.append(ingredient)
        
        if entities.dosage_form:
            query_parts.append(entities.dosage_form)
        
        if not query_parts:
            return KnowledgeRetrievalResult(
                chunks=[],
                query_used=None,
                processing_time_ms=(time.time() - start_time) * 1000
            )
        
        query = " ".join(query_parts)
        
        # Query ChromaDB
        chunks = self._query_collection(query, top_k, min_relevance)
        
        processing_time = (time.time() - start_time) * 1000
        
        return KnowledgeRetrievalResult(
            chunks=chunks,
            query_used=query,
            total_chunks_searched=self.knowledge_base_size,
            processing_time_ms=processing_time
        )
    
    def retrieve_by_drug_name(
        self,
        drug_name: str,
        top_k: int = 5
    ) -> KnowledgeRetrievalResult:
        """Retrieve knowledge for a specific drug name."""
        start_time = time.time()
        
        self._initialize()
        
        chunks = self._query_collection(drug_name, top_k)
        
        return KnowledgeRetrievalResult(
            chunks=chunks,
            query_used=drug_name,
            total_chunks_searched=self.knowledge_base_size,
            processing_time_ms=(time.time() - start_time) * 1000
        )
    
    def retrieve_by_ingredient(
        self,
        ingredient: str,
        top_k: int = 5
    ) -> KnowledgeRetrievalResult:
        """Retrieve knowledge for an active ingredient."""
        start_time = time.time()
        
        self._initialize()
        
        # Query with ingredient context
        query = f"active ingredient {ingredient} uses effects"
        chunks = self._query_collection(query, top_k)
        
        return KnowledgeRetrievalResult(
            chunks=chunks,
            query_used=query,
            total_chunks_searched=self.knowledge_base_size,
            processing_time_ms=(time.time() - start_time) * 1000
        )
    
    def _query_collection(
        self,
        query: str,
        top_k: int,
        min_relevance: float = 0.0
    ) -> List[KnowledgeChunk]:
        """Query ChromaDB collection."""
        if self._collection is None:
            return []
        
        try:
            # Check if collection is empty
            if self._collection.count() == 0:
                self.logger.warning("Knowledge base is empty")
                return []
            
            results = self._collection.query(
                query_texts=[query],
                n_results=min(top_k, self._collection.count())
            )
            
            chunks = []
            
            if results and results.get("documents"):
                documents = results["documents"][0]
                distances = results.get("distances", [[]])[0]
                metadatas = results.get("metadatas", [[]])[0]
                
                for i, doc in enumerate(documents):
                    # Convert distance to relevance score
                    # ChromaDB uses L2 distance, so smaller is better
                    distance = distances[i] if i < len(distances) else 1.0
                    relevance = max(0, 1 - (distance / 2))  # Normalize to 0-1
                    
                    if relevance >= min_relevance:
                        metadata = metadatas[i] if i < len(metadatas) else {}
                        
                        chunks.append(KnowledgeChunk(
                            content=doc,
                            source=metadata.get("source", "unknown"),
                            relevance_score=relevance,
                            metadata=metadata
                        ))
            
            return chunks
            
        except Exception as e:
            self.logger.error(f"ChromaDB query failed: {e}")
            return []
    
    def index_document(
        self,
        content: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """
        Add a document to the knowledge base.
        
        Args:
            content: Document text content
            metadata: Document metadata (e.g., source, drug_name)
            
        Returns:
            True if successful
        """
        self._initialize()
        
        try:
            import uuid
            
            doc_id = metadata.get("id", str(uuid.uuid4()))
            
            self._collection.add(
                documents=[content],
                metadatas=[metadata],
                ids=[doc_id]
            )
            
            self.logger.info(f"Indexed document: {doc_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to index document: {e}")
            return False
    
    def index_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> int:
        """
        Add multiple documents to the knowledge base.
        
        Args:
            documents: List of dicts with 'content' and 'metadata'
            
        Returns:
            Number of documents successfully indexed
        """
        self._initialize()
        
        count = 0
        for doc in documents:
            if self.index_document(doc.get("content", ""), doc.get("metadata", {})):
                count += 1
        
        return count
    
    def clear(self) -> None:
        """Clear all documents from the knowledge base."""
        self._initialize()
        
        if self._collection:
            self._client.delete_collection(self._collection_name)
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name
            )
            self.logger.info("Knowledge base cleared")
    
    @property
    def retriever_name(self) -> str:
        return "ChromaDB"
    
    @property
    def knowledge_base_size(self) -> int:
        """Get number of documents in the knowledge base."""
        if not self._initialized:
            self._initialize()
        
        if self._collection:
            return self._collection.count()
        return 0


class DummyKnowledgeRetriever(KnowledgeRetrieverPort):
    """
    Dummy knowledge retriever for testing.
    
    Returns preset knowledge chunks.
    """
    
    DEFAULT_KNOWLEDGE = """
    This medication is used to treat pain and reduce fever. 
    It works by blocking certain natural substances in the body that cause pain and inflammation.
    Common side effects may include nausea, stomach upset, and dizziness.
    Take this medication as directed by your doctor, usually with food.
    Do not exceed the recommended dose.
    """
    
    def __init__(self, knowledge_text: Optional[str] = None):
        self._knowledge = knowledge_text or self.DEFAULT_KNOWLEDGE
    
    def retrieve(
        self,
        entities: EntityExtractionResult,
        options: Optional[Dict[str, Any]] = None
    ) -> KnowledgeRetrievalResult:
        return KnowledgeRetrievalResult(
            chunks=[
                KnowledgeChunk(
                    content=self._knowledge,
                    source="dummy",
                    relevance_score=0.9,
                    metadata={"type": "dummy"}
                )
            ],
            query_used=entities.drug_name or "dummy",
            total_chunks_searched=1,
            processing_time_ms=1.0
        )
    
    def retrieve_by_drug_name(self, drug_name: str, top_k: int = 5) -> KnowledgeRetrievalResult:
        return KnowledgeRetrievalResult(
            chunks=[
                KnowledgeChunk(
                    content=self._knowledge,
                    source="dummy",
                    relevance_score=0.9
                )
            ],
            query_used=drug_name
        )
    
    def retrieve_by_ingredient(self, ingredient: str, top_k: int = 5) -> KnowledgeRetrievalResult:
        return self.retrieve_by_drug_name(ingredient, top_k)
    
    def index_document(self, content: str, metadata: Dict[str, Any]) -> bool:
        return True
    
    @property
    def retriever_name(self) -> str:
        return "DummyRetriever"
    
    @property
    def knowledge_base_size(self) -> int:
        return 1
