"""
Extraction Result Entities

Domain entities representing results from each pipeline stage.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum

from ..value_objects.confidence_score import ConfidenceScore
from ..value_objects.bounding_box import BoundingBox


class DetectionClass(Enum):
    """Classes of detected pharmaceutical objects."""
    
    DRUG_BOX = "drug_box"
    BLISTER_PACK = "blister_pack"
    LEAFLET = "leaflet"
    LABEL = "label"
    BARCODE = "barcode"
    TEXT_REGION = "text_region"
    UNKNOWN = "unknown"


@dataclass
class DetectedObject:
    """
    Represents a detected object in the vision analysis stage.
    
    Attributes:
        detection_class: Type of detected object
        bounding_box: Location in the image
        confidence: Detection confidence score
        attributes: Additional detection attributes
    """
    
    detection_class: DetectionClass
    bounding_box: BoundingBox
    confidence: ConfidenceScore
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_pharmaceutical(self) -> bool:
        """Check if detection is a pharmaceutical-related object."""
        pharmaceutical_classes = {
            DetectionClass.DRUG_BOX,
            DetectionClass.BLISTER_PACK,
            DetectionClass.LEAFLET,
            DetectionClass.LABEL,
        }
        return self.detection_class in pharmaceutical_classes
    
    def __str__(self) -> str:
        return f"{self.detection_class.value} ({self.confidence})"


@dataclass
class VisionAnalysisResult:
    """
    Result from the vision analysis pipeline stage.
    
    Contains all detected objects and regions of interest
    from the pharmaceutical image analysis.
    
    Attributes:
        detected_objects: List of detected pharmaceutical objects
        image_quality_score: Overall image quality assessment
        is_pharmaceutical_image: Whether image contains pharmaceutical content
        processing_time_ms: Time taken for analysis
        raw_output: Original model output for debugging
    """
    
    detected_objects: List[DetectedObject] = field(default_factory=list)
    image_quality_score: Optional[ConfidenceScore] = None
    is_pharmaceutical_image: bool = False
    processing_time_ms: float = 0.0
    raw_output: Optional[Dict[str, Any]] = None
    
    @property
    def has_detections(self) -> bool:
        """Check if any objects were detected."""
        return len(self.detected_objects) > 0
    
    @property
    def primary_detection(self) -> Optional[DetectedObject]:
        """Get the highest confidence detection."""
        if not self.detected_objects:
            return None
        return max(self.detected_objects, key=lambda d: d.confidence.value)
    
    @property
    def text_regions(self) -> List[BoundingBox]:
        """Get bounding boxes of text regions for OCR."""
        return [
            obj.bounding_box
            for obj in self.detected_objects
            if obj.detection_class in {DetectionClass.TEXT_REGION, DetectionClass.LABEL}
        ]
    
    @property
    def overall_confidence(self) -> ConfidenceScore:
        """Calculate overall detection confidence."""
        if not self.detected_objects:
            return ConfidenceScore.zero()
        
        avg_confidence = sum(d.confidence.value for d in self.detected_objects) / len(self.detected_objects)
        return ConfidenceScore(value=avg_confidence, source="vision_analysis")
    
    def get_objects_by_class(self, detection_class: DetectionClass) -> List[DetectedObject]:
        """Filter detected objects by class."""
        return [obj for obj in self.detected_objects if obj.detection_class == detection_class]


@dataclass
class TextBlock:
    """
    Represents a block of extracted text from OCR.
    
    Attributes:
        text: The extracted text content
        bounding_box: Location in the image
        confidence: OCR confidence score
        language: Detected language code
        is_handwritten: Whether text appears handwritten
    """
    
    text: str
    bounding_box: Optional[BoundingBox] = None
    confidence: ConfidenceScore = field(default_factory=ConfidenceScore.zero)
    language: Optional[str] = None
    is_handwritten: bool = False
    
    @property
    def word_count(self) -> int:
        """Count words in the text block."""
        return len(self.text.split())
    
    @property
    def is_empty(self) -> bool:
        """Check if text block is empty or whitespace only."""
        return not self.text.strip()
    
    def __str__(self) -> str:
        preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        return f"TextBlock('{preview}', confidence={self.confidence})"


@dataclass
class TextExtractionResult:
    """
    Result from the OCR/text extraction pipeline stage.
    
    Contains all extracted text blocks from the image
    with their positions and confidence scores.
    
    Attributes:
        text_blocks: List of extracted text blocks
        full_text: Concatenated text from all blocks
        primary_language: Dominant language detected
        processing_time_ms: Time taken for extraction
        raw_output: Original OCR output for debugging
    """
    
    text_blocks: List[TextBlock] = field(default_factory=list)
    full_text: str = ""
    primary_language: Optional[str] = None
    processing_time_ms: float = 0.0
    raw_output: Optional[Dict[str, Any]] = None
    
    def __post_init__(self) -> None:
        """Build full text from blocks if not provided."""
        if not self.full_text and self.text_blocks:
            self.full_text = "\n".join(
                block.text for block in self.text_blocks if not block.is_empty
            )
    
    @property
    def has_text(self) -> bool:
        """Check if any text was extracted."""
        return bool(self.full_text.strip())
    
    @property
    def overall_confidence(self) -> ConfidenceScore:
        """Calculate overall OCR confidence."""
        if not self.text_blocks:
            return ConfidenceScore.zero()
        
        avg_confidence = sum(b.confidence.value for b in self.text_blocks) / len(self.text_blocks)
        return ConfidenceScore(value=avg_confidence, source="ocr")
    
    def get_text_in_region(self, region: BoundingBox) -> str:
        """Get text from blocks within a specific region."""
        # Simplified: just return all text for now
        # Full implementation would check bbox overlap
        return self.full_text


class EntityType(Enum):
    """Types of entities that can be extracted from drug text."""
    
    DRUG_NAME = "drug_name"
    ACTIVE_INGREDIENT = "active_ingredient"
    DOSAGE_FORM = "dosage_form"
    STRENGTH = "strength"
    MANUFACTURER = "manufacturer"
    BARCODE = "barcode"
    BATCH_NUMBER = "batch_number"
    EXPIRY_DATE = "expiry_date"
    WARNING = "warning"
    INDICATION = "indication"


@dataclass
class ExtractedEntity:
    """
    Represents an extracted named entity from drug text.
    
    Attributes:
        entity_type: Type of the entity
        value: Extracted value
        confidence: Extraction confidence
        source_span: Character span in source text
        normalized_value: Normalized/standardized value
    """
    
    entity_type: EntityType
    value: str
    confidence: ConfidenceScore = field(default_factory=ConfidenceScore.zero)
    source_span: Optional[tuple] = None  # (start, end) character positions
    normalized_value: Optional[str] = None
    
    @property
    def display_value(self) -> str:
        """Get the best value for display."""
        return self.normalized_value or self.value
    
    def __str__(self) -> str:
        return f"{self.entity_type.value}: {self.value}"


@dataclass
class EntityExtractionResult:
    """
    Result from the entity extraction pipeline stage.
    
    Contains all extracted pharmaceutical entities
    from the OCR text.
    
    Attributes:
        entities: List of extracted entities
        drug_name: Primary drug name if found
        active_ingredients: List of active ingredients
        processing_time_ms: Time taken for extraction
    """
    
    entities: List[ExtractedEntity] = field(default_factory=list)
    drug_name: Optional[str] = None
    active_ingredients: List[str] = field(default_factory=list)
    dosage_form: Optional[str] = None
    strength: Optional[str] = None
    manufacturer: Optional[str] = None
    processing_time_ms: float = 0.0
    
    @property
    def has_drug_name(self) -> bool:
        """Check if drug name was extracted."""
        return self.drug_name is not None
    
    @property
    def overall_confidence(self) -> ConfidenceScore:
        """Calculate overall extraction confidence."""
        if not self.entities:
            return ConfidenceScore.zero()
        
        avg_confidence = sum(e.confidence.value for e in self.entities) / len(self.entities)
        return ConfidenceScore(value=avg_confidence, source="entity_extraction")
    
    def get_entities_by_type(self, entity_type: EntityType) -> List[ExtractedEntity]:
        """Filter entities by type."""
        return [e for e in self.entities if e.entity_type == entity_type]
    
    def get_first_entity(self, entity_type: EntityType) -> Optional[ExtractedEntity]:
        """Get first entity of a specific type."""
        entities = self.get_entities_by_type(entity_type)
        return entities[0] if entities else None


@dataclass
class KnowledgeChunk:
    """
    Represents a chunk of retrieved knowledge from RAG.
    
    Attributes:
        content: The knowledge text content
        source: Source document/database identifier
        relevance_score: Similarity/relevance score
        metadata: Additional metadata about the chunk
    """
    
    content: str
    source: str
    relevance_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_relevant(self) -> bool:
        """Check if chunk meets relevance threshold."""
        return self.relevance_score >= 0.7
    
    def __str__(self) -> str:
        preview = self.content[:100] + "..." if len(self.content) > 100 else self.content
        return f"KnowledgeChunk(source='{self.source}', score={self.relevance_score:.2f})"


@dataclass
class KnowledgeRetrievalResult:
    """
    Result from the RAG knowledge retrieval stage.
    
    Contains retrieved knowledge chunks relevant
    to the identified drug.
    
    Attributes:
        chunks: List of retrieved knowledge chunks
        query_used: Query string used for retrieval
        total_chunks_searched: Number of chunks in the database
        processing_time_ms: Time taken for retrieval
    """
    
    chunks: List[KnowledgeChunk] = field(default_factory=list)
    query_used: Optional[str] = None
    total_chunks_searched: int = 0
    processing_time_ms: float = 0.0
    
    @property
    def has_knowledge(self) -> bool:
        """Check if any relevant knowledge was retrieved."""
        return len(self.chunks) > 0
    
    @property
    def best_chunk(self) -> Optional[KnowledgeChunk]:
        """Get the highest relevance chunk."""
        if not self.chunks:
            return None
        return max(self.chunks, key=lambda c: c.relevance_score)
    
    @property
    def combined_knowledge(self) -> str:
        """Combine all chunks into a single text."""
        return "\n\n".join(chunk.content for chunk in self.chunks)
    
    def get_relevant_chunks(self, min_score: float = 0.5) -> List[KnowledgeChunk]:
        """Get chunks above a relevance threshold."""
        return [c for c in self.chunks if c.relevance_score >= min_score]
