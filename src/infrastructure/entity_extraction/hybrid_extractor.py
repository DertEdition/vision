"""
Hybrid Entity Extractor

Combines rule-based patterns with optional NER for pharmaceutical entity extraction.
"""

from typing import Optional, Dict, Any, List, Tuple
import logging
import re
import time

from ...domain.ports.entity_extractor import EntityExtractorPort
from ...domain.value_objects.confidence_score import ConfidenceScore
from ...domain.entities.extraction_result import (
    EntityExtractionResult,
    ExtractedEntity,
    EntityType,
)
from ...domain.exceptions import EntityExtractionError


logger = logging.getLogger(__name__)


class HybridEntityExtractor(EntityExtractorPort):
    """
    Hybrid entity extractor combining rule-based patterns and optional NER.
    
    Strategy:
    1. Rule-based patterns for common Turkish drug naming conventions
    2. Dosage form and strength extraction using patterns
    3. Optional LLM refinement for ambiguous cases
    
    Patterns are optimized for Turkish pharmaceutical packaging.
    """
    
    # Common Turkish drug name suffixes
    DRUG_NAME_PATTERNS = [
        # Brand names often end with these
        r"\b([A-ZÇĞİÖŞÜ][A-Za-zçğıöşü]+(?:\s+(?:Fort|Plus|Forte|XR|SR|CR|MR|LA|ER))?)\s*\d+\s*(?:mg|gr|g|ml|mcg)",
        # All caps drug names
        r"\b([A-ZÇĞİÖŞÜ]{3,}(?:\s+[A-ZÇĞİÖŞÜ]+)*)\b",
        # Mixed case brand names
        r"\b([A-ZÇĞİÖŞÜ][a-zçğıöşü]+(?:[A-ZÇĞİÖŞÜ][a-zçğıöşü]*)*)\b",
    ]
    
    # Known Turkish drug names for fuzzy matching (helps with poor OCR)
    KNOWN_DRUG_NAMES = [
        "Parol", "Nurofen", "Aspirin", "Augmentin", "Zinnat", "Voltaren",
        "Aferin", "Gripin", "Majezik", "Apranax", "Calpol", "Panadol",
        "Theraflu", "Fervex", "Coldrex", "Tylol", "Dolven", "Dikloron",
        "Lansor", "Nexium", "Controloc", "Zyrtec", "Aerius", "Claritine",
        "Ventolin", "Concor", "Coraspin", "Supradyn", "Redoxon", "Centrum",
        "Eferalgan", "Advil", "Minoset", "Dolorex", "Naprosyn", "Brufen",
        "Amoklavin", "Largopen", "Cefaks", "Cipro", "Flagyl", "Metpamid",
        "Vermidon", "Buscopan", "Debridat", "Gaviscon", "Rennie", "Ulcuran",
    ]
    
    # Active ingredient patterns (generic drug names)
    INGREDIENT_PATTERNS = [
        # Ingredients often listed after specific keywords
        r"(?:etken madde|aktif madde|içerik)[:\s]*([A-Za-zçğıöşüÇĞİÖŞÜ\s]+?)(?:\d|\.|,|$)",
        # Common ingredient endings
        r"\b([A-Za-z]+(?:azol|amin|mycin|cillin|pril|sartan|statin|olol|dipine|formin|gliptin|oxin|ine|ate|ide))\b",
        # Scientific names in parentheses
        r"\(([A-Za-z\s]+)\s*\d*\s*(?:mg|gr|g|mcg)\)",
        # Common Turkish drug ingredients
        r"\b(Parasetamol|Paracetamol|İbuprofen|Ibuprofen|Diklofenak|Naproksen|Amoksisilin|Klavulanik)\b",
    ]
    
    # Dosage form patterns (Turkish and English)
    DOSAGE_FORM_PATTERNS = {
        "tablet": [r"\btablet\b", r"\bfilm\s+(?:kaplı\s+)?tablet\b", r"\btab\b"],
        "capsule": [r"\bkapsül\b", r"\bcapsule\b", r"\bcap\b", r"\bsert\s+kapsül\b"],
        "syrup": [r"\bşurup\b", r"\bsyrup\b"],
        "solution": [r"\bsolüsyon\b", r"\bçözelti\b", r"\bsolution\b"],
        "suspension": [r"\bsüspansiyon\b", r"\bsuspension\b"],
        "cream": [r"\bkrem\b", r"\bcream\b"],
        "ointment": [r"\bmerhem\b", r"\bpomad\b", r"\bointment\b"],
        "gel": [r"\bjel\b", r"\bgel\b"],
        "drops": [r"\bdamla\b", r"\bdrops\b"],
        "injection": [r"\benjeksiyon\b", r"\binjection\b", r"\biv\b", r"\bim\b"],
        "spray": [r"\bsprey\b", r"\bspray\b"],
        "inhaler": [r"\binhaler\b"],
    }
    
    # Strength/dosage patterns
    STRENGTH_PATTERNS = [
        r"(\d+(?:[.,]\d+)?)\s*(mg|g|gr|ml|mcg|µg|iu|IU)",
        r"(\d+(?:[.,]\d+)?)\s*(mg|g|ml)/\s*(\d+(?:[.,]\d+)?)\s*(ml|g)",
        r"(\d+)\s*%",
    ]
    
    # Manufacturer patterns (Turkish pharmaceutical companies)
    MANUFACTURER_PATTERNS = [
        r"\b(Abdi İbrahim|Eczacıbaşı|Deva|Sanofi|Pfizer|Novartis|Bayer|Roche|"
        r"AstraZeneca|GSK|GlaxoSmithKline|Johnson|Merck|Abbott|Zentiva|"
        r"Sandoz|Teva|Mylan|Nobel|Biofarma|Bilim|İlko|Koçak|Mustafa Nevzat|Recordati|Haleon)\b"
    ]
    
    def __init__(
        self,
        use_llm_refinement: bool = False,
        confidence_boost_per_match: float = 0.1,
        min_drug_name_length: int = 3
    ):
        """
        Initialize the hybrid extractor.
        
        Args:
            use_llm_refinement: Whether to use LLM for ambiguous cases
            confidence_boost_per_match: Confidence boost per pattern match
            min_drug_name_length: Minimum length for drug name candidates
        """
        self._use_llm_refinement = use_llm_refinement
        self._confidence_boost = confidence_boost_per_match
        self._min_name_length = min_drug_name_length
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def extract(
        self,
        text: str,
        options: Optional[Dict[str, Any]] = None
    ) -> EntityExtractionResult:
        """
        Extract pharmaceutical entities from text.
        
        Args:
            text: Raw OCR text
            options: Optional configuration
            
        Returns:
            EntityExtractionResult with extracted entities
        """
        start_time = time.time()
        options = options or {}
        
        entities: List[ExtractedEntity] = []
        
        # Normalize text
        text_normalized = self._normalize_text(text)
        
        # Extract drug name
        drug_name_entity = self._extract_drug_name(text_normalized)
        if drug_name_entity:
            entities.append(drug_name_entity)
        
        # Extract active ingredients
        ingredient_entities = self._extract_ingredients(text_normalized)
        entities.extend(ingredient_entities)
        
        # Extract dosage form
        dosage_entity = self._extract_dosage_form(text_normalized)
        if dosage_entity:
            entities.append(dosage_entity)
        
        # Extract strength
        strength_entity = self._extract_strength(text_normalized)
        if strength_entity:
            entities.append(strength_entity)
        
        # Extract manufacturer
        manufacturer_entity = self._extract_manufacturer(text_normalized)
        if manufacturer_entity:
            entities.append(manufacturer_entity)
        
        processing_time = (time.time() - start_time) * 1000
        
        # Build result
        result = EntityExtractionResult(
            entities=entities,
            drug_name=drug_name_entity.value if drug_name_entity else None,
            active_ingredients=[
                e.value for e in ingredient_entities
            ],
            dosage_form=dosage_entity.value if dosage_entity else None,
            strength=strength_entity.value if strength_entity else None,
            manufacturer=manufacturer_entity.value if manufacturer_entity else None,
            processing_time_ms=processing_time
        )
        
        return result
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for pattern matching."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Keep original case for drug names
        return text.strip()
    
    def _extract_drug_name(self, text: str) -> Optional[ExtractedEntity]:
        """Extract the primary drug name."""
        candidates: List[Tuple[str, float]] = []
        
        # First, try to find known drug names in text (fuzzy match)
        known_match = self._find_known_drug_name(text)
        if known_match:
            return ExtractedEntity(
                entity_type=EntityType.DRUG_NAME,
                value=known_match[0],
                confidence=ConfidenceScore(value=known_match[1], source="known_drug_match")
            )
        
        for pattern in self.DRUG_NAME_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                name = match.group(1).strip()
                if len(name) >= self._min_name_length:
                    # Calculate confidence based on position and format
                    confidence = 0.5
                    
                    # Boost for early position in text
                    if match.start() < 50:
                        confidence += 0.2
                    
                    # Boost for all caps
                    if name.isupper():
                        confidence += 0.15
                    
                    # Boost if followed by dosage
                    if re.search(r'\d+\s*mg', text[match.end():match.end()+20], re.IGNORECASE):
                        confidence += 0.15
                    
                    candidates.append((name, min(confidence, 0.95)))
        
        if not candidates:
            # Fallback: take first capitalized word that's not common
            common_words = {"film", "tablet", "kapsül", "şurup", "mg", "ml", "adet"}
            words = text.split()
            for word in words:
                word_clean = re.sub(r'[^\w\s]', '', word)
                if (word_clean and word_clean[0].isupper() and 
                    len(word_clean) >= self._min_name_length and
                    word_clean.lower() not in common_words):
                    candidates.append((word_clean, 0.4))
                    break
        
        if candidates:
            # Sort by confidence and take best
            candidates.sort(key=lambda x: x[1], reverse=True)
            best_name, confidence = candidates[0]
            
            return ExtractedEntity(
                entity_type=EntityType.DRUG_NAME,
                value=best_name,
                confidence=ConfidenceScore(value=confidence, source="rule_based")
            )
        
        return None
    
    def _find_known_drug_name(self, text: str) -> Optional[Tuple[str, float]]:
        """Try to find a known drug name in text using fuzzy matching."""
        text_lower = text.lower()
        text_clean = re.sub(r'[^a-z\s]', '', text_lower)
        
        best_match = None
        best_score = 0.0
        
        for drug_name in self.KNOWN_DRUG_NAMES:
            drug_lower = drug_name.lower()
            
            # Exact match
            if drug_lower in text_lower:
                return (drug_name, 0.95)
            
            # Check for partial/garbled matches (OCR errors)
            # Look for words that start with same letters
            words = text_clean.split()
            for word in words:
                if len(word) >= 3:
                    # Check prefix match (e.g., "par" matches "parol")
                    if drug_lower.startswith(word[:3]) or word.startswith(drug_lower[:3]):
                        similarity = self._calculate_similarity(word, drug_lower)
                        if similarity > best_score and similarity > 0.5:
                            best_score = similarity
                            best_match = drug_name
        
        if best_match:
            return (best_match, min(0.7 + best_score * 0.2, 0.9))
        return None
    
    def _calculate_similarity(self, s1: str, s2: str) -> float:
        """Calculate simple similarity between two strings."""
        if not s1 or not s2:
            return 0.0
        
        # Simple Jaccard-like similarity on characters
        set1 = set(s1)
        set2 = set(s2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def _extract_ingredients(self, text: str) -> List[ExtractedEntity]:
        """Extract active ingredients."""
        ingredients: List[ExtractedEntity] = []
        seen = set()
        
        for pattern in self.INGREDIENT_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                ingredient = match.group(1).strip()
                ingredient_lower = ingredient.lower()
                
                if ingredient_lower not in seen and len(ingredient) >= 3:
                    seen.add(ingredient_lower)
                    ingredients.append(ExtractedEntity(
                        entity_type=EntityType.ACTIVE_INGREDIENT,
                        value=ingredient.title(),
                        confidence=ConfidenceScore(value=0.7, source="rule_based")
                    ))
        
        return ingredients
    
    def _extract_dosage_form(self, text: str) -> Optional[ExtractedEntity]:
        """Extract dosage form."""
        text_lower = text.lower()
        
        for form_name, patterns in self.DOSAGE_FORM_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return ExtractedEntity(
                        entity_type=EntityType.DOSAGE_FORM,
                        value=form_name,
                        confidence=ConfidenceScore(value=0.85, source="rule_based")
                    )
        
        return None
    
    def _extract_strength(self, text: str) -> Optional[ExtractedEntity]:
        """Extract dosage strength."""
        for pattern in self.STRENGTH_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                strength = match.group(0)
                # Normalize
                strength = re.sub(r'\s+', ' ', strength).strip()
                
                return ExtractedEntity(
                    entity_type=EntityType.STRENGTH,
                    value=strength,
                    confidence=ConfidenceScore(value=0.9, source="rule_based")
                )
        
        return None
    
    def _extract_manufacturer(self, text: str) -> Optional[ExtractedEntity]:
        """Extract manufacturer name."""
        for pattern in self.MANUFACTURER_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return ExtractedEntity(
                    entity_type=EntityType.MANUFACTURER,
                    value=match.group(1),
                    confidence=ConfidenceScore(value=0.9, source="rule_based")
                )
        
        return None
    
    def extract_drug_name(self, text: str) -> Optional[ExtractedEntity]:
        """Extract just the drug name."""
        return self._extract_drug_name(self._normalize_text(text))
    
    def extract_active_ingredients(self, text: str) -> List[ExtractedEntity]:
        """Extract just the active ingredients."""
        return self._extract_ingredients(self._normalize_text(text))
    
    @property
    def extractor_name(self) -> str:
        return "HybridEntityExtractor"


class DummyEntityExtractor(EntityExtractorPort):
    """
    Dummy entity extractor for testing.
    """
    
    def __init__(
        self,
        drug_name: str = "Sample Drug",
        ingredients: List[str] = None
    ):
        self._drug_name = drug_name
        self._ingredients = ingredients or ["Paracetamol"]
    
    def extract(
        self,
        text: str,
        options: Optional[Dict[str, Any]] = None
    ) -> EntityExtractionResult:
        return EntityExtractionResult(
            entities=[
                ExtractedEntity(
                    entity_type=EntityType.DRUG_NAME,
                    value=self._drug_name,
                    confidence=ConfidenceScore(value=0.9, source="dummy")
                )
            ],
            drug_name=self._drug_name,
            active_ingredients=self._ingredients,
            dosage_form="tablet",
            processing_time_ms=1.0
        )
    
    def extract_drug_name(self, text: str) -> Optional[ExtractedEntity]:
        return ExtractedEntity(
            entity_type=EntityType.DRUG_NAME,
            value=self._drug_name,
            confidence=ConfidenceScore(value=0.9, source="dummy")
        )
    
    def extract_active_ingredients(self, text: str) -> List[ExtractedEntity]:
        return [
            ExtractedEntity(
                entity_type=EntityType.ACTIVE_INGREDIENT,
                value=ing,
                confidence=ConfidenceScore(value=0.8, source="dummy")
            )
            for ing in self._ingredients
        ]
    
    @property
    def extractor_name(self) -> str:
        return "DummyEntityExtractor"
