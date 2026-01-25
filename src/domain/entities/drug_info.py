"""
Drug Information Entity

Core domain entity representing identified pharmaceutical drug information.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime

from ..value_objects.confidence_score import ConfidenceScore
from ..value_objects.dosage_info import DosageForm, DosageInfo


@dataclass
class DrugInfo:
    """
    Domain entity representing identified drug information.
    
    This is the primary output of the entity extraction stage,
    containing all relevant information extracted from a drug image.
    
    Attributes:
        drug_name: The identified drug brand name
        active_ingredients: List of active pharmaceutical ingredients
        dosage_info: Dosage form and strength information
        manufacturer: Manufacturer or pharmaceutical company name
        confidence: Overall confidence in the extraction
        barcode: Barcode number if detected
        batch_number: Batch/lot number if detected
        expiry_date: Expiry date if detected
        extracted_at: Timestamp of extraction
        source_text: Raw text from which info was extracted
    """
    
    drug_name: str
    active_ingredients: List[str] = field(default_factory=list)
    dosage_info: Optional[DosageInfo] = None
    manufacturer: Optional[str] = None
    confidence: ConfidenceScore = field(default_factory=ConfidenceScore.zero)
    barcode: Optional[str] = None
    batch_number: Optional[str] = None
    expiry_date: Optional[str] = None
    extracted_at: datetime = field(default_factory=datetime.now)
    source_text: Optional[str] = None
    
    def __post_init__(self) -> None:
        """Validate drug information."""
        if not self.drug_name or not self.drug_name.strip():
            raise ValueError("Drug name cannot be empty")
        
        # Normalize drug name
        self.drug_name = self.drug_name.strip()
    
    @property
    def has_active_ingredients(self) -> bool:
        """Check if active ingredients were identified."""
        return len(self.active_ingredients) > 0
    
    @property
    def dosage_form(self) -> Optional[DosageForm]:
        """Get dosage form if available."""
        return self.dosage_info.form if self.dosage_info else None
    
    @property
    def strength(self) -> Optional[str]:
        """Get strength/concentration if available."""
        return self.dosage_info.strength if self.dosage_info else None
    
    @property
    def is_high_confidence(self) -> bool:
        """Check if extraction has high confidence."""
        return self.confidence.value >= 0.7
    
    @property
    def is_complete(self) -> bool:
        """Check if essential information is available."""
        return (
            bool(self.drug_name) and
            self.has_active_ingredients and
            self.dosage_info is not None
        )
    
    def get_summary(self) -> str:
        """
        Get a brief summary of the drug information.
        
        Returns:
            Human-readable summary string
        """
        parts = [self.drug_name]
        
        if self.dosage_info and self.dosage_info.strength:
            parts.append(self.dosage_info.strength)
        
        if self.dosage_info:
            parts.append(f"({self.dosage_info.form.value})")
        
        return " ".join(parts)
    
    def get_active_ingredients_string(self) -> str:
        """
        Get active ingredients as a formatted string.
        
        Returns:
            Comma-separated list of active ingredients
        """
        if not self.active_ingredients:
            return "Unknown"
        return ", ".join(self.active_ingredients)
    
    def merge_with(self, other: "DrugInfo") -> "DrugInfo":
        """
        Merge with another DrugInfo, preferring higher confidence values.
        
        Args:
            other: Another DrugInfo to merge with
            
        Returns:
            New DrugInfo with merged information
        """
        # Prefer higher confidence drug name
        if other.confidence.value > self.confidence.value:
            drug_name = other.drug_name
            primary_confidence = other.confidence
        else:
            drug_name = self.drug_name
            primary_confidence = self.confidence
        
        # Merge active ingredients (union)
        all_ingredients = list(set(self.active_ingredients + other.active_ingredients))
        
        # Prefer non-None values, then higher confidence
        dosage_info = self.dosage_info or other.dosage_info
        manufacturer = self.manufacturer or other.manufacturer
        barcode = self.barcode or other.barcode
        batch_number = self.batch_number or other.batch_number
        expiry_date = self.expiry_date or other.expiry_date
        
        return DrugInfo(
            drug_name=drug_name,
            active_ingredients=all_ingredients,
            dosage_info=dosage_info,
            manufacturer=manufacturer,
            confidence=primary_confidence,
            barcode=barcode,
            batch_number=batch_number,
            expiry_date=expiry_date,
            source_text=f"{self.source_text or ''}\n{other.source_text or ''}".strip() or None
        )
    
    def __str__(self) -> str:
        return self.get_summary()
    
    def __repr__(self) -> str:
        return (
            f"DrugInfo(name='{self.drug_name}', "
            f"ingredients={self.active_ingredients}, "
            f"confidence={self.confidence})"
        )
    
    @classmethod
    def unknown(cls, source_text: Optional[str] = None) -> "DrugInfo":
        """
        Create a placeholder for unknown/unidentified drug.
        
        Args:
            source_text: Raw text that couldn't be parsed
            
        Returns:
            DrugInfo with unknown marker
        """
        return cls(
            drug_name="Unknown Drug",
            confidence=ConfidenceScore.zero(),
            source_text=source_text
        )
