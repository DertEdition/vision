"""
Dosage Information Value Objects

Represents pharmaceutical dosage forms and related information.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class DosageForm(Enum):
    """Enumeration of common pharmaceutical dosage forms."""
    
    # Solid forms
    TABLET = "tablet"
    CAPSULE = "capsule"
    POWDER = "powder"
    GRANULE = "granule"
    LOZENGE = "lozenge"
    SUPPOSITORY = "suppository"
    
    # Liquid forms
    SYRUP = "syrup"
    SOLUTION = "solution"
    SUSPENSION = "suspension"
    EMULSION = "emulsion"
    DROPS = "drops"
    
    # Semi-solid forms
    CREAM = "cream"
    OINTMENT = "ointment"
    GEL = "gel"
    PASTE = "paste"
    
    # Injectable forms
    INJECTION = "injection"
    INFUSION = "infusion"
    
    # Inhalation forms
    INHALER = "inhaler"
    NEBULIZER = "nebulizer"
    
    # Topical forms
    PATCH = "patch"
    SPRAY = "spray"
    
    # Other
    UNKNOWN = "unknown"
    
    @classmethod
    def from_string(cls, value: str) -> "DosageForm":
        """
        Parse dosage form from string, handling common variations.
        
        Args:
            value: String representation of dosage form
            
        Returns:
            Matching DosageForm enum value
        """
        value_lower = value.lower().strip()
        
        # Direct match
        for form in cls:
            if form.value == value_lower:
                return form
        
        # Common variations and translations (including Turkish)
        variations = {
            # Tablets
            "tablet": cls.TABLET,
            "tab": cls.TABLET,
            "tablets": cls.TABLET,
            "film tablet": cls.TABLET,
            "film kaplı tablet": cls.TABLET,
            "çiğneme tableti": cls.TABLET,
            "efervesan tablet": cls.TABLET,
            
            # Capsules
            "capsule": cls.CAPSULE,
            "cap": cls.CAPSULE,
            "capsules": cls.CAPSULE,
            "kapsül": cls.CAPSULE,
            "sert kapsül": cls.CAPSULE,
            "yumuşak kapsül": cls.CAPSULE,
            
            # Syrups
            "syrup": cls.SYRUP,
            "şurup": cls.SYRUP,
            "oral süspansiyon": cls.SUSPENSION,
            
            # Creams and ointments
            "cream": cls.CREAM,
            "krem": cls.CREAM,
            "ointment": cls.OINTMENT,
            "merhem": cls.OINTMENT,
            "pomad": cls.OINTMENT,
            "jel": cls.GEL,
            
            # Solutions
            "solution": cls.SOLUTION,
            "solüsyon": cls.SOLUTION,
            "çözelti": cls.SOLUTION,
            "damla": cls.DROPS,
            
            # Injections
            "injection": cls.INJECTION,
            "enjeksiyon": cls.INJECTION,
            "enjeksiyonluk çözelti": cls.INJECTION,
            
            # Sprays
            "spray": cls.SPRAY,
            "sprey": cls.SPRAY,
            "nazal sprey": cls.SPRAY,
        }
        
        if value_lower in variations:
            return variations[value_lower]
        
        # Partial match
        for key, form in variations.items():
            if key in value_lower or value_lower in key:
                return form
        
        return cls.UNKNOWN


@dataclass(frozen=True)
class DosageInfo:
    """
    Immutable value object representing dosage information.
    
    Attributes:
        form: The pharmaceutical dosage form
        strength: Strength/concentration (e.g., "500 mg", "10 mg/ml")
        unit_count: Number of units in package (e.g., 30 tablets)
        route: Administration route (e.g., "oral", "topical")
    """
    
    form: DosageForm
    strength: Optional[str] = None
    unit_count: Optional[int] = None
    route: Optional[str] = None
    
    @property
    def is_oral(self) -> bool:
        """Check if this is an oral dosage form."""
        oral_forms = {
            DosageForm.TABLET,
            DosageForm.CAPSULE,
            DosageForm.SYRUP,
            DosageForm.SOLUTION,
            DosageForm.SUSPENSION,
            DosageForm.DROPS,
            DosageForm.POWDER,
            DosageForm.GRANULE,
            DosageForm.LOZENGE,
        }
        return self.form in oral_forms
    
    @property
    def is_topical(self) -> bool:
        """Check if this is a topical dosage form."""
        topical_forms = {
            DosageForm.CREAM,
            DosageForm.OINTMENT,
            DosageForm.GEL,
            DosageForm.PASTE,
            DosageForm.PATCH,
            DosageForm.SPRAY,
        }
        return self.form in topical_forms
    
    @property
    def is_injectable(self) -> bool:
        """Check if this is an injectable dosage form."""
        injectable_forms = {
            DosageForm.INJECTION,
            DosageForm.INFUSION,
        }
        return self.form in injectable_forms
    
    def __str__(self) -> str:
        parts = [self.form.value.capitalize()]
        if self.strength:
            parts.append(self.strength)
        if self.unit_count:
            parts.append(f"({self.unit_count} units)")
        return " ".join(parts)
    
    @classmethod
    def unknown(cls) -> "DosageInfo":
        """Create an unknown dosage info."""
        return cls(form=DosageForm.UNKNOWN)
