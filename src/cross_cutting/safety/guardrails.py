"""
Safety Guardrails

Implements safety checks and content filtering for pharmaceutical responses.
"""

from typing import List, Optional, Tuple
import re
import logging

from ...domain.value_objects.confidence_score import ConfidenceScore


logger = logging.getLogger(__name__)


class SafetyGuardrails:
    """
    Safety guardrails for pharmaceutical content.
    
    Ensures responses don't contain:
    - Diagnostic language
    - Treatment recommendations
    - Dosage prescriptions
    - Personalized medical advice
    - Potentially harmful content
    """
    
    # Patterns indicating unsafe content
    UNSAFE_PATTERNS = [
        # Diagnostic language
        (r"\byou\s+(?:have|suffer\s+from|are\s+diagnosed)\b", "diagnostic_language"),
        (r"\bdiagnos(?:e|is|ing)\b", "diagnostic_language"),
        
        # Treatment recommendations
        (r"\byou\s+should\s+(?:take|use|try)\b", "treatment_recommendation"),
        (r"\bI\s+(?:recommend|suggest|advise)\s+(?:that\s+)?you\b", "treatment_recommendation"),
        (r"\bstart\s+taking\b", "treatment_recommendation"),
        (r"\bstop\s+taking\b", "treatment_recommendation"),
        
        # Dosage prescriptions
        (r"\btake\s+\d+\s*(?:mg|g|ml|tablet|pill|capsule)", "dosage_prescription"),
        (r"\bincrease\s+(?:your\s+)?(?:dose|dosage)\b", "dosage_prescription"),
        (r"\btake\s+(?:one|two|three|\d+)\s+(?:times?\s+)?(?:a\s+)?day\b", "dosage_prescription"),
        
        # Personalized advice
        (r"\bfor\s+your\s+(?:condition|case|situation)\b", "personalized_advice"),
        (r"\bin\s+your\s+case\b", "personalized_advice"),
        
        # Prescription language
        (r"\bprescribe\b", "prescription_language"),
        (r"\bprescription\s+(?:is|would\s+be)\b", "prescription_language"),
    ]
    
    # Required elements in safe responses
    REQUIRED_ELEMENTS = [
        "disclaimer",  # Must have some form of disclaimer
    ]
    
    # Disclaimer indicators
    DISCLAIMER_PATTERNS = [
        r"(?:not\s+)?(?:a\s+)?substitute\s+for\s+(?:professional|medical)\s+advice",
        r"consult\s+(?:a\s+)?(?:doctor|physician|healthcare|pharmacist)",
        r"danışınız",  # Turkish: consult
        r"doktorunuza",  # Turkish: your doctor
        r"eczacınıza",  # Turkish: your pharmacist
        r"tıbbi\s+tavsiye",  # Turkish: medical advice
        r"⚠️",  # Warning emoji as indicator
        r"UYARI|WARNING|DISCLAIMER",
    ]
    
    def __init__(
        self,
        confidence_threshold: float = 0.5,
        strict_mode: bool = True
    ):
        """
        Initialize safety guardrails.
        
        Args:
            confidence_threshold: Minimum confidence for reliable outputs
            strict_mode: If True, applies stricter safety checks
        """
        self.confidence_threshold = confidence_threshold
        self.strict_mode = strict_mode
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def check_response(self, response: str) -> Tuple[bool, List[str]]:
        """
        Check if a response passes safety validation.
        
        Args:
            response: Response text to validate
            
        Returns:
            Tuple of (is_safe, list of violations)
        """
        violations = []
        
        # Check for unsafe patterns
        for pattern, violation_type in self.UNSAFE_PATTERNS:
            if re.search(pattern, response, re.IGNORECASE):
                violations.append(violation_type)
                self.logger.warning(f"Unsafe pattern detected: {violation_type}")
        
        # Check for required elements
        if self.strict_mode:
            has_disclaimer = any(
                re.search(p, response, re.IGNORECASE)
                for p in self.DISCLAIMER_PATTERNS
            )
            if not has_disclaimer:
                violations.append("missing_disclaimer")
                self.logger.warning("Response missing disclaimer")
        
        is_safe = len(violations) == 0
        return is_safe, violations
    
    def check_confidence(
        self,
        confidence: ConfidenceScore
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if confidence level is acceptable.
        
        Args:
            confidence: Confidence score to check
            
        Returns:
            Tuple of (is_acceptable, warning_message)
        """
        if confidence.value < self.confidence_threshold:
            warning = (
                "Confidence level is low. Results may be inaccurate. "
                "Please verify the drug information manually."
            )
            return False, warning
        
        if confidence.value < 0.7:
            warning = (
                "Some information may be incomplete. "
                "Please consult the drug packaging for accurate details."
            )
            return True, warning
        
        return True, None
    
    def sanitize_response(self, response: str) -> str:
        """
        Remove or modify unsafe content from a response.
        
        Args:
            response: Response to sanitize
            
        Returns:
            Sanitized response
        """
        sanitized = response
        
        # Replace dosage-specific numbers with general guidance
        sanitized = re.sub(
            r"take\s+\d+\s*(mg|tablet|pill|capsule)",
            "follow the prescribed dosage",
            sanitized,
            flags=re.IGNORECASE
        )
        
        # Replace diagnostic language
        sanitized = re.sub(
            r"you\s+have\s+",
            "if you experience ",
            sanitized,
            flags=re.IGNORECASE
        )
        
        # Replace recommendation language
        sanitized = re.sub(
            r"you\s+should\s+take",
            "this medication is typically taken",
            sanitized,
            flags=re.IGNORECASE
        )
        
        return sanitized
    
    def get_low_confidence_warning(self, language: str = "tr") -> str:
        """Get low confidence warning in specified language."""
        warnings = {
            "tr": (
                "⚠️ Dikkat: Otomatik tanıma güvenilirliği düşük. "
                "Lütfen ilaç bilgilerini ambalajdan doğrulayın."
            ),
            "en": (
                "⚠️ Warning: Automatic recognition confidence is low. "
                "Please verify drug information from the packaging."
            )
        }
        return warnings.get(language, warnings["en"])
