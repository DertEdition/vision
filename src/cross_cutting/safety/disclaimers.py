"""
Disclaimer Injection

Mandatory disclaimers for pharmaceutical information.
"""

from typing import Optional
from enum import Enum


class DisclaimerLanguage(Enum):
    """Supported disclaimer languages."""
    TURKISH = "tr"
    ENGLISH = "en"


# Medical disclaimers in different languages
MEDICAL_DISCLAIMER = {
    "tr": """
⚠️ ÖNEMLİ UYARI

Bu bilgiler yalnızca genel bilgilendirme amaçlıdır ve profesyonel tıbbi tavsiye, teşhis veya tedavi yerine geçmez.

• Herhangi bir ilaç kullanmadan önce mutlaka doktorunuza veya eczacınıza danışınız.
• İlaç tedavisini doktorunuza danışmadan başlatmayın, değiştirmeyin veya sonlandırmayın.
• Bu bilgiler kişisel tıbbi tavsiye niteliği taşımaz.
• Acil bir sağlık sorunu yaşıyorsanız derhal tıbbi yardım alınız.
""".strip(),
    
    "en": """
⚠️ IMPORTANT DISCLAIMER

This information is for general informational purposes only and is NOT a substitute for professional medical advice, diagnosis, or treatment.

• Always consult your doctor or pharmacist before taking any medication.
• Do not start, change, or discontinue any medication without consulting your doctor.
• This information does not constitute personal medical advice.
• If you are experiencing a medical emergency, seek immediate medical attention.
""".strip()
}


# Short disclaimers for inline use
SHORT_DISCLAIMER = {
    "tr": "⚠️ Doktorunuza danışmadan ilaç kullanmayınız.",
    "en": "⚠️ Do not use medication without consulting your doctor."
}


class DisclaimerInjector:
    """
    Injects mandatory disclaimers into responses.
    
    Ensures all pharmaceutical information includes appropriate
    medical disclaimers and warnings.
    """
    
    def __init__(self, language: str = "tr"):
        """
        Initialize disclaimer injector.
        
        Args:
            language: Disclaimer language code
        """
        self.language = language
    
    def get_full_disclaimer(self, language: Optional[str] = None) -> str:
        """
        Get the full medical disclaimer.
        
        Args:
            language: Override language
            
        Returns:
            Full disclaimer text
        """
        lang = language or self.language
        return MEDICAL_DISCLAIMER.get(lang, MEDICAL_DISCLAIMER["en"])
    
    def get_short_disclaimer(self, language: Optional[str] = None) -> str:
        """
        Get a short inline disclaimer.
        
        Args:
            language: Override language
            
        Returns:
            Short disclaimer text
        """
        lang = language or self.language
        return SHORT_DISCLAIMER.get(lang, SHORT_DISCLAIMER["en"])
    
    def inject_disclaimer(
        self,
        response: str,
        position: str = "end",
        language: Optional[str] = None
    ) -> str:
        """
        Inject disclaimer into a response.
        
        Args:
            response: Response text
            position: Where to inject ('start', 'end', 'both')
            language: Override language
            
        Returns:
            Response with injected disclaimer
        """
        disclaimer = self.get_full_disclaimer(language)
        
        if position == "start":
            return f"{disclaimer}\n\n---\n\n{response}"
        elif position == "end":
            return f"{response}\n\n---\n\n{disclaimer}"
        elif position == "both":
            short = self.get_short_disclaimer(language)
            return f"{short}\n\n{response}\n\n---\n\n{disclaimer}"
        else:
            return f"{response}\n\n{disclaimer}"
    
    def has_disclaimer(self, response: str) -> bool:
        """
        Check if response already has a disclaimer.
        
        Args:
            response: Response to check
            
        Returns:
            True if disclaimer present
        """
        # Check for common disclaimer indicators
        indicators = [
            "ÖNEMLİ UYARI",
            "IMPORTANT DISCLAIMER",
            "doktorunuza danışınız",
            "consult your doctor",
            "tıbbi tavsiye yerine geçmez",
            "NOT a substitute",
        ]
        
        response_upper = response.upper()
        return any(ind.upper() in response_upper for ind in indicators)
    
    def ensure_disclaimer(
        self,
        response: str,
        language: Optional[str] = None
    ) -> str:
        """
        Ensure response has a disclaimer, adding if needed.
        
        Args:
            response: Response text
            language: Override language
            
        Returns:
            Response guaranteed to have disclaimer
        """
        if self.has_disclaimer(response):
            return response
        return self.inject_disclaimer(response, "end", language)
