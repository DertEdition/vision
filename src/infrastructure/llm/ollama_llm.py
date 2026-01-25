"""
Ollama LLM Response Generator

Local LLM implementation using Ollama for running quantized models
on consumer hardware (e.g., RTX 3060 with 6GB VRAM).

Supports:
- Qwen3, Llama, Mistral, Phi, Gemma and other Ollama models
- Streaming responses
- Drug-specific prompt templates
- Turkish language support
"""

from typing import Optional, Dict, Any, List
import logging
import time
import json
import re

from ...domain.ports.response_generator import ResponseGeneratorPort
from ...domain.entities.extraction_result import (
    KnowledgeRetrievalResult,
    KnowledgeChunk,
)
from ...domain.entities.drug_info import DrugInfo
from ...domain.exceptions import (
    LLMConnectionError,
    ResponseGenerationError,
)


logger = logging.getLogger(__name__)


# Response templates
TEMPLATES = {
    "default": """Sen bir ilaç bilgi asistanısın. Aşağıdaki ilaç hakkında Türkçe olarak kapsamlı bilgi ver.

**İlaç Adı:** {drug_name}
**Etken Madde(ler):** {active_ingredients}
**Dozaj Formu:** {dosage_form}
**Doz:** {strength}
**Üretici:** {manufacturer}

**Bilgi Tabanından Bulunan Bilgiler:**
{knowledge_context}

Lütfen aşağıdaki formatta yanıt ver:

**İlaç Adı:** [İlacın ticari adı]
**Etken Madde:** [Etken madde(ler)]
**Üretici:** [Üretici firma]
**Kullanım Alanları:** [Ne için kullanılır]
**Doz ve Kullanım:** [Nasıl kullanılır]
**Uyarılar:** [Dikkat edilmesi gerekenler]
**Yan Etkiler:** [Olası yan etkiler]

⚠️ **UYARI:** Bu bilgiler yalnızca eğitim amaçlıdır. Herhangi bir ilaç kullanmadan önce doktorunuza veya eczacınıza danışın.""",

    "detailed": """Sen bir farmasötik bilgi uzmanısın. Aşağıdaki ilaç hakkında detaylı ve kapsamlı bilgi ver.

**İlaç:** {drug_name} ({active_ingredients})
**Form:** {dosage_form}, {strength}
**Üretici:** {manufacturer}

**Mevcut Bilgiler:**
{knowledge_context}

Detaylı açıklama yap:
1. İlaç hakkında genel bilgi
2. Etki mekanizması
3. Kullanım alanları ve endikasyonlar
4. Dozaj ve uygulama
5. Kontrendikasyonlar ve uyarılar
6. Yan etkiler
7. İlaç etkileşimleri
8. Saklama koşulları

⚠️ Bu bilgiler profesyonel tıbbi tavsiye yerine geçmez.""",

    "brief": """İlaç: {drug_name}
Etken: {active_ingredients}
Form: {dosage_form} {strength}

Bilgi: {knowledge_context}

Kısa ve öz bilgi ver (maksimum 3 cümle).""",

    "patient_friendly": """Merhaba! {drug_name} hakkında basit ve anlaşılır bilgi veriyorum.

Bu ilacın etken maddesi {active_ingredients}, {dosage_form} formundadır.

Bilgi tabanından:
{knowledge_context}

Lütfen hastanın anlayabileceği basit dilde açıkla:
- Bu ilaç ne için kullanılır?
- Nasıl kullanılır?
- Nelere dikkat edilmeli?

⚠️ Doktorunuza danışmadan ilaç kullanmayın."""
}


class OllamaResponseGenerator(ResponseGeneratorPort):
    """
    Response generator using Ollama for local LLM inference.
    
    Supports various quantized models optimized for consumer GPUs.
    Recommended models for 6GB VRAM:
    - qwen3:4b (best quality/speed balance)
    - qwen2.5:7b-instruct-q4_K_M
    - phi3.5
    - gemma3:4b
    
    Attributes:
        base_url: Ollama API base URL (default: http://localhost:11434)
        model: Model name (e.g., "qwen3:4b")
        temperature: Response temperature
        language: Response language
    """

    # Dangerous phrases that should not appear in responses
    PROHIBITED_PHRASES = [
        "teşhis koyuyorum",
        "tanı koyuyorum",
        "hastalığınız",
        "bu dozda kullanın",
        "reçete ediyorum",
        "tedavi planı",
        "bu ilacı alın",
    ]

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen3:4b",
        temperature: float = 0.3,
        max_tokens: int = 1000,
        language: str = "tr",
        timeout: int = 300
    ):
        """
        Initialize Ollama response generator.
        
        Args:
            base_url: Ollama API URL
            model: Model name
            temperature: Creativity (0.0-1.0)
            max_tokens: Maximum response length
            language: Response language (tr/en)
            timeout: Request timeout in seconds
        """
        self._base_url = base_url.rstrip('/')
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._language = language
        self._timeout = timeout
        self._session = None
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def _init_session(self):
        """Initialize HTTP session."""
        try:
            import requests
            self._session = requests.Session()
            self._session.headers.update({
                'Content-Type': 'application/json',
            })
        except ImportError:
            raise LLMConnectionError("requests not installed")
    
    def _check_ollama_available(self) -> bool:
        """Check if Ollama is running and model is available."""
        if not self._session:
            self._init_session()
        
        try:
            # Check Ollama server
            response = self._session.get(
                f"{self._base_url}/api/tags",
                timeout=5
            )
            response.raise_for_status()
            
            # Check if model exists
            models = response.json().get('models', [])
            model_names = [m.get('name', '') for m in models]
            
            # Check for exact match or partial match
            model_base = self._model.split(':')[0]
            if self._model in model_names:
                return True
            
            for name in model_names:
                if model_base in name:
                    self._model = name  # Use the actual name
                    return True
            
            self.logger.warning(f"Model '{self._model}' not found. Available: {model_names}")
            return False
            
        except Exception as e:
            self.logger.error(f"Ollama connection failed: {e}")
            return False
    
    def _format_knowledge_context(self, knowledge: KnowledgeRetrievalResult) -> str:
        """Format knowledge chunks into context string."""
        if not knowledge or not knowledge.chunks:
            return "Bilgi tabanında bu ilaç hakkında ek bilgi bulunamadı."
        
        context_parts = []
        for i, chunk in enumerate(knowledge.chunks[:3], 1):
            source = chunk.source or "Bilinmeyen"
            score = chunk.relevance_score or 0.0
            content = chunk.content[:500] if chunk.content else ""
            context_parts.append(f"[Kaynak {i}: {source}, Skor: {score:.2f}]\n{content}")
        
        return "\n\n".join(context_parts)
    
    def _build_prompt(
        self,
        drug_info: DrugInfo,
        knowledge: KnowledgeRetrievalResult,
        template_name: str = "default"
    ) -> str:
        """Build the prompt from drug info and knowledge context."""
        template = TEMPLATES.get(template_name, TEMPLATES["detailed"])
        
        knowledge_context = self._format_knowledge_context(knowledge)
        
        # Format prompt - get dosage form as string
        dosage_form_str = "Bilinmiyor"
        if drug_info.dosage_form:
            dosage_form_str = drug_info.dosage_form.value if hasattr(drug_info.dosage_form, 'value') else str(drug_info.dosage_form)
        
        prompt = template.format(
            drug_name=drug_info.drug_name or "Bilinmiyor",
            active_ingredients=", ".join(drug_info.active_ingredients) if drug_info.active_ingredients else "Bilinmiyor",
            dosage_form=dosage_form_str,
            strength=drug_info.strength or "Bilinmiyor",
            manufacturer=drug_info.manufacturer or "Bilinmiyor",
            knowledge_context=knowledge_context
        )
        
        return prompt
    
    def generate(
        self,
        drug_info: DrugInfo,
        knowledge: KnowledgeRetrievalResult,
        options: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate comprehensive drug information response.
        
        Args:
            drug_info: Identified drug information
            knowledge: Retrieved knowledge chunks
            options: Additional options
            
        Returns:
            Generated explanation text
        """
        start_time = time.time()
        options = options or {}
        
        # Initialize
        if not self._session:
            self._init_session()
        
        # Check Ollama availability
        if not self._check_ollama_available():
            raise LLMConnectionError(
                f"Ollama not available or model '{self._model}' not found. "
                f"Run: ollama pull {self._model}"
            )
        
        # Build prompt
        template_name = options.get("template", "default")
        prompt = self._build_prompt(drug_info, knowledge, template_name)
        
        # Generate response
        try:
            response_text = self._call_ollama(prompt)
            
            # Validate response
            if not self.validate_response(response_text):
                self.logger.warning("Response failed safety validation, using sanitized version")
                response_text = self._sanitize_response(response_text)
            
            processing_time = (time.time() - start_time) * 1000
            self.logger.info(f"Generated response in {processing_time:.0f}ms")
            
            return response_text
            
        except Exception as e:
            raise ResponseGenerationError(f"Ollama generation failed: {e}")
    
    def generate_with_template(
        self,
        drug_info: DrugInfo,
        knowledge: KnowledgeRetrievalResult,
        template_name: str
    ) -> str:
        """Generate response using a specific template."""
        return self.generate(drug_info, knowledge, {"template": template_name})
    
    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API and get response."""
        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self._temperature,
                "num_predict": min(self._max_tokens, 600),  # Limit to prevent repetition
                "repeat_penalty": 1.5,  # Higher penalty for repetition
                "repeat_last_n": 128,  # Larger context window for repeat detection
            }
        }
        
        try:
            self.logger.info(f"Calling Ollama with model {self._model}...")
            response = self._session.post(
                f"{self._base_url}/api/generate",
                json=payload,
                timeout=self._timeout
            )
            response.raise_for_status()
            
            result = response.json()
            raw_response = result.get('response', '')
            
            # Post-process to remove duplicates
            cleaned_response = self._remove_duplicate_sections(raw_response)
            
            return cleaned_response
            
        except Exception as e:
            self.logger.error(f"Ollama API call failed: {e}")
            raise LLMConnectionError(f"Ollama API error: {e}")
    
    def _remove_duplicate_sections(self, text: str) -> str:
        """Remove duplicate sections from LLM output."""
        if not text:
            return text
        
        lines = text.split('\n')
        seen_lines = set()
        unique_lines = []
        
        for line in lines:
            # Normalize line for comparison
            normalized = line.strip().lower()
            
            # Skip empty lines (allow them)
            if not normalized:
                unique_lines.append(line)
                continue
            
            # Skip if we've seen this exact line
            if normalized in seen_lines:
                continue
            
            seen_lines.add(normalized)
            unique_lines.append(line)
        
        return '\n'.join(unique_lines)
    
    def validate_response(self, response: str) -> bool:
        """
        Validate that generated response meets safety requirements.
        
        Checks for:
        - Absence of diagnostic language
        - Absence of treatment recommendations
        - Absence of dosage prescriptions
        - Presence of appropriate disclaimers
        """
        response_lower = response.lower()
        
        # Check for prohibited phrases
        for phrase in self.PROHIBITED_PHRASES:
            if phrase in response_lower:
                self.logger.warning(f"Response contains prohibited phrase: {phrase}")
                return False
        
        # Check minimum length
        if len(response) < 50:
            return False
        
        return True
    
    def _sanitize_response(self, response: str) -> str:
        """Sanitize response by adding warnings if needed."""
        disclaimer = "\n\n⚠️ **ÖNEMLİ:** Bu bilgiler yalnızca genel bilgilendirme amaçlıdır ve profesyonel tıbbi tavsiye yerine geçmez. İlaç kullanmadan önce mutlaka doktorunuza veya eczacınıza danışın."
        
        if "⚠️" not in response:
            response += disclaimer
        
        return response
    
    @property
    def model_name(self) -> str:
        """Get the name of the LLM model."""
        return self._model
    
    @property
    def max_context_length(self) -> int:
        """Get the maximum context length supported."""
        # Qwen3:4b supports up to 32k context
        context_lengths = {
            "qwen3": 32768,
            "qwen2.5": 32768,
            "llama": 8192,
            "gemma": 8192,
            "mistral": 8192,
            "phi": 4096,
        }
        
        model_base = self._model.split(':')[0].lower()
        for key, length in context_lengths.items():
            if key in model_base:
                return length
        
        return 4096  # Default
    
    @property
    def available_templates(self) -> list:
        """Get list of available response templates."""
        return list(TEMPLATES.keys())


class OllamaStreamingGenerator(OllamaResponseGenerator):
    """
    Streaming version of Ollama generator.
    
    Yields response tokens as they are generated for
    real-time display in UI applications.
    """
    
    def generate_stream(
        self,
        drug_info: DrugInfo,
        knowledge: KnowledgeRetrievalResult,
        options: Optional[Dict[str, Any]] = None
    ):
        """
        Generate response with streaming.
        
        Yields:
            str: Response tokens as they are generated
        """
        if not self._session:
            self._init_session()
        
        if not self._check_ollama_available():
            raise LLMConnectionError(f"Ollama not available")
        
        prompt = self._build_prompt(drug_info, knowledge)
        
        # Streaming request
        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": self._temperature,
                "num_predict": self._max_tokens,
            }
        }
        
        try:
            response = self._session.post(
                f"{self._base_url}/api/generate",
                json=payload,
                stream=True,
                timeout=self._timeout
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if token := data.get('response'):
                        yield token
                    
                    if data.get('done', False):
                        break
                        
        except Exception as e:
            self.logger.error(f"Streaming error: {e}")
            raise ResponseGenerationError(f"Streaming failed: {e}")
