"""
OpenAI Response Generator

LLM response generation using OpenAI GPT models.
"""

from typing import Optional, Dict, Any, List
import logging
import time
import re

from ...domain.ports.response_generator import ResponseGeneratorPort
from ...domain.entities.drug_info import DrugInfo
from ...domain.entities.extraction_result import KnowledgeRetrievalResult
from ...domain.exceptions import (
    ResponseGenerationError,
    LLMConnectionError,
    LLMRateLimitError,
    UnsafeResponseError,
    ContextTooLongError,
)


logger = logging.getLogger(__name__)


# System prompt for pharmaceutical information
SYSTEM_PROMPT = """You are a helpful pharmaceutical information assistant. Your role is to provide 
clear, accurate, and safe information about medications based ONLY on the provided knowledge.

CRITICAL RULES:
1. ONLY use information from the provided knowledge context
2. If information is not in the context, say "I don't have information about this"
3. NEVER provide:
   - Medical diagnoses
   - Treatment recommendations
   - Dosage prescriptions
   - Personalized medical advice
4. Always recommend consulting a healthcare professional
5. Use clear, simple language suitable for general users
6. Include relevant warnings if present in the knowledge

Format your response clearly with:
- Drug name and form
- Active ingredients
- General purpose/uses
- Common usage guidance
- Important warnings (if any)
"""

# Response template
RESPONSE_TEMPLATE = """Based on the identified drug "{drug_name}":

**İlaç Adı (Drug Name):** {drug_name}
{strength_line}

**Etken Madde(ler) (Active Ingredients):** {ingredients}

**Dozaj Formu (Dosage Form):** {dosage_form}

**Genel Kullanım (General Purpose):**
{purpose}

**Kullanım Bilgisi (Usage Guidance):**
{usage}

{warnings_section}

---
⚠️ **ÖNEMLİ UYARI:** Bu bilgiler yalnızca eğitim amaçlıdır ve profesyonel tıbbi tavsiye, 
teşhis veya tedavi yerine geçmez. Herhangi bir ilaç kullanmadan önce mutlaka doktorunuza 
veya eczacınıza danışınız.
"""


class OpenAIResponseGenerator(ResponseGeneratorPort):
    """
    Response generator implementation using OpenAI GPT models.
    
    Generates user-friendly, safe explanations about identified drugs
    based on retrieved knowledge.
    
    Attributes:
        api_key: OpenAI API key
        model: GPT model to use
        temperature: Response creativity (0-1)
        max_tokens: Maximum response length
    """
    
    # Unsafe patterns to check in responses
    UNSAFE_PATTERNS = [
        r"you\s+should\s+take",
        r"take\s+\d+\s*(mg|pill|tablet|capsule)",
        r"I\s+diagnose",
        r"you\s+have\s+(?:a\s+)?(?:disease|condition|illness)",
        r"increase\s+(?:your\s+)?(?:dose|dosage)",
        r"stop\s+taking\s+(?:your\s+)?(?:medication|medicine)",
        r"start\s+taking",
        r"prescribe",
    ]
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4",
        temperature: float = 0.3,
        max_tokens: int = 1000,
        language: str = "tr"
    ):
        """
        Initialize OpenAI generator.
        
        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            model: GPT model name
            temperature: Response temperature
            max_tokens: Maximum response tokens
            language: Response language
        """
        self._api_key = api_key
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._language = language
        self._client = None
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def _initialize(self) -> None:
        """Lazy initialization of OpenAI client."""
        if self._client is not None:
            return
        
        try:
            from openai import OpenAI
            import os
            
            api_key = self._api_key or os.environ.get("OPENAI_API_KEY")
            
            if not api_key:
                raise LLMConnectionError(
                    "OpenAI API key not provided. Set OPENAI_API_KEY or pass api_key parameter.",
                    provider="OpenAI"
                )
            
            self._client = OpenAI(api_key=api_key)
            self.logger.info(f"OpenAI client initialized with model={self._model}")
            
        except ImportError:
            raise LLMConnectionError(
                "openai package not installed. Install with: pip install openai",
                provider="OpenAI"
            )
        except Exception as e:
            raise LLMConnectionError(f"Failed to initialize OpenAI: {e}", provider="OpenAI")
    
    def generate(
        self,
        drug_info: DrugInfo,
        knowledge: KnowledgeRetrievalResult,
        options: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate user-friendly explanation about a drug.
        
        Args:
            drug_info: Identified drug information
            knowledge: Retrieved pharmaceutical knowledge
            options: Optional configuration
                - template: Response template name
                - language: Override response language
                
        Returns:
            Generated explanation text
        """
        start_time = time.time()
        options = options or {}
        
        # Check if we should use template-based generation (doesn't require LLM)
        if options.get("use_template_only", False) or not knowledge.has_knowledge:
            return self._generate_from_template(drug_info, knowledge)
        
        # Initialize OpenAI client
        self._initialize()
        
        # Build prompt
        user_prompt = self._build_user_prompt(drug_info, knowledge)
        
        # Check context length
        estimated_tokens = len(user_prompt.split()) * 1.5
        if estimated_tokens > self.max_context_length * 0.8:
            # Truncate knowledge
            self.logger.warning("Context too long, truncating knowledge")
            knowledge = self._truncate_knowledge(knowledge)
            user_prompt = self._build_user_prompt(drug_info, knowledge)
        
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self._temperature,
                max_tokens=self._max_tokens
            )
            
            generated_text = response.choices[0].message.content
            
            # Validate response
            if not self.validate_response(generated_text):
                self.logger.warning("Generated response failed safety validation, using template")
                return self._generate_from_template(drug_info, knowledge)
            
            return generated_text
            
        except Exception as e:
            error_str = str(e).lower()
            
            if "rate_limit" in error_str:
                raise LLMRateLimitError(message=str(e))
            elif "context_length" in error_str:
                raise ContextTooLongError(
                    context_length=int(estimated_tokens),
                    max_length=self.max_context_length
                )
            else:
                self.logger.warning(f"OpenAI API error: {e}, falling back to template")
                return self._generate_from_template(drug_info, knowledge)
    
    def _build_user_prompt(
        self,
        drug_info: DrugInfo,
        knowledge: KnowledgeRetrievalResult
    ) -> str:
        """Build the user prompt for the LLM."""
        prompt_parts = [
            f"Drug identified: {drug_info.drug_name}",
        ]
        
        if drug_info.active_ingredients:
            prompt_parts.append(f"Active ingredients: {', '.join(drug_info.active_ingredients)}")
        
        if drug_info.dosage_form:
            prompt_parts.append(f"Dosage form: {drug_info.dosage_form.value}")
        
        if drug_info.strength:
            prompt_parts.append(f"Strength: {drug_info.strength}")
        
        prompt_parts.append("\n--- Knowledge Context ---")
        
        if knowledge.has_knowledge:
            for chunk in knowledge.chunks[:3]:  # Limit to top 3 chunks
                prompt_parts.append(f"\n[Source: {chunk.source}]\n{chunk.content}")
        else:
            prompt_parts.append("\nNo additional knowledge available.")
        
        prompt_parts.append("\n--- End Context ---")
        prompt_parts.append(
            f"\nProvide a clear, safe explanation about this drug in {self._language}. "
            "Include what you know from the context about its uses and any important warnings."
        )
        
        return "\n".join(prompt_parts)
    
    def _generate_from_template(
        self,
        drug_info: DrugInfo,
        knowledge: KnowledgeRetrievalResult
    ) -> str:
        """Generate response using template without LLM."""
        # Extract information
        ingredients = drug_info.get_active_ingredients_string()
        dosage_form = drug_info.dosage_form.value if drug_info.dosage_form else "Bilinmiyor"
        
        strength_line = ""
        if drug_info.strength:
            strength_line = f"**Doz (Strength):** {drug_info.strength}\n"
        
        # Get purpose and usage from knowledge
        purpose = "Bu ilaç hakkında detaylı bilgi mevcut değil."
        usage = "Kullanım talimatları için doktorunuza veya eczacınıza danışınız."
        
        if knowledge.has_knowledge:
            combined = knowledge.combined_knowledge
            if combined:
                purpose = combined[:500] + "..." if len(combined) > 500 else combined
                usage = "Bu ilacı doktorunuzun önerdiği şekilde kullanınız."
        
        # Warnings section
        warnings_section = ""
        if drug_info.confidence.requires_warning:
            warnings_section = (
                "⚠️ **Dikkat:** Otomatik tanıma güvenilirliği düşük. "
                "İlaç bilgilerini doğrulamak için ambalajı kontrol edin."
            )
        
        return RESPONSE_TEMPLATE.format(
            drug_name=drug_info.drug_name,
            strength_line=strength_line,
            ingredients=ingredients,
            dosage_form=dosage_form.capitalize(),
            purpose=purpose,
            usage=usage,
            warnings_section=warnings_section
        )
    
    def _truncate_knowledge(
        self,
        knowledge: KnowledgeRetrievalResult
    ) -> KnowledgeRetrievalResult:
        """Truncate knowledge to fit context window."""
        from ...domain.entities.extraction_result import KnowledgeChunk
        
        truncated_chunks = []
        max_chunk_length = 500
        
        for chunk in knowledge.chunks[:2]:  # Take only top 2
            content = chunk.content
            if len(content) > max_chunk_length:
                content = content[:max_chunk_length] + "..."
            
            truncated_chunks.append(KnowledgeChunk(
                content=content,
                source=chunk.source,
                relevance_score=chunk.relevance_score,
                metadata=chunk.metadata
            ))
        
        return KnowledgeRetrievalResult(
            chunks=truncated_chunks,
            query_used=knowledge.query_used,
            total_chunks_searched=knowledge.total_chunks_searched,
            processing_time_ms=knowledge.processing_time_ms
        )
    
    def generate_with_template(
        self,
        drug_info: DrugInfo,
        knowledge: KnowledgeRetrievalResult,
        template_name: str
    ) -> str:
        """Generate using a specific template."""
        # For now, all templates use the same format
        return self._generate_from_template(drug_info, knowledge)
    
    def validate_response(self, response: str) -> bool:
        """
        Validate that response meets safety requirements.
        
        Args:
            response: Generated response text
            
        Returns:
            True if response is safe
        """
        response_lower = response.lower()
        
        for pattern in self.UNSAFE_PATTERNS:
            if re.search(pattern, response_lower, re.IGNORECASE):
                self.logger.warning(f"Unsafe pattern detected: {pattern}")
                return False
        
        return True
    
    @property
    def model_name(self) -> str:
        return self._model
    
    @property
    def max_context_length(self) -> int:
        # Context lengths for different models
        context_lengths = {
            "gpt-4": 8192,
            "gpt-4-turbo": 128000,
            "gpt-4-turbo-preview": 128000,
            "gpt-3.5-turbo": 4096,
            "gpt-3.5-turbo-16k": 16384,
        }
        return context_lengths.get(self._model, 4096)


class DummyResponseGenerator(ResponseGeneratorPort):
    """
    Dummy response generator for testing.
    """
    
    def generate(
        self,
        drug_info: DrugInfo,
        knowledge: KnowledgeRetrievalResult,
        options: Optional[Dict[str, Any]] = None
    ) -> str:
        return f"""**{drug_info.drug_name}** Bilgileri:

Bu bir test yanıtıdır. Gerçek uygulamada, bu ilaç hakkında detaylı bilgi gösterilecektir.

⚠️ **UYARI:** Bu bilgiler yalnızca eğitim amaçlıdır. Doktorunuza danışmadan ilaç kullanmayınız.
"""
    
    def generate_with_template(
        self,
        drug_info: DrugInfo,
        knowledge: KnowledgeRetrievalResult,
        template_name: str
    ) -> str:
        return self.generate(drug_info, knowledge)
    
    def validate_response(self, response: str) -> bool:
        return True
    
    @property
    def model_name(self) -> str:
        return "DummyLLM"
    
    @property
    def max_context_length(self) -> int:
        return 10000
