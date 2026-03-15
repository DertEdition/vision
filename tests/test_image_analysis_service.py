"""
Test Image Analysis Service - TC-IMG-01 through TC-IMG-15

Tests from CARE_Test_Plan_Report.docx Section 6.3: Image Analysis Service
All pipeline dependencies are mocked to run without external services.
"""

import sys
import os
import time
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

# Ensure project root is on path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.domain.value_objects.image_data import ImageData
from src.domain.value_objects.confidence_score import ConfidenceScore
from src.domain.entities.pipeline_result import (
    PipelineResult, PipelineStage, PipelineError, StageStatus,
)
from src.domain.entities.extraction_result import (
    VisionAnalysisResult, DetectedObject, DetectionClass,
    TextExtractionResult, TextBlock,
    EntityExtractionResult, ExtractedEntity, EntityType,
    KnowledgeRetrievalResult, KnowledgeChunk,
)
from src.domain.entities.drug_info import DrugInfo
from src.domain.value_objects.bounding_box import BoundingBox
from src.domain.value_objects.dosage_info import DosageForm, DosageInfo
from src.domain.exceptions import (
    InvalidImageError, NoPharmaceuticalContentError,
    NoTextFoundError, DrugNameNotFoundError,
    KnowledgeBaseConnectionError, UnsafeResponseError,
    ImageQualityError, PipelineTimeoutError,
)
from src.application.pipeline.orchestrator import PipelineOrchestrator, PipelineConfig
from src.application.pipeline.stages import StageConfig
from src.application.services.drug_analysis_service import DrugAnalysisService


# ─────────────────────────────────────────────────────────────
# Helpers: mock port factories
# ─────────────────────────────────────────────────────────────

def _make_vision_analyzer(
    is_pharma: bool = True,
    quality: float = 0.9,
    detections: list | None = None,
    side_effect=None,
):
    """Create a mock VisionAnalyzerPort."""
    mock = MagicMock()
    mock.model_name = "mock_yolo"

    if side_effect:
        mock.analyze.side_effect = side_effect
    else:
        result = VisionAnalysisResult(
            detected_objects=detections or [
                DetectedObject(
                    detection_class=DetectionClass.DRUG_BOX,
                    bounding_box=BoundingBox(x_min=0.0, y_min=0.0, x_max=0.5, y_max=0.5),
                    confidence=ConfidenceScore(value=0.95),
                ),
            ],
            image_quality_score=ConfidenceScore(value=quality),
            is_pharmaceutical_image=is_pharma,
        )
        mock.analyze.return_value = result
        mock.is_pharmaceutical_image.return_value = is_pharma

    return mock


def _make_text_extractor(
    text: str = "Parol 500 mg Tablet\nParasetamol 500 mg",
    has_text: bool = True,
    side_effect=None,
):
    """Create a mock TextExtractorPort."""
    mock = MagicMock()
    mock.engine_name = "mock_ocr"
    mock.supported_languages = ["tr", "en"]

    if side_effect:
        mock.extract.side_effect = side_effect
    else:
        blocks = (
            [TextBlock(text=text, confidence=ConfidenceScore(value=0.9))]
            if has_text
            else []
        )
        result = TextExtractionResult(
            text_blocks=blocks,
            full_text=text if has_text else "",
        )
        mock.extract.return_value = result

    return mock


def _make_entity_extractor(
    drug_name: str | None = "Parol",
    ingredients: list | None = None,
    dosage_form: str | None = "tablet",
    strength: str | None = "500 mg",
    manufacturer: str | None = "Atabay",
    side_effect=None,
):
    """Create a mock EntityExtractorPort."""
    mock = MagicMock()
    mock.extractor_name = "mock_entity"

    if side_effect:
        mock.extract.side_effect = side_effect
    else:
        entities = []
        if drug_name:
            entities.append(
                ExtractedEntity(
                    entity_type=EntityType.DRUG_NAME,
                    value=drug_name,
                    confidence=ConfidenceScore(value=0.95),
                )
            )
        result = EntityExtractionResult(
            entities=entities,
            drug_name=drug_name,
            active_ingredients=ingredients or ["Parasetamol"],
            dosage_form=dosage_form,
            strength=strength,
            manufacturer=manufacturer,
        )
        mock.extract.return_value = result

    return mock


def _make_knowledge_retriever(
    has_knowledge: bool = True,
    side_effect=None,
):
    """Create a mock KnowledgeRetrieverPort."""
    mock = MagicMock()
    mock.retriever_name = "mock_chroma"
    mock.knowledge_base_size = 100

    if side_effect:
        mock.retrieve.side_effect = side_effect
    else:
        chunks = (
            [
                KnowledgeChunk(
                    content="Parol (Parasetamol) ağrı ve ateş kesici olarak kullanılır.",
                    source="pharma_db",
                    relevance_score=0.92,
                )
            ]
            if has_knowledge
            else []
        )
        result = KnowledgeRetrievalResult(chunks=chunks, query_used="Parol")
        mock.retrieve.return_value = result

    return mock


def _make_response_generator(
    response: str = "Parol, parasetamol içeren bir ağrı kesicidir.",
    validate: bool = True,
    side_effect=None,
):
    """Create a mock ResponseGeneratorPort."""
    mock = MagicMock()
    mock.model_name = "mock_llm"
    mock.max_context_length = 4096

    if side_effect:
        mock.generate.side_effect = side_effect
    else:
        mock.generate.return_value = response

    mock.validate_response.return_value = validate

    return mock


def _build_pipeline(**overrides):
    """Build a PipelineOrchestrator with default mocks, applying any overrides."""
    defaults = {
        "vision_analyzer": _make_vision_analyzer(),
        "text_extractor": _make_text_extractor(),
        "entity_extractor": _make_entity_extractor(),
        "knowledge_retriever": _make_knowledge_retriever(),
        "response_generator": _make_response_generator(),
    }
    defaults.update(overrides)

    config = overrides.pop("config", None) or PipelineConfig(
        timeout_seconds=120,
        fail_fast=False,
        stages={
            PipelineStage.VISION_ANALYSIS: StageConfig(retry_count=0, retry_delay_seconds=0),
            PipelineStage.TEXT_EXTRACTION: StageConfig(retry_count=0, retry_delay_seconds=0),
            PipelineStage.ENTITY_EXTRACTION: StageConfig(retry_count=0, retry_delay_seconds=0),
            PipelineStage.KNOWLEDGE_RETRIEVAL: StageConfig(retry_count=0, retry_delay_seconds=0),
            PipelineStage.RESPONSE_GENERATION: StageConfig(retry_count=0, retry_delay_seconds=0),
        },
    )

    return PipelineOrchestrator(
        vision_analyzer=defaults["vision_analyzer"],
        text_extractor=defaults["text_extractor"],
        entity_extractor=defaults["entity_extractor"],
        knowledge_retriever=defaults["knowledge_retriever"],
        response_generator=defaults["response_generator"],
        config=config,
    )


def _make_valid_image() -> ImageData:
    """Create a minimal valid JPEG-like ImageData for testing."""
    # Minimal JPEG header bytes
    jpeg_header = (
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
        + b"\x00" * 100
    )
    return ImageData.from_bytes(jpeg_header, format="jpeg")


# ═════════════════════════════════════════════════════════════
# TC-IMG-01 : Drug packaging image analyzed successfully
# ═════════════════════════════════════════════════════════════

class TestTCIMG01:
    """TC-IMG-01: Drug packaging image analyzed successfully.

    Input: Clear image of a drug box
    Expected: Drug name, active ingredients, and dosage information extracted correctly
    Method: System
    """

    def test_successful_drug_analysis(self):
        pipeline = _build_pipeline()
        image = _make_valid_image()
        result = pipeline.run(image)

        assert result.is_successful, "Pipeline should succeed for a clear drug image"
        assert result.drug_info is not None, "Drug info should be populated"
        assert result.drug_info.drug_name == "Parol"
        assert "Parasetamol" in result.drug_info.active_ingredients
        assert result.explanation != ""
        assert result.disclaimer != ""
        print(f"  ✓ Drug name: {result.drug_info.drug_name}")
        print(f"  ✓ Active ingredients: {result.drug_info.active_ingredients}")
        print(f"  ✓ Explanation present: {bool(result.explanation)}")


# ═════════════════════════════════════════════════════════════
# TC-IMG-02 : Non-pharmaceutical image handled correctly
# ═════════════════════════════════════════════════════════════

class TestTCIMG02:
    """TC-IMG-02: Non-pharmaceutical image handled correctly.

    Input: Image of a non-pharmaceutical object
    Expected: 'No pharmaceutical content detected' message returned
    Method: Integration
    """

    def test_non_pharma_image_detected(self):
        pipeline = _build_pipeline(
            vision_analyzer=_make_vision_analyzer(is_pharma=False, detections=[]),
        )
        image = _make_valid_image()
        result = pipeline.run(image)

        assert not result.is_successful, "Pipeline should not succeed for non-pharma image"
        error_messages = [e.message for e in result.errors]
        has_no_pharma = any(
            "pharmaceutical" in m.lower() or "NoPharmaceuticalContentError" in e.error_type
            for e in result.errors
            for m in [e.message]
        )
        assert has_no_pharma, f"Should have 'no pharmaceutical content' error, got: {error_messages}"
        print(f"  ✓ Detected non-pharmaceutical content, errors: {error_messages}")


# ═════════════════════════════════════════════════════════════
# TC-IMG-03 : Unsupported image format rejected
# ═════════════════════════════════════════════════════════════

class TestTCIMG03:
    """TC-IMG-03: Unsupported image format rejected.

    Input: BMP or TIFF file upload
    Expected: 415 Unsupported Media Type – UnsupportedFormatException
    Method: Integration
    """

    def test_unsupported_tiff_format(self, tmp_path):
        pipeline = _build_pipeline()
        service = DrugAnalysisService(pipeline)

        # Create a fake .tiff file
        tiff_file = tmp_path / "image.tiff"
        tiff_file.write_bytes(b"\x49\x49\x2a\x00" + b"\x00" * 100)

        with pytest.raises(InvalidImageError) as exc_info:
            service.analyze_from_file(str(tiff_file))

        assert "Unsupported image format" in str(exc_info.value) or "unsupported" in str(exc_info.value).lower()
        print(f"  ✓ TIFF format rejected: {exc_info.value}")

    def test_unsupported_ext_txt(self, tmp_path):
        pipeline = _build_pipeline()
        service = DrugAnalysisService(pipeline)

        txt_file = tmp_path / "image.txt"
        txt_file.write_bytes(b"not an image")

        with pytest.raises(InvalidImageError):
            service.analyze_from_file(str(txt_file))
        print("  ✓ .txt extension correctly rejected")


# ═════════════════════════════════════════════════════════════
# TC-IMG-04 : Low-resolution image rejected
# ═════════════════════════════════════════════════════════════

class TestTCIMG04:
    """TC-IMG-04: Low-resolution image rejected.

    Input: Image smaller than 50x50 pixels
    Expected: 400 Bad Request – ImageResolutionException
    Method: Integration
    """

    def test_low_resolution_image(self):
        pipeline = _build_pipeline(
            vision_analyzer=_make_vision_analyzer(
                side_effect=ImageQualityError(
                    "Image resolution too low (10x10 < 50x50 minimum)",
                    quality_score=0.1,
                )
            ),
        )
        image = ImageData.from_bytes(b"\x00" * 30, format="jpeg", source="low_res.jpg")
        result = pipeline.run(image)

        assert result.has_errors, "Should have errors for low-resolution image"
        error_types = [e.error_type for e in result.errors]
        assert any("ImageQualityError" in t or "Quality" in t for t in error_types), \
            f"Should have image quality error, got: {error_types}"
        print(f"  ✓ Low-resolution image rejected, error types: {error_types}")


# ═════════════════════════════════════════════════════════════
# TC-IMG-05 : Corrupted image file rejected
# ═════════════════════════════════════════════════════════════

class TestTCIMG05:
    """TC-IMG-05: Corrupted image file rejected.

    Input: Unreadable or corrupted image file
    Expected: 400 Bad Request – InvalidImageException
    Method: Integration
    """

    def test_corrupted_image_bytes(self):
        pipeline = _build_pipeline()
        service = DrugAnalysisService(pipeline)

        with pytest.raises(InvalidImageError):
            service.analyze_from_bytes(b"")  # empty bytes → invalid

        print("  ✓ Empty/corrupted bytes correctly rejected")

    def test_corrupted_base64(self):
        pipeline = _build_pipeline()
        service = DrugAnalysisService(pipeline)

        with pytest.raises(InvalidImageError):
            service.analyze_from_base64("")  # empty base64 → invalid

        print("  ✓ Empty base64 correctly rejected")


# ═════════════════════════════════════════════════════════════
# TC-IMG-06 : OCR text extraction from readable drug label
# ═════════════════════════════════════════════════════════════

class TestTCIMG06:
    """TC-IMG-06: OCR text extraction from readable drug label.

    Input: Drug packaging image with readable text
    Expected: Text blocks extracted and full text field populated
    Method: Unit
    """

    def test_ocr_text_extraction(self):
        mock_text = "Parol 500 mg Film Tablet\nHer tablette 500 mg Parasetamol bulunur"
        pipeline = _build_pipeline(
            text_extractor=_make_text_extractor(text=mock_text),
        )
        image = _make_valid_image()
        result = pipeline.run(image)

        assert result.text_result is not None, "Text result should be populated"
        assert result.text_result.has_text, "Should have extracted text"
        assert "Parol" in result.text_result.full_text
        assert "Parasetamol" in result.text_result.full_text
        print(f"  ✓ OCR extracted: {result.text_result.full_text[:80]}...")


# ═════════════════════════════════════════════════════════════
# TC-IMG-07 : OCR returns no text but pipeline continues
# ═════════════════════════════════════════════════════════════

class TestTCIMG07:
    """TC-IMG-07: OCR returns no text but pipeline continues.

    Input: Drug package image with unreadable text
    Expected: 200 OK (partial) – warning returned; OCR result empty
    Method: Integration
    """

    def test_no_text_pipeline_continues(self):
        pipeline = _build_pipeline(
            text_extractor=_make_text_extractor(
                side_effect=NoTextFoundError("No readable text found")
            ),
        )
        image = _make_valid_image()
        result = pipeline.run(image)

        # Pipeline should have continued (fail_soft default) but with errors
        has_text_error = any(
            "NoTextFoundError" in e.error_type or "text" in e.message.lower()
            for e in result.errors
        )
        assert has_text_error, "Should have text extraction error"
        print(f"  ✓ Pipeline continued with OCR failure, errors: {[e.error_type for e in result.errors]}")


# ═════════════════════════════════════════════════════════════
# TC-IMG-08 : Entity extraction identifies drug information
# ═════════════════════════════════════════════════════════════

class TestTCIMG08:
    """TC-IMG-08: Entity extraction identifies drug information.

    Input: OCR output containing drug name, ingredient, and dosage
    Expected: Drug entities extracted into structured fields correctly
    Method: Unit
    """

    def test_entity_extraction(self):
        pipeline = _build_pipeline(
            entity_extractor=_make_entity_extractor(
                drug_name="Augmentin",
                ingredients=["Amoksisilin", "Klavulanik asit"],
                dosage_form="tablet",
                strength="625 mg",
                manufacturer="GlaxoSmithKline",
            ),
        )
        image = _make_valid_image()
        result = pipeline.run(image)

        assert result.drug_info is not None
        assert result.drug_info.drug_name == "Augmentin"
        assert "Amoksisilin" in result.drug_info.active_ingredients
        assert "Klavulanik asit" in result.drug_info.active_ingredients
        print(f"  ✓ Drug: {result.drug_info.drug_name}, ingredients: {result.drug_info.active_ingredients}")


# ═════════════════════════════════════════════════════════════
# TC-IMG-09 : No drug name identified handled with low confidence
# ═════════════════════════════════════════════════════════════

class TestTCIMG09:
    """TC-IMG-09: No drug name identified handled with low confidence.

    Input: Partial packaging image with incomplete label
    Expected: 200 OK (partial) – drugName=unknown with low confidence warning
    Method: Integration
    """

    def test_no_drug_name_low_confidence(self):
        pipeline = _build_pipeline(
            entity_extractor=_make_entity_extractor(
                side_effect=DrugNameNotFoundError(
                    extracted_text="partial text without clear drug name"
                )
            ),
        )
        image = _make_valid_image()
        result = pipeline.run(image)

        has_entity_error = any(
            "DrugNameNotFoundError" in e.error_type or "drug name" in e.message.lower()
            for e in result.errors
        )
        assert has_entity_error, "Should have drug name not found error"
        # Drug info should be None or unknown
        assert result.drug_info is None or result.drug_info.drug_name == "Unknown Drug"
        print(f"  ✓ No drug name → appropriate error, drug_info={result.drug_info}")


# ═════════════════════════════════════════════════════════════
# TC-IMG-10 : Knowledge retrieval succeeds with recognized drug
# ═════════════════════════════════════════════════════════════

class TestTCIMG10:
    """TC-IMG-10: Knowledge retrieval succeeds with recognized drug.

    Input: Valid extracted drug name
    Expected: Relevant supporting information retrieved from ChromaDB
    Method: Integration
    """

    def test_knowledge_retrieval_success(self):
        pipeline = _build_pipeline(
            knowledge_retriever=_make_knowledge_retriever(has_knowledge=True),
        )
        image = _make_valid_image()
        result = pipeline.run(image)

        assert result.knowledge_result is not None
        assert result.knowledge_result.has_knowledge
        assert len(result.knowledge_result.chunks) > 0
        print(f"  ✓ Knowledge retrieved: {len(result.knowledge_result.chunks)} chunks")


# ═════════════════════════════════════════════════════════════
# TC-IMG-11 : Knowledge retrieval database failure handled
# ═════════════════════════════════════════════════════════════

class TestTCIMG11:
    """TC-IMG-11: Knowledge retrieval database failure handled.

    Input: Valid extracted drug name while ChromaDB is unavailable
    Expected: 503 Service Unavailable – DatabaseConnectionException
    Method: Integration
    """

    def test_knowledge_db_failure(self):
        pipeline = _build_pipeline(
            knowledge_retriever=_make_knowledge_retriever(
                side_effect=KnowledgeBaseConnectionError(
                    "Failed to connect to ChromaDB"
                )
            ),
        )
        image = _make_valid_image()
        result = pipeline.run(image)

        has_kb_error = any(
            "KnowledgeBaseConnectionError" in e.error_type
            or "connect" in e.message.lower()
            or "database" in e.message.lower()
            for e in result.errors
        )
        assert has_kb_error, f"Should have knowledge base connection error, got: {[e.error_type for e in result.errors]}"
        print(f"  ✓ ChromaDB failure handled, errors: {[e.error_type for e in result.errors]}")


# ═════════════════════════════════════════════════════════════
# TC-IMG-12 : Patient-friendly explanation generated
# ═════════════════════════════════════════════════════════════

class TestTCIMG12:
    """TC-IMG-12: Patient-friendly explanation generated.

    Input: Valid extracted drug info with retrieved context
    Expected: Readable explanation generated and included in pipeline result
    Method: Integration
    """

    def test_explanation_generated(self):
        explanation = (
            "Parol, parasetamol etkin maddesi içeren bir ağrı kesici ve ateş "
            "düşürücü ilaçtır. Yetişkinler için önerilen doz günde 3-4 kez "
            "1 tablettir."
        )
        pipeline = _build_pipeline(
            response_generator=_make_response_generator(response=explanation),
        )
        image = _make_valid_image()
        result = pipeline.run(image)

        assert result.explanation != "", "Explanation should not be empty"
        assert "parasetamol" in result.explanation.lower() or "Parol" in result.explanation
        print(f"  ✓ Explanation generated: {result.explanation[:80]}...")


# ═════════════════════════════════════════════════════════════
# TC-IMG-13 : Invalid LLM explanation falls back to raw data
# ═════════════════════════════════════════════════════════════

class TestTCIMG13:
    """TC-IMG-13: Invalid LLM explanation falls back to raw data.

    Input: Valid extracted info but explanation fails validation
    Expected: 200 OK (degraded) – raw extracted drug data returned without generated explanation
    Method: Integration
    """

    def test_invalid_explanation_fallback(self):
        pipeline = _build_pipeline(
            response_generator=_make_response_generator(
                response="Bu ilacı günde 3 kez alın",  # unsafe response
                validate=False,
            ),
        )
        image = _make_valid_image()
        result = pipeline.run(image)

        # Drug data should still be present even if explanation fails
        assert result.drug_info is not None, "Drug info should still be available"
        assert result.drug_info.drug_name == "Parol"
        # Explanation may be empty or contain error info
        has_response_error = any(
            "UnsafeResponseError" in e.error_type or "unsafe" in e.message.lower()
            for e in result.errors
        )
        assert has_response_error, "Should have unsafe response error"
        print(f"  ✓ Fallback to raw data, drug_name={result.drug_info.drug_name}, errors={[e.error_type for e in result.errors]}")


# ═════════════════════════════════════════════════════════════
# TC-IMG-14 : Pipeline timeout returns partial result
# ═════════════════════════════════════════════════════════════

class TestTCIMG14:
    """TC-IMG-14: Pipeline timeout returns partial result.

    Input: Valid image causing one pipeline stage to exceed timeout
    Expected: 504 Gateway Timeout – partial result returned with error list
    Method: System
    """

    def test_pipeline_timeout(self):
        def slow_analyze(*args, **kwargs):
            time.sleep(0.5)
            return VisionAnalysisResult(
                detected_objects=[],
                is_pharmaceutical_image=True,
            )

        slow_vision = MagicMock()
        slow_vision.model_name = "slow_yolo"
        slow_vision.analyze.side_effect = slow_analyze

        config = PipelineConfig(
            timeout_seconds=0.01,  # very short timeout
            fail_fast=False,
            stages={
                PipelineStage.VISION_ANALYSIS: StageConfig(retry_count=0, retry_delay_seconds=0),
                PipelineStage.TEXT_EXTRACTION: StageConfig(retry_count=0, retry_delay_seconds=0),
                PipelineStage.ENTITY_EXTRACTION: StageConfig(retry_count=0, retry_delay_seconds=0),
                PipelineStage.KNOWLEDGE_RETRIEVAL: StageConfig(retry_count=0, retry_delay_seconds=0),
                PipelineStage.RESPONSE_GENERATION: StageConfig(retry_count=0, retry_delay_seconds=0),
            },
        )

        pipeline = PipelineOrchestrator(
            vision_analyzer=slow_vision,
            text_extractor=_make_text_extractor(),
            entity_extractor=_make_entity_extractor(),
            knowledge_retriever=_make_knowledge_retriever(),
            response_generator=_make_response_generator(),
            config=config,
        )

        image = _make_valid_image()
        result = pipeline.run(image)

        # Pipeline should have timed out or produced partial result
        has_timeout_or_partial = (
            result.has_errors
            or not result.is_successful
            or any("timeout" in e.message.lower() or "Timeout" in e.error_type for e in result.errors)
        )
        assert has_timeout_or_partial, "Should have timeout or partial result"
        print(f"  ✓ Timeout handled, errors={[e.error_type for e in result.errors]}, successful={result.is_successful}")


# ═════════════════════════════════════════════════════════════
# TC-IMG-15 : User evaluates clarity of image analysis result
# ═════════════════════════════════════════════════════════════

class TestTCIMG15:
    """TC-IMG-15: User evaluates clarity of image analysis result screen.

    Input: Successful analysis viewed from UI
    Expected: User can understand extracted drug information, warnings, and explanation content
    Method: User (adapted to automated: verify get_user_response() structure)
    """

    def test_user_response_clarity(self):
        pipeline = _build_pipeline()
        image = _make_valid_image()
        result = pipeline.run(image)

        user_response = result.get_user_response()

        # Verify all required fields for a clear user-facing response
        assert "success" in user_response, "Response should have 'success' field"
        assert "disclaimer" in user_response, "Response should have 'disclaimer' field"
        assert "drug" in user_response, "Response should have 'drug' object"
        assert "explanation" in user_response, "Response should have 'explanation' field"
        assert "confidence" in user_response, "Response should have 'confidence' field"

        # Verify drug sub-fields
        drug = user_response["drug"]
        assert "name" in drug, "Drug should have 'name'"
        assert "active_ingredients" in drug, "Drug should have 'active_ingredients'"

        # Verify disclaimer is non-empty
        assert len(user_response["disclaimer"]) > 20, "Disclaimer should be substantial"

        print(f"  ✓ User response keys: {list(user_response.keys())}")
        print(f"  ✓ Drug fields: {list(drug.keys())}")
        print(f"  ✓ Disclaimer length: {len(user_response['disclaimer'])} chars")
