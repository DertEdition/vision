"""
Drug Image Pipeline - Main Entry Point

Example usage and CLI interface for the drug image analysis pipeline.
"""

import argparse
import sys
import logging
from pathlib import Path

# Configure UTF-8 encoding for Windows terminal
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from src.cross_cutting.logging import setup_logging
from src.application.pipeline import PipelineBuilder
from src.application.services import DrugAnalysisService
from src.infrastructure.vision import VisionAnalyzerFactory, VisionAnalyzerType
from src.infrastructure.ocr import OCRFactory, OCRType
from src.infrastructure.entity_extraction import EntityExtractorFactory, EntityExtractorType
from src.infrastructure.rag import KnowledgeRetrieverFactory, KnowledgeRetrieverType
from src.infrastructure.llm import LLMFactory, LLMType
from config import get_default_config


def create_pipeline(config=None, use_dummy=False):
    """
    Create a configured pipeline.
    
    Args:
        config: Optional configuration object
        use_dummy: Use dummy implementations for testing
        
    Returns:
        Configured PipelineOrchestrator
    """
    if config is None:
        config = get_default_config()
    
    builder = PipelineBuilder()
    
    if use_dummy:
        # Use dummy implementations for testing
        from src.infrastructure.vision.yolo_analyzer import DummyVisionAnalyzer
        from src.infrastructure.ocr.paddle_ocr import DummyOCRExtractor
        from src.infrastructure.entity_extraction.hybrid_extractor import DummyEntityExtractor
        from src.infrastructure.rag.chroma_retriever import DummyKnowledgeRetriever
        from src.infrastructure.llm.openai_generator import DummyResponseGenerator
        
        builder.with_vision_analyzer(DummyVisionAnalyzer())
        builder.with_text_extractor(DummyOCRExtractor())
        builder.with_entity_extractor(DummyEntityExtractor())
        builder.with_knowledge_retriever(DummyKnowledgeRetriever())
        builder.with_response_generator(DummyResponseGenerator())
    else:
        # Use real implementations
        builder.with_vision_analyzer(
            VisionAnalyzerFactory.create(
                VisionAnalyzerType(config.vision.type),
                model_path=config.vision.model_path,
                confidence_threshold=config.vision.confidence_threshold,
                device=config.vision.device
            )
        )
        
        builder.with_text_extractor(
            OCRFactory.create(
                OCRType(config.ocr.type),
                lang=config.ocr.language,
                use_gpu=config.ocr.use_gpu
            )
        )
        
        builder.with_entity_extractor(
            EntityExtractorFactory.create(
                EntityExtractorType(config.entity_extraction.type),
                use_llm_refinement=config.entity_extraction.use_llm_refinement
            )
        )
        
        builder.with_knowledge_retriever(
            KnowledgeRetrieverFactory.create(
                KnowledgeRetrieverType(config.rag.type),
                persist_directory=config.rag.persist_directory,
                collection_name=config.rag.collection_name
            )
        )
        
        builder.with_response_generator(
            LLMFactory.create(
                LLMType(config.llm.type),
                api_key=config.llm.api_key,
                model=config.llm.model,
                temperature=config.llm.temperature,
                language=config.llm.language
            )
        )
    
    return builder.build()


def analyze_image(image_path: str, use_dummy: bool = False):
    """
    Analyze a drug image and print results.
    
    Args:
        image_path: Path to the drug image
        use_dummy: Use dummy implementations for testing
    """
    # Create pipeline
    pipeline = create_pipeline(use_dummy=use_dummy)
    
    # Create service
    service = DrugAnalysisService(pipeline)
    
    # Analyze
    print(f"\n{'='*60}")
    print(f"Analyzing: {image_path}")
    print(f"{'='*60}\n")
    
    result = service.analyze_from_file(image_path)
    
    # Print results
    if result.is_successful:
        print("[SUCCESS] Analysis Successful\n")
        
        response = result.get_user_response()
        
        if drug := response.get("drug"):
            print(f"Drug Name: {drug.get('name', 'Unknown')}")
            print(f"Active Ingredients: {', '.join(drug.get('active_ingredients', []))}")
            print(f"Dosage Form: {drug.get('dosage_form', 'Unknown')}")
            print(f"Strength: {drug.get('strength', 'Unknown')}")
            print(f"Manufacturer: {drug.get('manufacturer', 'Unknown')}")
        
        print(f"\nConfidence: {response.get('confidence', 'N/A')}")
        
        if explanation := response.get("explanation"):
            print(f"\n--- Explanation ---\n{explanation}")
        
        if warnings := response.get("warnings"):
            print(f"\n[WARNING] Warnings:")
            for warning in warnings:
                print(f"  - {warning}")
        
        print(f"\n{response.get('disclaimer', '')}")
    else:
        print("[FAILED] Analysis Failed\n")
        print(f"Error: {result._get_user_friendly_error()}")
        
        if result.has_errors:
            print("\nDetailed Errors:")
            for error in result.errors:
                print(f"  - [{error.stage.value}] {error.message}")
    
    print(f"\n{'='*60}")
    print(f"Total Processing Time: {result.total_processing_time_ms:.2f}ms")
    print(f"{'='*60}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Drug Image Understanding & Information Pipeline"
    )
    
    parser.add_argument(
        "image",
        nargs="?",
        help="Path to drug image to analyze"
    )
    
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="Use dummy implementations for testing"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="Drug Image Pipeline v1.0.0"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(level=log_level)
    
    if args.image:
        # Analyze the provided image
        if not Path(args.image).exists():
            print(f"Error: Image file not found: {args.image}")
            sys.exit(1)
        
        analyze_image(args.image, use_dummy=args.dummy)
    else:
        # Show help
        parser.print_help()
        print("\n" + "="*60)
        print("Example usage:")
        print("  python main.py path/to/drug_image.jpg")
        print("  python main.py --dummy path/to/drug_image.jpg  # Test mode")
        print("="*60)


if __name__ == "__main__":
    main()
