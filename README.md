# Drug Image Understanding & Information Pipeline

A production-grade AI system for pharmaceutical drug package image analysis.

## Overview

This system analyzes images of pharmaceutical drug packages (boxes, blisters, leaflets) and provides reliable, non-diagnostic information about the identified drug.

**Pipeline Flow:** VISION → OCR → ENTITY → RAG → LLM

## Features

- **Vision Analysis**: Detect pharmaceutical packaging using YOLOv8
- **OCR**: Extract text with PaddleOCR (Turkish language support)
- **Entity Extraction**: Identify drug names, ingredients, dosage forms
- **RAG**: Retrieve verified pharmaceutical knowledge from ChromaDB
- **LLM Response**: Generate safe, user-friendly explanations

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Application Layer                         │
│                   (Pipeline Orchestration)                        │
├─────────────────────────────────────────────────────────────────┤
│                         Domain Layer                              │
│              (Entities, Value Objects, Ports)                     │
├─────────────────────────────────────────────────────────────────┤
│                     Infrastructure Layer                          │
│    (YOLO, PaddleOCR, ChromaDB, OpenAI Adapters)                  │
├─────────────────────────────────────────────────────────────────┤
│                    Cross-Cutting Concerns                         │
│           (Logging, Validation, Safety, Error Handling)           │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
export OPENAI_API_KEY="your-api-key"
```

## Quick Start

```python
from drug_image_pipeline.src.application import DrugAnalysisService
from drug_image_pipeline.src.application.pipeline import PipelineBuilder
from drug_image_pipeline.src.infrastructure import (
    VisionAnalyzerFactory, VisionAnalyzerType,
    OCRFactory, OCRType,
    EntityExtractorFactory, EntityExtractorType,
    KnowledgeRetrieverFactory, KnowledgeRetrieverType,
    LLMFactory, LLMType,
)

# Create pipeline with factories
pipeline = (
    PipelineBuilder()
    .with_vision_analyzer(VisionAnalyzerFactory.create(VisionAnalyzerType.YOLO))
    .with_text_extractor(OCRFactory.create(OCRType.PADDLE, lang="tr"))
    .with_entity_extractor(EntityExtractorFactory.create(EntityExtractorType.HYBRID))
    .with_knowledge_retriever(KnowledgeRetrieverFactory.create(KnowledgeRetrieverType.CHROMA))
    .with_response_generator(LLMFactory.create(LLMType.OPENAI_GPT4))
    .build()
)

# Create service
service = DrugAnalysisService(pipeline)

# Analyze an image
result = service.analyze_from_file("path/to/drug_image.jpg")

# Get user-friendly response
response = result.get_user_response()
print(response)
```

## Project Structure

```
drug_image_pipeline/
├── src/
│   ├── domain/              # Pure business logic
│   │   ├── entities/        # Domain entities
│   │   ├── value_objects/   # Immutable value objects
│   │   ├── ports/           # Interface definitions
│   │   └── exceptions.py    # Domain exceptions
│   │
│   ├── application/         # Application orchestration
│   │   ├── pipeline/        # Pipeline orchestration
│   │   └── services/        # Application services
│   │
│   ├── infrastructure/      # External implementations
│   │   ├── vision/          # YOLO adapters
│   │   ├── ocr/             # PaddleOCR, Tesseract adapters
│   │   ├── entity_extraction/
│   │   ├── rag/             # ChromaDB adapter
│   │   └── llm/             # OpenAI adapter
│   │
│   └── cross_cutting/       # Cross-cutting concerns
│       ├── safety/          # Safety guardrails
│       └── ...
│
├── config/                  # Configuration
├── data/                    # Data assets
├── tests/                   # Test suite
└── requirements.txt
```

## Design Patterns

| Pattern | Usage |
|---------|-------|
| **Strategy** | Interchangeable OCR, Vision, LLM backends |
| **Factory** | Runtime selection of implementations |
| **Chain of Responsibility** | Sequential pipeline stage execution |
| **Adapter** | External services wrapped to match ports |
| **Fail-Soft** | Stage failures don't crash pipeline |

## Safety Features

- **No Diagnosis**: System never provides medical diagnoses
- **No Prescriptions**: No dosage or treatment recommendations
- **Mandatory Disclaimers**: All responses include medical disclaimers
- **Confidence Warnings**: Low-confidence results trigger warnings
- **Content Validation**: Responses checked for unsafe content

## Configuration

Set via environment variables or configuration file:

```bash
# Vision
export DRUG_PIPELINE_VISION_DEVICE=cpu

# OCR
export DRUG_PIPELINE_OCR_TYPE=paddle
export DRUG_PIPELINE_OCR_LANGUAGE=tr

# LLM
export OPENAI_API_KEY=your-api-key
export DRUG_PIPELINE_LLM_MODEL=gpt-4

# Data
export DRUG_PIPELINE_DATA_DIR=./data
```

## Extending the System

### Adding a New OCR Engine

1. Create `src/infrastructure/ocr/new_ocr.py` implementing `TextExtractorPort`
2. Register in `OCRFactory`
3. No changes to domain or application layers

### Adding a New LLM Provider

1. Create `src/infrastructure/llm/new_llm.py` implementing `ResponseGeneratorPort`
2. Register in `LLMFactory`
3. Update configuration

## License

MIT License

## Disclaimer

⚠️ **IMPORTANT**: This system is for informational purposes only and is NOT a substitute for professional medical advice. Always consult a healthcare provider before taking any medication.
