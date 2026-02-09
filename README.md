# Drug Image Understanding & Information Pipeline

A production-grade AI system for pharmaceutical drug package image analysis with flexible LLM backend support.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Configuration](#configuration)
- [Using Custom LLM Models](#using-custom-llm-models)
- [API Usage](#api-usage)
- [Data Pipeline](#data-pipeline)
- [Project Structure](#project-structure)
- [Advanced Usage](#advanced-usage)
- [Safety Features](#safety-features)

---

## Overview

This system analyzes images of pharmaceutical drug packages (boxes, blisters, leaflets) and provides reliable, non-diagnostic information about the identified drug.

**Pipeline Flow:** VISION → OCR → ENTITY → RAG → LLM

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                           API Layer                             │
│              (FastAPI, Routes, Request/Response Models)           │
├─────────────────────────────────────────────────────────────────┤
│                        Application Layer                         │
│                   (Pipeline Orchestration)                        │
├─────────────────────────────────────────────────────────────────┤
│                         Domain Layer                              │
│              (Entities, Value Objects, Ports)                     │
├─────────────────────────────────────────────────────────────────┤
│                     Infrastructure Layer                          │
│    (YOLO, PaddleOCR, ChromaDB, OpenAI/Ollama Adapters)           │
├─────────────────────────────────────────────────────────────────┤
│                    Cross-Cutting Concerns                         │
│           (Logging, Validation, Safety, Error Handling)           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Features

- ✅ **Vision Analysis**: Detect pharmaceutical packaging using YOLOv8
- ✅ **OCR**: Extract text with PaddleOCR/Tesseract (Turkish language support)
- ✅ **Entity Extraction**: Identify drug names, ingredients, dosage forms
- ✅ **RAG**: Retrieve verified pharmaceutical knowledge from ChromaDB
- ✅ **Flexible LLM Backend**: 
  - OpenAI (GPT-4, GPT-3.5)
  - Ollama (Local models: Gemma, Qwen, custom fine-tuned models)
  - Easy to extend with custom implementations
- ✅ **REST API**: FastAPI-based server for real-time analysis
- ✅ **Data Pipeline**: Automated scripts for drug data collection and processing
- ✅ **Safety Guardrails**: Medical disclaimers, confidence warnings, content validation

---

## Quick Start

### 1. Install Dependencies

```bash
# Clone the repository
git clone <your-repo-url>
cd drug_image_pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Set Up LLM Backend

**Option A: Using Ollama (Local, Free)**

```bash
# Install Ollama from https://ollama.ai
# Pull a model (e.g., Gemma)
ollama pull gemma3:4b

# The system is already configured to use Ollama by default!
```

**Option B: Using OpenAI**

```bash
# Set your API key
export OPENAI_API_KEY="your-api-key-here"
```

### 3. Run the API Server

```bash
python run_api.py
```

Visit `http://localhost:8000/docs` for interactive API documentation.

### 4. Analyze an Image

```python
from src.application.services import DrugAnalysisService
from src.application.pipeline import PipelineBuilder
from config import get_default_config

# Load configuration
config = get_default_config()

# Build pipeline (automatically uses config settings)
from src.api.dependencies import get_pipeline
pipeline = get_pipeline()

# Create service
service = DrugAnalysisService(pipeline)

# Analyze
result = service.analyze_from_file("path/to/drug_image.jpg")
print(result.get_user_response())
```

---

## Installation

### Prerequisites

- Python 3.8+
- (Optional) CUDA-capable GPU for faster processing
- (Optional) Ollama for local LLM inference

### Step-by-Step Installation

```bash
# 1. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download YOLO model (if not present)
# The system will download it automatically on first run

# 4. Set up knowledge base (optional, for RAG)
python scripts/populate_knowledge_base.py
```

---

## Configuration

The system uses a flexible configuration system located in `config/settings.py`. You can configure it via:

1. **Environment variables** (recommended for production)
2. **Direct code modification** in `config/settings.py`
3. **Runtime configuration** via Python code

### Configuration File Structure

```python
# config/settings.py

@dataclass
class LLMConfig:
    type: str = "ollama"              # "ollama" or "openai"
    model: str = "gemma3:4b"          # Model name
    api_key: Optional[str] = None     # Only for OpenAI
    ollama_base_url: str = "http://localhost:11434"
    temperature: float = 0.3
    max_tokens: int = 1000
    language: str = "tr"              # "tr" or "en"
```

### Environment Variables

```bash
# LLM Configuration
export DRUG_PIPELINE_LLM_TYPE=ollama           # or "openai"
export DRUG_PIPELINE_LLM_MODEL=gemma3:4b       # Your model
export OPENAI_API_KEY=sk-...                   # Only if using OpenAI

# Vision Configuration
export DRUG_PIPELINE_VISION_DEVICE=cuda        # cpu, cuda, or mps

# OCR Configuration
export DRUG_PIPELINE_OCR_TYPE=paddle           # paddle or tesseract
export DRUG_PIPELINE_OCR_LANGUAGE=tr           # Language code

# Data Paths
export DRUG_PIPELINE_DATA_DIR=./data
```

### Modifying Configuration Programmatically

```python
from config import AppConfig

# Create custom config
config = AppConfig()

# Change LLM settings
config.llm.type = "ollama"
config.llm.model = "your-custom-model:latest"
config.llm.temperature = 0.5

# Use in pipeline
from src.api.dependencies import _create_pipeline
pipeline = _create_pipeline(config)
```

---

## Using Custom LLM Models

### Scenario 1: Using Your Fine-Tuned Ollama Model

Your team member fine-tuned a model and you want to use it:

```bash
# 1. Pull or load the custom model in Ollama
ollama pull your-custom-model:latest
# or if it's a local model file:
ollama create your-custom-model -f Modelfile

# 2. Update configuration
# Option A: Via environment variable
export DRUG_PIPELINE_LLM_MODEL=your-custom-model:latest

# Option B: Edit config/settings.py
# Change line 58:
model: str = "your-custom-model:latest"

# 3. Restart the API
python run_api.py
```

**That's it!** The system will now use your custom model.

### Scenario 2: Switching Between Models

```python
# In your code or config/settings.py

# Use Gemma (default)
config.llm.model = "gemma3:4b"

# Use Qwen
config.llm.model = "qwen3:4b"

# Use your fine-tuned model
config.llm.model = "medical-turkish-llm:v2"

# Use OpenAI
config.llm.type = "openai"
config.llm.model = "gpt-4"
config.llm.api_key = "sk-..."
```

### Scenario 3: Adding a Completely New LLM Provider

If you want to integrate a different LLM service (e.g., Anthropic Claude, Hugging Face):

1. **Create adapter** implementing `ResponseGeneratorPort`:

```python
# src/infrastructure/llm/custom_llm.py

from src.domain.ports.response_generator import ResponseGeneratorPort

class CustomLLMGenerator(ResponseGeneratorPort):
    def __init__(self, api_key: str, model: str, **kwargs):
        self.api_key = api_key
        self.model = model
        # Initialize your LLM client
    
    def generate_response(self, context: dict) -> str:
        # Implement your LLM call logic
        pass
```

2. **Register in factory**:

```python
# src/infrastructure/llm/factory.py

class LLMType(Enum):
    # ... existing types ...
    CUSTOM = "custom"

# In LLMFactory.create():
elif llm_type == LLMType.CUSTOM:
    from .custom_llm import CustomLLMGenerator
    return CustomLLMGenerator(**kwargs)
```

3. **Update config**:

```python
config.llm.type = "custom"
config.llm.model = "your-model-name"
```

---

## API Usage

### Starting the Server

```bash
# Basic start
python run_api.py

# With custom host/port
python run_api.py --host 0.0.0.0 --port 8080

# With auto-reload (development)
python run_api.py --reload

# With debug logging
python run_api.py --debug
```

### API Endpoints

#### 1. Health Check

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2024-02-09T18:30:00Z"
}
```

#### 2. Analyze Image (File Upload)

```bash
curl -X POST http://localhost:8000/analyze/upload \
  -F "file=@drug_image.jpg"
```

**Response:**
```json
{
  "request_id": "abc123",
  "success": true,
  "drug": {
    "name": "Parol 500 mg",
    "active_ingredients": ["Paracetamol"],
    "dosage_form": "tablet",
    "strength": "500 mg",
    "manufacturer": "Atabay"
  },
  "explanation": "Bu ilaç parasetamol içeren bir ağrı kesicidir...",
  "confidence": "0.85",
  "warnings": [],
  "disclaimer": "⚠️ Bu bilgi eğitim amaçlıdır...",
  "processing_time_ms": 2341.5
}
```

#### 3. Analyze Image (Base64)

```bash
curl -X POST http://localhost:8000/analyze/base64 \
  -H "Content-Type: application/json" \
  -d '{
    "image_base64": "iVBORw0KGgoAAAANS...",
    "format": "jpeg"
  }'
```

#### 4. Interactive Documentation

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

---

## Data Pipeline

### Available Scripts

#### 1. Scrape Drug Information

```bash
python scripts/scrape_drug_info.py
```

Scrapes drug data from ilacabak.com and saves to JSON.

#### 2. Convert Excel to JSON

```bash
python scripts/convert_excel_to_drugs_json.py
```

Converts Excel drug database to system JSON format.

#### 3. Populate Knowledge Base

```bash
python scripts/populate_knowledge_base.py
```

Loads drug data into ChromaDB for RAG retrieval.

#### 4. Interactive Drug Entry

```bash
python scripts/add_drug_interactive.py
```

Manually add drug information via interactive CLI.

---

## Project Structure

```
drug_image_pipeline/
├── config/                      # Configuration management
│   ├── settings.py             # Main config (EDIT THIS for LLM settings)
│   └── __init__.py
│
├── src/
│   ├── api/                    # REST API
│   │   ├── app.py             # FastAPI application
│   │   ├── routes/            # Endpoints (health, analysis)
│   │   ├── models/            # Request/Response schemas
│   │   ├── dependencies.py    # Dependency injection
│   │   └── exceptions.py      # API exception handlers
│   │
│   ├── domain/                # Business logic (framework-agnostic)
│   │   ├── entities/          # Core entities (DrugInfo, PipelineResult)
│   │   ├── value_objects/     # Immutable values (ConfidenceScore)
│   │   ├── ports/             # Interfaces (ResponseGeneratorPort)
│   │   └── exceptions.py
│   │
│   ├── application/           # Use cases & orchestration
│   │   ├── pipeline/          # Pipeline orchestrator
│   │   └── services/          # DrugAnalysisService
│   │
│   ├── infrastructure/        # External integrations
│   │   ├── vision/            # YOLO implementation
│   │   ├── ocr/               # PaddleOCR, Tesseract
│   │   ├── entity_extraction/ # NER, regex extractors
│   │   ├── rag/               # ChromaDB integration
│   │   └── llm/               # LLM adapters
│   │       ├── openai_generator.py
│   │       ├── ollama_llm.py  # Ollama adapter
│   │       └── factory.py     # LLM factory (ADD NEW MODELS HERE)
│   │
│   └── cross_cutting/         # Shared utilities
│       ├── logging/
│       ├── safety/
│       └── validation/
│
├── scripts/                   # Data processing utilities
│   ├── scrape_drug_info.py
│   ├── convert_excel_to_drugs_json.py
│   ├── populate_knowledge_base.py
│   └── add_drug_interactive.py
│
├── data/                      # Data storage
│   ├── drug_knowledge_base/   # Drug information JSON files
│   └── chroma_db/            # Vector database
│
├── tests/                     # Test suite
├── run_api.py                # API entry point
├── main.py                   # CLI entry point
├── requirements.txt
└── README.md
```

---

## Advanced Usage

### Design Patterns Used

| Pattern | Usage | Benefit |
|---------|-------|---------|
| **Ports & Adapters** | Domain ports, Infrastructure adapters | Easy to swap implementations |
| **Factory** | LLMFactory, OCRFactory, etc. | Runtime component selection |
| **Strategy** | Interchangeable OCR, Vision, LLM | Flexible algorithm selection |
| **Chain of Responsibility** | Pipeline stages | Sequential processing |
| **Dependency Injection** | FastAPI dependencies | Testable, decoupled code |

### Extending the System

#### Adding a New OCR Engine

1. Implement `TextExtractorPort` in `src/infrastructure/ocr/`
2. Register in `OCRFactory`
3. Update config: `config.ocr.type = "your_ocr"`

#### Adding a New Vision Model

1. Implement `VisionAnalyzerPort` in `src/infrastructure/vision/`
2. Register in `VisionAnalyzerFactory`
3. Update config: `config.vision.type = "your_model"`

#### Adding Custom Pipeline Stages

```python
from src.application.pipeline import PipelineBuilder

pipeline = (
    PipelineBuilder()
    .with_vision_analyzer(...)
    .with_text_extractor(...)
    .with_custom_stage(YourCustomStage())  # Add your stage
    .build()
)
```

---

## Safety Features

- ✅ **No Medical Diagnosis**: System never provides diagnoses
- ✅ **No Prescriptions**: No dosage recommendations
- ✅ **Mandatory Disclaimers**: All responses include medical disclaimers
- ✅ **Confidence Warnings**: Low-confidence results flagged
- ✅ **Content Validation**: Responses checked for unsafe content
- ✅ **Error Handling**: Graceful degradation on failures

---

## Troubleshooting

### Common Issues

**Issue: "Ollama connection refused"**
```bash
# Make sure Ollama is running
ollama serve

# Check if model is available
ollama list
```

**Issue: "CUDA out of memory"**
```bash
# Switch to CPU
export DRUG_PIPELINE_VISION_DEVICE=cpu
```

**Issue: "Model not found"**
```bash
# Pull the model
ollama pull gemma3:4b
```

---

## License

MIT License

## Disclaimer

⚠️ **IMPORTANT**: This system is for informational and educational purposes only and is NOT a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider or pharmacist before taking any medication. Do not start, stop, or change any treatment without professional medical advice.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Support

For issues and questions:
- Open an issue on GitHub
- Check existing documentation
- Review API docs at `/docs`
