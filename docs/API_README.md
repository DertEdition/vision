# REST API Documentation

## Drug Image Analysis API

This document describes the REST API for the Drug Image Analysis Pipeline. The API allows you to analyze pharmaceutical drug images via HTTP requests.

---

## Quick Start

### Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Running the API Server

Start the API server:

```bash
python run_api.py
```

The server will start on `http://localhost:8000` by default.

### API Documentation (Swagger UI)

Once the server is running, open your browser to:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## Configuration

The API can be configured via environment variables or command-line arguments.

### Command-Line Options

```bash
python run_api.py --help

Options:
  --host TEXT      Host to bind to (default: 0.0.0.0)
  --port INTEGER   Port to bind to (default: 8000)
  --workers INT    Number of worker processes (default: 1)
  --reload         Enable auto-reload for development
  --debug          Enable debug logging
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DRUG_PIPELINE_VISION_DEVICE` | Vision device (cpu/cuda) | cuda |
| `DRUG_PIPELINE_OCR_TYPE` | OCR engine (tesseract/paddle) | tesseract |
| `DRUG_PIPELINE_LLM_MODEL` | LLM model name | gemma3:4b |
| `DRUG_PIPELINE_LOG_LEVEL` | Logging level | INFO |

---

## API Endpoints

### Health Check

#### `GET /health`

Basic health check.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2026-01-28T14:30:00.000Z",
  "components": null
}
```

#### `GET /health/ready`

Readiness check (verifies ML models are loaded).

---

### Image Analysis

#### `POST /analyze/upload`

Analyze a drug image from file upload.

**Request:**
- Content-Type: `multipart/form-data`
- Body: `file` (image file)

**Example (curl):**
```bash
curl -X POST http://localhost:8000/analyze/upload \
  -F "file=@path/to/drug_image.jpg"
```

**Example (Python):**
```python
import requests

with open("drug_image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/analyze/upload",
        files={"file": f}
    )
print(response.json())
```

---

#### `POST /analyze/base64`

Analyze a drug image from base64-encoded data.

**Request:**
```json
{
  "image_base64": "/9j/4AAQSkZJRg...",
  "format": "jpeg",
  "options": null
}
```

**Example (JavaScript):**
```javascript
const response = await fetch('http://localhost:8000/analyze/base64', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    image_base64: btoa(imageData),
    format: 'jpeg'
  })
});
const result = await response.json();
```

---

#### `POST /analyze/path`

Analyze a drug image from a server-side file path.

**Request:**
```json
{
  "file_path": "./data/test1.jpeg",
  "options": null
}
```

**Example (curl):**
```bash
curl -X POST http://localhost:8000/analyze/path \
  -H "Content-Type: application/json" \
  -d '{"file_path": "./data/test1.jpeg"}'
```

---

## Response Format

All analysis endpoints return the same response structure:

```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "success": true,
  "drug": {
    "name": "Aspirin",
    "active_ingredients": ["Acetylsalicylic acid"],
    "dosage_form": "Tablet",
    "strength": "500mg",
    "manufacturer": "Bayer"
  },
  "explanation": "Aspirin is a nonsteroidal anti-inflammatory drug...",
  "confidence": "high",
  "warnings": ["Do not use if allergic to NSAIDs"],
  "disclaimer": "⚠️ This information is for educational purposes only...",
  "processing_time_ms": 1234.56,
  "stage_timings": [
    {"stage": "vision_analysis", "status": "completed", "duration_ms": 250.0},
    {"stage": "text_extraction", "status": "completed", "duration_ms": 180.0}
  ],
  "errors": null
}
```

### Error Response

```json
{
  "success": false,
  "error": "Image file not found",
  "error_type": "InvalidImageError",
  "details": [
    {
      "error_type": "InvalidImageError",
      "message": "Image file not found",
      "is_recoverable": false
    }
  ],
  "request_id": null
}
```

---

## HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request (invalid image, empty file) |
| 422 | Validation Error (invalid request body) |
| 500 | Internal Server Error (pipeline failure) |

---

## Development

### Running with Auto-Reload

For development, use the `--reload` flag:

```bash
python run_api.py --reload --debug
```

### Running via Uvicorn Directly

```bash
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

---

## Performance Notes

- **GPU Workloads**: The API defaults to 1 worker to avoid CUDA memory issues. Scale horizontally with a load balancer for production.
- **Request Timeout**: Default timeout is 180 seconds to accommodate LLM inference.
- **File Upload Limit**: Maximum upload size is 50MB.

---

## ⚠️ Disclaimer

This API provides information for **educational purposes only** and is NOT a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider before taking any medication.
