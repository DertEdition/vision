"""
Pydantic Request Models

Request validation schemas for the Drug Image Analysis API.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any


class AnalyzeFromPathRequest(BaseModel):
    """
    Request to analyze a drug image from a file path on the server.
    
    Attributes:
        file_path: Absolute or relative path to the drug image file.
        options: Optional analysis configuration overrides.
    """
    
    file_path: str = Field(
        ...,
        description="Path to the drug image file on the server",
        examples=["./data/test1.jpeg", "C:/images/drug.jpg"]
    )
    options: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional analysis options (e.g., OCR language, confidence thresholds)"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "file_path": "./data/test1.jpeg",
                    "options": None
                }
            ]
        }
    }


class AnalyzeFromBase64Request(BaseModel):
    """
    Request to analyze a drug image from base64-encoded data.
    
    Attributes:
        image_base64: Base64-encoded image data.
        format: Image format hint (jpeg, png, etc.).
        options: Optional analysis configuration overrides.
    """
    
    image_base64: str = Field(
        ...,
        description="Base64-encoded image data (without data URL prefix)",
        min_length=100  # Basic validation: base64 images are at least this long
    )
    format: Optional[str] = Field(
        None,
        description="Image format (jpeg, png, webp, etc.). Auto-detected if not provided.",
        examples=["jpeg", "png"]
    )
    options: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional analysis options"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "image_base64": "/9j/4AAQSkZJRg...",
                    "format": "jpeg",
                    "options": None
                }
            ]
        }
    }
