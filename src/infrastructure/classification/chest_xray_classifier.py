"""
Chest X-ray Classifier

CNN-based chest X-ray classification using a trained DenseNet121 model.
Multi-label classification for 14 thoracic disease findings.
"""

import logging
from typing import Optional, Dict, Any, List
from pathlib import Path

import numpy as np

from ...domain.ports.image_classifier import ImageClassifierPort
from ...domain.value_objects.image_data import ImageData

logger = logging.getLogger(__name__)


# NIH Chest X-ray 14 disease labels
CHEST_XRAY_CLASSES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]


class ChestXrayClassifier(ImageClassifierPort):
    """
    Chest X-ray classifier using a trained ResNet50 model.
    
    Multi-label classification for 14 thoracic disease findings.
    
    Usage:
        classifier = ChestXrayClassifier(
            model_path="models/chest_xray_model.pth",
            device="cuda"
        )
        result = classifier.classify(image_data)
    """
    
    def __init__(
        self,
        model_path: str = "./models/chest_xray_model.pth",
        device: str = "cuda",
        confidence_threshold: float = 0.5,
        image_size: int = 224,
    ):
        """
        Initialize the chest X-ray classifier.
        
        Args:
            model_path: Path to the trained model weights
            device: Device to run inference on (cuda/cpu)
            confidence_threshold: Minimum confidence for positive findings
            image_size: Input image size for the model
        """
        self._model_path = model_path
        self._device = device
        self._confidence_threshold = confidence_threshold
        self._image_size = image_size
        self._model = None
        self._class_labels = CHEST_XRAY_CLASSES
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the trained model from disk."""
        import torch
        
        model_file = Path(self._model_path)
        if not model_file.exists():
            logger.warning(
                f"Chest X-ray model not found at {self._model_path}. "
                "Classifier will use random weights. Train the model first."
            )
            self._model = self._create_model_architecture()
            self._model.eval()
            return
        
        try:
            device = torch.device(
                self._device if torch.cuda.is_available() and self._device == "cuda"
                else "cpu"
            )
            
            checkpoint = torch.load(self._model_path, map_location=device, weights_only=False)
            
            if "class_labels" in checkpoint:
                self._class_labels = checkpoint["class_labels"]
            
            self._model = self._create_model_architecture()
            self._model.load_state_dict(checkpoint["model_state_dict"])
            self._model.to(device)
            self._model.eval()
            
            logger.info(
                f"Chest X-ray model loaded from {self._model_path} "
                f"({len(self._class_labels)} disease classes)"
            )
        except Exception as e:
            logger.error(f"Failed to load chest X-ray model: {e}")
            self._model = self._create_model_architecture()
            self._model.eval()
    
    def _create_model_architecture(self):
        """Create the model architecture (ResNet50 with custom classifier)."""
        import torch
        import torch.nn as nn
        import torchvision.models as models
        
        model = models.resnet50(weights=None)
        num_features = model.fc.in_features
        # Multi-label: sigmoid output for each disease
        model.fc = nn.Linear(num_features, len(self._class_labels))
        
        return model
    
    def classify(
        self,
        image: ImageData,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Classify a chest X-ray image.
        
        Args:
            image: Chest X-ray image to classify
            options: Optional classification configuration
            
        Returns:
            Dictionary with multi-label disease findings and probabilities
        """
        import torch
        import cv2
        import numpy as np
        
        options = options or {}
        threshold = options.get("confidence_threshold", self._confidence_threshold)
        
        # Convert ImageData bytes to OpenCV image
        img_array = np.frombuffer(image.bytes, dtype=np.uint8)
        cv_image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if cv_image is None:
            raise ValueError("Failed to decode image")
        
        # Convert BGR to RGB
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        # Resize
        cv_image = cv2.resize(cv_image, (self._image_size, self._image_size), interpolation=cv2.INTER_LINEAR)
        
        # Convert to float and normalize to [0, 1]
        cv_image = cv_image.astype(np.float32) / 255.0
        
        # Normalize with ImageNet stats
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        cv_image = (cv_image - mean) / std
        
        # Convert to tensor (H, W, C) -> (C, H, W)
        input_tensor = torch.from_numpy(cv_image.transpose(2, 0, 1)).unsqueeze(0)
        
        # Move to model device
        device = next(self._model.parameters()).device
        input_tensor = input_tensor.to(device)
        
        # Run inference (multi-label → sigmoid)
        with torch.no_grad():
            logits = self._model(input_tensor)
            probabilities = torch.sigmoid(logits)[0]
        
        # Build results
        finding_probabilities = {
            cls: probabilities[i].item()
            for i, cls in enumerate(self._class_labels)
        }
        
        # Findings above threshold
        positive_findings = [
            cls for cls, prob in finding_probabilities.items()
            if prob >= threshold
        ]
        
        # Primary finding (highest probability)
        primary_idx = probabilities.argmax().item()
        primary_finding = self._class_labels[primary_idx]
        primary_confidence = probabilities[primary_idx].item()
        
        return {
            "findings": positive_findings,
            "finding_probabilities": finding_probabilities,
            "has_abnormality": len(positive_findings) > 0,
            "primary_finding": primary_finding,
            "primary_confidence": primary_confidence,
        }
    
    def get_class_labels(self) -> List[str]:
        """Get the list of disease labels."""
        return self._class_labels.copy()
    
    @property
    def model_name(self) -> str:
        """Get model identifier."""
        return "ChestXray-ResNet50"
