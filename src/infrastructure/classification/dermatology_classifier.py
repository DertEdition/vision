"""
Dermatology Classifier

CNN-based dermatology image classification using a trained ResNet18 model.
Classifies skin lesion images for malignancy and disease type.
"""

import logging
from typing import Optional, Dict, Any, List
from pathlib import Path

import numpy as np

from ...domain.ports.image_classifier import ImageClassifierPort
from ...domain.value_objects.image_data import ImageData

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DERMATOLOGY_MODEL_PATH = PROJECT_ROOT / "models" / "dermatology_model.pth"


# DERM12345 disease sub-class labels (alphabetically sorted)
DERM_MALIGNANCY_CLASSES = ["benign", "indeterminate", "malignant"]

DERM_DISEASE_CLASSES = [
    "acral_lentiginious", "acral_nodular", "actinic_keratosis",
    "angiokeratoma", "basal_cell_carcinoma", "blue", "bowen_disease",
    "compound", "congenital", "cutaneous_horn", "dermal",
    "dermatofibroma", "lentigo_simplex", "melanoma_in_situ",
    "melanoma_invasive", "seborrheic_keratosis", "vascular"
]


class DermatologyClassifier(ImageClassifierPort):
    """
    Dermatology image classifier using a trained ResNet18 model.
    
    Provides two classification outputs:
    1. Malignancy classification: benign / malignant / indeterminate
    2. Disease type classification: specific sub-class
    
    Usage:
        classifier = DermatologyClassifier(
            model_path="models/dermatology_model.pth",
            device="cuda"
        )
        result = classifier.classify(image_data)
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
        confidence_threshold: float = 0.5,
        image_size: int = 224,
    ):
        """
        Initialize the dermatology classifier.
        
        Args:
            model_path: Path to the trained model weights
            device: Device to run inference on (cuda/cpu)
            confidence_threshold: Minimum confidence for predictions
            image_size: Input image size for the model
        """
        self._model_path = self._resolve_model_path(model_path)
        self._device = device
        self._confidence_threshold = confidence_threshold
        self._image_size = image_size
        self._model = None
        self._malignancy_classes = DERM_MALIGNANCY_CLASSES
        self._disease_classes = DERM_DISEASE_CLASSES
        
        self._load_model()

    @staticmethod
    def _resolve_model_path(model_path: Optional[str]) -> Path:
        """
        Resolve model path for robust startup.

        Priority:
        1) Explicit absolute path
        2) Explicit relative path from current working directory (if exists)
        3) Explicit relative path from project root
        4) Default canonical path: models/dermatology_model.pth under project root
        """
        if not model_path:
            return DEFAULT_DERMATOLOGY_MODEL_PATH

        candidate = Path(model_path).expanduser()
        if candidate.is_absolute():
            return candidate

        cwd_candidate = (Path.cwd() / candidate).resolve()
        if cwd_candidate.exists():
            return cwd_candidate

        return (PROJECT_ROOT / candidate).resolve()
    
    def _load_model(self) -> None:
        """Load the trained model from disk."""
        import torch
        
        model_file = self._model_path
        if not model_file.exists():
            logger.warning(
                f"Dermatology model not found at {self._model_path}. "
                "Classifier will use random weights. Train the model first."
            )
            # Create model with random weights for structure validation
            self._model = self._create_model_architecture(use_hidden_heads=True)
            self._model.eval()
            return
        
        try:
            device = torch.device(
                self._device if torch.cuda.is_available() and self._device == "cuda" 
                else "cpu"
            )
            
            checkpoint = torch.load(model_file, map_location=device, weights_only=False)
            
            # Update class lists from checkpoint if available
            if "malignancy_classes" in checkpoint:
                self._malignancy_classes = checkpoint["malignancy_classes"]
            if "disease_classes" in checkpoint:
                self._disease_classes = checkpoint["disease_classes"]
            
            state_dict = checkpoint["model_state_dict"]
            detected_hidden_heads = self._checkpoint_uses_hidden_heads(state_dict)

            load_error = None
            use_hidden_heads = detected_hidden_heads
            for candidate in (detected_hidden_heads, not detected_hidden_heads):
                try:
                    model_candidate = self._create_model_architecture(use_hidden_heads=candidate)
                    model_candidate.load_state_dict(state_dict)
                    self._model = model_candidate
                    use_hidden_heads = candidate
                    load_error = None
                    break
                except Exception as e:
                    load_error = e

            if load_error is not None:
                raise load_error

            self._model.to(device)
            self._model.eval()
            
            logger.info(
                f"Dermatology model loaded from {self._model_path} "
                f"({len(self._malignancy_classes)} malignancy classes, "
                f"{len(self._disease_classes)} disease classes, "
                f"hidden_heads={use_hidden_heads}, detected_hidden_heads={detected_hidden_heads})"
            )
        except Exception as e:
            logger.error(f"Failed to load dermatology model: {e}")
            self._model = self._create_model_architecture(use_hidden_heads=True)
            self._model.eval()
    
    @staticmethod
    def _checkpoint_uses_hidden_heads(state_dict: Dict[str, Any]) -> bool:
        """Detect whether checkpoint uses the newer hidden-layer dual heads."""
        for key in state_dict.keys():
            if key.endswith("malignancy_head.1.weight") or key.endswith("disease_head.1.weight"):
                return True
        return False

    def _create_model_architecture(self, use_hidden_heads: bool = True):
        """Create the model architecture (ResNet18 with dual classification heads)."""
        import torch
        import torch.nn as nn
        import torchvision.models as models
        
        class DermatologyNet(nn.Module):
            def __init__(self, num_malignancy_classes, num_disease_classes, use_hidden_heads):
                super().__init__()
                self.backbone = models.resnet18(weights=None)
                num_features = self.backbone.fc.in_features
                self.backbone.fc = nn.Identity()

                if use_hidden_heads:
                    # Matches train_dermatology_model.py architecture.
                    self.malignancy_head = nn.Sequential(
                        nn.Dropout(0.3),
                        nn.Linear(num_features, 128),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(128, num_malignancy_classes),
                    )
                    self.disease_head = nn.Sequential(
                        nn.Dropout(0.3),
                        nn.Linear(num_features, 256),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(256, num_disease_classes),
                    )
                else:
                    # Backward compatibility with older checkpoints.
                    self.malignancy_head = nn.Linear(num_features, num_malignancy_classes)
                    self.disease_head = nn.Linear(num_features, num_disease_classes)
            
            def forward(self, x):
                features = self.backbone(x)
                malignancy_out = self.malignancy_head(features)
                disease_out = self.disease_head(features)
                return malignancy_out, disease_out
        
        return DermatologyNet(
            num_malignancy_classes=len(self._malignancy_classes),
            num_disease_classes=len(self._disease_classes),
            use_hidden_heads=use_hidden_heads,
        )
    
    def classify(
        self,
        image: ImageData,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Classify a dermatology image.
        
        Args:
            image: Skin lesion image to classify
            options: Optional classification configuration
            
        Returns:
            Dictionary with malignancy and disease type predictions
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
        
        # Run inference
        with torch.no_grad():
            malignancy_logits, disease_logits = self._model(input_tensor)
            
            malignancy_probs = torch.softmax(malignancy_logits, dim=1)[0]
            disease_probs = torch.softmax(disease_logits, dim=1)[0]
        
        # Extract predictions
        mal_pred_idx = malignancy_probs.argmax().item()
        mal_confidence = malignancy_probs[mal_pred_idx].item()
        
        dis_pred_idx = disease_probs.argmax().item()
        dis_confidence = disease_probs[dis_pred_idx].item()
        
        # Build probability dictionaries
        malignancy_probabilities = {
            cls: malignancy_probs[i].item()
            for i, cls in enumerate(self._malignancy_classes)
        }
        
        disease_probabilities = {
            cls: disease_probs[i].item()
            for i, cls in enumerate(self._disease_classes)
        }
        
        return {
            "malignancy": {
                "prediction": self._malignancy_classes[mal_pred_idx],
                "confidence": mal_confidence,
                "probabilities": malignancy_probabilities,
            },
            "disease_type": {
                "prediction": self._disease_classes[dis_pred_idx],
                "confidence": dis_confidence,
                "probabilities": disease_probabilities,
            },
        }
    
    def get_class_labels(self) -> List[str]:
        """Get combined list of all class labels."""
        return self._malignancy_classes + self._disease_classes
    
    @property
    def model_name(self) -> str:
        """Get model identifier."""
        return "DermatologyNet-ResNet18"
