"""
Chest X-ray Classifier

CNN-based chest X-ray classification using a trained ResNet50 model.
Multi-label classification for 14 thoracic disease findings.
"""

import logging
from typing import Optional, Dict, Any, List
from pathlib import Path

import numpy as np
import cv2

from ...domain.ports.image_classifier import ImageClassifierPort
from ...domain.value_objects.image_data import ImageData

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHEST_XRAY_MODEL_PATH = PROJECT_ROOT / "models" / "chest_xray_model.pth"


# NIH Chest X-ray 14 disease labels
CHEST_XRAY_CLASSES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]


def _odd_kernel_size(value: int, fallback: int = 3) -> int:
    """Ensure OpenCV kernel sizes are valid odd positive integers."""
    if value is None:
        return fallback
    k = int(value)
    if k < 1:
        return fallback
    if k % 2 == 0:
        k += 1
    return k


def _gaussian_kernel_for_sigma(sigma: float) -> int:
    """Build an odd Gaussian kernel size from sigma."""
    if sigma <= 0:
        return 3
    k = int(round(6 * sigma + 1))
    return _odd_kernel_size(k, fallback=3)


def _apply_gamma_float(image_01: np.ndarray, gamma: float = 0.5, c: float = 1.0) -> np.ndarray:
    """Apply gamma transform on [0,1] float image."""
    if gamma <= 0:
        return image_01
    if abs(gamma - 1.0) < 1e-6 and abs(c - 1.0) < 1e-6:
        return image_01
    transformed = c * np.power(np.clip(image_01, 0.0, 1.0), float(gamma))
    return np.clip(transformed, 0.0, 1.0)


def _safe_percentile_ref(arr: np.ndarray, q: float = 99.0, fallback: float = 1.0) -> float:
    """Get a stable positive normalization reference from percentile statistics."""
    if arr.size == 0:
        return float(fallback)
    ref = float(np.percentile(arr, float(q)))
    if not np.isfinite(ref) or ref <= 1e-8:
        ref = float(np.max(arr))
    if not np.isfinite(ref) or ref <= 1e-8:
        ref = float(fallback)
    return ref


def _enhance_chest_xray_bgr(
    image_bgr: np.ndarray,
    force_grayscale: bool,
    laplacian_ksize: int,
    sobel_ksize: int,
    lowpass_sigma: float,
    detail_gain: float,
    gamma: float,
    gamma_c: float,
) -> np.ndarray:
    """Deterministic enhancement pipeline matching training-time preprocessing."""
    if force_grayscale:
        gray_u8 = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray_u8 = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    gray = gray_u8.astype(np.float32) / 255.0

    lap_k = _odd_kernel_size(laplacian_ksize, fallback=3)
    sobel_k = _odd_kernel_size(sobel_ksize, fallback=3)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=lap_k)

    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=sobel_k)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=sobel_k)
    sobel_mag = cv2.magnitude(sobel_x, sobel_y)

    if lowpass_sigma > 0:
        lowpass_kernel = _gaussian_kernel_for_sigma(float(lowpass_sigma))
        sobel_lowpass = cv2.GaussianBlur(
            sobel_mag,
            (lowpass_kernel, lowpass_kernel),
            sigmaX=float(lowpass_sigma),
            sigmaY=float(lowpass_sigma),
        )
    else:
        sobel_lowpass = sobel_mag

    mask_ref = _safe_percentile_ref(sobel_lowpass, q=99.0, fallback=1.0)
    detail_mask = np.clip(sobel_lowpass / (mask_ref + 1e-8), 0.0, 1.0)
    detail_mask = np.clip((detail_mask - 0.15) / 0.85, 0.0, 1.0)

    lap_ref = _safe_percentile_ref(np.abs(laplacian), q=99.0, fallback=1.0)
    laplacian_norm = laplacian / (lap_ref + 1e-8)

    fine_detail = np.clip(laplacian_norm * detail_mask, -1.0, 1.0)
    enhanced_linear = np.clip(gray + float(detail_gain) * 0.35 * fine_detail, 0.0, 1.0)
    enhanced_gamma = _apply_gamma_float(enhanced_linear, gamma=float(gamma), c=float(gamma_c))

    enhanced_u8 = np.clip(enhanced_gamma * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(enhanced_u8, cv2.COLOR_GRAY2RGB)


def _enhance_chest_xray_clahe(
    image_bgr: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: int = 8,
) -> np.ndarray:
    """CLAHE-based preprocessing matching training-time pipeline."""
    gray_u8 = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=(int(tile_grid_size), int(tile_grid_size)),
    )
    enhanced = clahe.apply(gray_u8)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)


def _preprocess_chest_xray(image_bgr: np.ndarray, mode: str = "clahe", **kwargs) -> np.ndarray:
    """Dispatch preprocessing based on mode."""
    if mode == "clahe":
        return _enhance_chest_xray_clahe(
            image_bgr,
            clip_limit=kwargs.get("clip_limit", 2.0),
            tile_grid_size=kwargs.get("tile_grid_size", 8),
        )
    elif mode == "laplacian_sobel":
        return _enhance_chest_xray_bgr(
            image_bgr,
            force_grayscale=kwargs.get("force_grayscale", True),
            laplacian_ksize=kwargs.get("laplacian_ksize", 3),
            sobel_ksize=kwargs.get("sobel_ksize", 3),
            lowpass_sigma=kwargs.get("lowpass_sigma", 1.2),
            detail_gain=kwargs.get("detail_gain", 1.0),
            gamma=kwargs.get("gamma", 0.5),
            gamma_c=kwargs.get("gamma_c", 1.0),
        )
    elif mode == "none":
        gray_u8 = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2RGB)
    else:
        return _enhance_chest_xray_clahe(image_bgr)


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
        model_path: Optional[str] = None,
        device: str = "cuda",
        confidence_threshold: float = 0.5,
        abnormality_threshold: Optional[float] = None,
        preprocess_mode: str = "auto",
        image_size: int = 300,
        force_grayscale: bool = True,
        laplacian_ksize: int = 3,
        sobel_ksize: int = 3,
        lowpass_sigma: float = 1.2,
        detail_gain: float = 1.0,
        gamma: float = 0.5,
        gamma_c: float = 1.0,
    ):
        """
        Initialize the chest X-ray classifier.

        Args:
            model_path: Path to the trained model weights
            device: Device to run inference on (cuda/cpu)
            confidence_threshold: Minimum confidence for reporting a finding
            abnormality_threshold: Higher threshold for deciding has_abnormality
            preprocess_mode: 'auto' reads from checkpoint; or 'clahe'/'laplacian_sobel'/'none'
            image_size: Input image size for the model
        """
        self._model_path = self._resolve_model_path(model_path)
        self._device = device
        self._confidence_threshold = float(confidence_threshold)
        if abnormality_threshold is None:
            abnormality_threshold = max(self._confidence_threshold + 0.1, 0.65)
        self._abnormality_threshold = float(abnormality_threshold)
        self._preprocess_mode = preprocess_mode  # resolved in _load_model
        self._preprocess_kwargs = {}  # populated from checkpoint
        self._force_grayscale = bool(force_grayscale)
        self._laplacian_ksize = _odd_kernel_size(laplacian_ksize, fallback=3)
        self._sobel_ksize = _odd_kernel_size(sobel_ksize, fallback=3)
        self._lowpass_sigma = max(0.0, float(lowpass_sigma))
        self._detail_gain = max(0.0, float(detail_gain))
        self._gamma = float(gamma)
        self._gamma_c = float(gamma_c)
        self._image_size = image_size
        self._backbone = "resnet50"  # default, updated from checkpoint
        self._dropout = 0.3
        self._hidden_dim = 512
        self._use_new_head = False  # True for v2 models with Dropout+BN head
        self._model = None
        self._class_labels = CHEST_XRAY_CLASSES

        self._load_model()

    @staticmethod
    def _resolve_model_path(model_path: Optional[str]) -> Path:
        """
        Resolve model path for robust startup.

        Priority:
        1) Explicit absolute path
        2) Explicit relative path from current working directory (if exists)
        3) Explicit relative path from project root
        4) Default canonical path: models/chest_xray_model.pth under project root
        """
        if not model_path:
            return DEFAULT_CHEST_XRAY_MODEL_PATH

        candidate = Path(model_path).expanduser()
        if candidate.is_absolute():
            return candidate

        cwd_candidate = (Path.cwd() / candidate).resolve()
        if cwd_candidate.exists():
            return cwd_candidate

        return (PROJECT_ROOT / candidate).resolve()

    def _runtime_preprocessing_signature(self) -> Dict[str, Any]:
        """Build runtime preprocessing signature for consistency checks."""
        sig = {
            "pipeline": self._preprocess_mode,
            "image_size": int(self._image_size),
        }
        sig.update(self._preprocess_kwargs)
        return sig

    @staticmethod
    def _extract_training_preprocessing(checkpoint: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract training-time preprocessing metadata from checkpoint."""
        preproc = checkpoint.get("preprocessing")
        if isinstance(preproc, dict):
            return preproc

        args = checkpoint.get("args")
        if not isinstance(args, dict):
            return None

        legacy_keys = (
            "enable_clahe",
            "clahe_clip_limit",
            "clahe_tile_size",
            "enable_unsharp",
            "unsharp_sigma",
            "unsharp_amount",
            "enable_contrast_stretch",
            "stretch_low_pct",
            "stretch_high_pct",
            "disable_clahe",
            "disable_unsharp",
            "disable_contrast_stretch",
        )
        if any(k in args for k in legacy_keys):
            return {"pipeline": "legacy_clahe_unsharp_from_args"}

        new_keys = (
            "laplacian_ksize",
            "sobel_ksize",
            "lowpass_sigma",
            "detail_gain",
            "gamma",
            "gamma_c",
            "disable_force_grayscale",
            "image_size",
        )
        if not any(k in args for k in new_keys):
            return None

        return {
            "pipeline": "laplacian_sobel_lowpass_gamma_v1",
            "force_grayscale": bool(not args.get("disable_force_grayscale", False)),
            "laplacian_ksize": int(args.get("laplacian_ksize", 3)),
            "sobel_ksize": int(args.get("sobel_ksize", 3)),
            "lowpass_sigma": float(args.get("lowpass_sigma", 1.2)),
            "detail_gain": float(args.get("detail_gain", 1.0)),
            "gamma": float(args.get("gamma", 0.5)),
            "gamma_c": float(args.get("gamma_c", 1.0)),
            "image_size": int(args.get("image_size", 224)),
        }

    def _log_preprocessing_compatibility(self, training_preproc: Optional[Dict[str, Any]]) -> None:
        """Log training-vs-runtime preprocessing consistency to catch distribution shift risks."""
        runtime = self._runtime_preprocessing_signature()
        logger.info("Chest X-ray runtime preprocessing: %s", runtime)

        if not training_preproc:
            logger.warning(
                "Checkpoint has no preprocessing metadata. "
                "Cannot verify train/inference preprocessing consistency."
            )
            return

        logger.info("Chest X-ray checkpoint preprocessing: %s", training_preproc)

        train_pipeline = training_preproc.get("pipeline")
        runtime_pipeline = runtime.get("pipeline")
        if train_pipeline and train_pipeline != runtime_pipeline:
            logger.warning(
                "Preprocessing pipeline mismatch: checkpoint=%s, runtime=%s",
                train_pipeline,
                runtime_pipeline,
            )
            return

        float_keys = {"lowpass_sigma", "detail_gain", "gamma", "gamma_c"}
        compare_keys = (
            "force_grayscale",
            "laplacian_ksize",
            "sobel_ksize",
            "lowpass_sigma",
            "detail_gain",
            "gamma",
            "gamma_c",
            "image_size",
        )

        mismatches = []
        for key in compare_keys:
            if key not in training_preproc:
                continue
            train_val = training_preproc.get(key)
            run_val = runtime.get(key)
            if key in float_keys:
                try:
                    if abs(float(train_val) - float(run_val)) > 1e-6:
                        mismatches.append((key, train_val, run_val))
                except Exception:
                    mismatches.append((key, train_val, run_val))
            else:
                if str(train_val) != str(run_val):
                    mismatches.append((key, train_val, run_val))

        if mismatches:
            mismatch_text = ", ".join(
                f"{k}: ckpt={tv} runtime={rv}" for (k, tv, rv) in mismatches
            )
            logger.warning(
                "Preprocessing parameter mismatch detected between training checkpoint and runtime: %s",
                mismatch_text,
            )
        else:
            logger.info("Training and runtime preprocessing parameters are aligned.")
    
    def _load_model(self) -> None:
        """Load the trained model from disk."""
        import torch

        model_file = self._model_path
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

            checkpoint = torch.load(model_file, map_location=device, weights_only=False)

            if "class_labels" in checkpoint:
                self._class_labels = checkpoint["class_labels"]

            # Read model config from v2 checkpoints
            if "model_config" in checkpoint:
                mc = checkpoint["model_config"]
                self._backbone = mc.get("backbone", "resnet50")
                self._dropout = mc.get("dropout", 0.3)
                self._hidden_dim = mc.get("hidden_dim", 512)
                self._use_new_head = True
            elif "backbone" in checkpoint:
                self._backbone = checkpoint["backbone"]
                self._use_new_head = self._backbone != "resnet50"
            else:
                # Legacy checkpoint: plain ResNet50 Linear head
                self._backbone = "resnet50"
                self._use_new_head = False

            # Read preprocessing config from checkpoint
            training_preproc = self._extract_training_preprocessing(checkpoint)
            if self._preprocess_mode == "auto" and training_preproc:
                self._preprocess_mode = training_preproc.get("pipeline", "clahe")
                # Copy preprocessing params
                self._preprocess_kwargs = {
                    k: v for k, v in training_preproc.items()
                    if k not in ("pipeline", "image_size")
                }
                if "image_size" in training_preproc:
                    self._image_size = int(training_preproc["image_size"])
            elif self._preprocess_mode == "auto":
                self._preprocess_mode = "clahe"  # safe default

            self._log_preprocessing_compatibility(training_preproc)

            self._model = self._create_model_architecture()
            self._model.load_state_dict(checkpoint["model_state_dict"])
            self._model.to(device)
            self._model.eval()

            logger.info(
                f"Chest X-ray model loaded: backbone={self._backbone}, "
                f"preprocess={self._preprocess_mode}, "
                f"classes={len(self._class_labels)}, "
                f"thr={self._confidence_threshold:.2f}, "
                f"abn_thr={self._abnormality_threshold:.2f})"
            )
        except Exception as e:
            logger.error(f"Failed to load chest X-ray model: {e}")
            self._model = self._create_model_architecture()
            self._model.eval()

    def _create_model_architecture(self):
        """Create model architecture matching training config."""
        import torch.nn as nn
        import torchvision.models as models

        backbone = self._backbone
        dropout = self._dropout
        hidden_dim = self._hidden_dim
        n_classes = len(self._class_labels)

        if backbone in ("efficientnet_b4", "efficientnet_b3") or self._use_new_head:
            # v2 architecture with Dropout+BN classifier head
            if backbone == "efficientnet_b4":
                base = models.efficientnet_b4(weights=None)
                num_features = base.classifier[1].in_features
                base.classifier = nn.Identity()
            elif backbone == "efficientnet_b3":
                base = models.efficientnet_b3(weights=None)
                num_features = base.classifier[1].in_features
                base.classifier = nn.Identity()
            else:  # resnet50 with new head
                base = models.resnet50(weights=None)
                num_features = base.fc.in_features
                base.fc = nn.Identity()

            class _ChestModel(nn.Module):
                def __init__(self, bb, clf):
                    super().__init__()
                    self.backbone = bb
                    self.classifier = clf
                def forward(self, x):
                    features = self.backbone(x)
                    if features.dim() > 2:
                        features = features.view(features.size(0), -1)
                    return self.classifier(features)

            clf = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(num_features, hidden_dim),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout * 0.5),
                nn.Linear(hidden_dim, n_classes),
            )
            return _ChestModel(base, clf)
        else:
            # Legacy: plain ResNet50 Linear head
            model = models.resnet50(weights=None)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, n_classes)
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
        import numpy as np
        
        options = options or {}
        finding_threshold = float(options.get("confidence_threshold", self._confidence_threshold))
        abnormality_threshold = float(options.get("abnormality_threshold", self._abnormality_threshold))
        abnormality_threshold = max(abnormality_threshold, finding_threshold)
        force_grayscale = bool(options.get("force_grayscale", self._force_grayscale))
        laplacian_ksize = _odd_kernel_size(options.get("laplacian_ksize", self._laplacian_ksize), fallback=3)
        sobel_ksize = _odd_kernel_size(options.get("sobel_ksize", self._sobel_ksize), fallback=3)
        lowpass_sigma = max(0.0, float(options.get("lowpass_sigma", self._lowpass_sigma)))
        detail_gain = max(0.0, float(options.get("detail_gain", self._detail_gain)))
        gamma = float(options.get("gamma", self._gamma))
        gamma_c = float(options.get("gamma_c", self._gamma_c))
        
        # Convert ImageData bytes to OpenCV image
        img_array = np.frombuffer(image.bytes, dtype=np.uint8)
        cv_image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if cv_image is None:
            raise ValueError("Failed to decode image")

        cv_image = _preprocess_chest_xray(cv_image, mode=self._preprocess_mode, **self._preprocess_kwargs)
        
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
        
        # Findings above each threshold (reporting vs abnormality decision).
        positive_findings = [
            cls for cls, prob in finding_probabilities.items() if prob >= finding_threshold
        ]
        abnormal_findings = [
            cls for cls, prob in finding_probabilities.items() if prob >= abnormality_threshold
        ]
        
        # Primary finding (highest probability)
        primary_idx = probabilities.argmax().item()
        primary_finding = self._class_labels[primary_idx]
        primary_confidence = probabilities[primary_idx].item()
        has_abnormality = len(abnormal_findings) > 0

        return {
            # Use abnormality-thresholded findings to reduce false positives on No Finding.
            "findings": abnormal_findings if has_abnormality else [],
            "finding_probabilities": finding_probabilities,
            "has_abnormality": has_abnormality,
            "primary_finding": primary_finding if has_abnormality else "No Finding",
            "primary_confidence": primary_confidence,
            "reporting_threshold": finding_threshold,
            "abnormality_threshold": abnormality_threshold,
            "candidate_findings": positive_findings,
        }
    
    def get_class_labels(self) -> List[str]:
        """Get the list of disease labels."""
        return self._class_labels.copy()
    
    @property
    def model_name(self) -> str:
        """Get model identifier."""
        return "ChestXray-ResNet50"
