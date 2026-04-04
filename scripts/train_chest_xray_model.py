"""
Chest X-ray Model Training Script

Train a ResNet50 CNN model on the NIH Chest X-ray dataset for
multi-label thoracic disease classification.

14 Disease Labels:
    Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule,
    Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis,
    Pleural_Thickening, Hernia

Usage:
    python scripts/train_chest_xray_model.py --epochs 10 --batch-size 32
    python scripts/train_chest_xray_model.py --epochs 5 --sample-size 1000  # Quick test

Prerequisites:
    Download NIH Chest X-ray dataset from Kaggle:
    kaggle datasets download -d nih-chest-xrays/data -p data/chest_xray/
"""

import argparse
import math
import os
import sys
import time
import logging
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
import cv2
from sklearn.metrics import roc_auc_score
try:
    from tqdm.auto import tqdm
    TQDM_AVAILABLE = True
except ImportError:  # pragma: no cover - fallback for minimal environments
    TQDM_AVAILABLE = False

    def tqdm(iterable=None, *args, **kwargs):
        return iterable

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# NIH Chest X-ray 14 disease labels
DISEASE_LABELS = [
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


def enhance_chest_xray_bgr(
    image_bgr: np.ndarray,
    force_grayscale: bool = True,
    laplacian_ksize: int = 3,
    sobel_ksize: int = 3,
    lowpass_sigma: float = 1.2,
    detail_gain: float = 1.0,
    gamma: float = 0.5,
    gamma_c: float = 1.0,
    return_debug: bool = False,
):
    """
    Enhancement pipeline requested for chest X-ray images:
    1) Laplacian detail extraction
    2) Sobel gradients -> low-pass filtered detail mask
    3) Fine-detail mask = Laplacian * detail_mask
    4) Add fine-detail mask back to original
    5) Gamma transform (default gamma=0.5, c=1.0)
    """
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

    # Robust normalization avoids a single outlier edge dominating the mask.
    mask_ref = _safe_percentile_ref(sobel_lowpass, q=99.0, fallback=1.0)
    detail_mask = np.clip(sobel_lowpass / (mask_ref + 1e-8), 0.0, 1.0)
    # Soft floor suppresses weak/noisy gradients after low-pass filtering.
    detail_mask = np.clip((detail_mask - 0.15) / 0.85, 0.0, 1.0)

    # Normalize Laplacian magnitude so detail_gain behaves consistently across images.
    lap_ref = _safe_percentile_ref(np.abs(laplacian), q=99.0, fallback=1.0)
    laplacian_norm = laplacian / (lap_ref + 1e-8)

    fine_detail = np.clip(laplacian_norm * detail_mask, -1.0, 1.0)
    # 0.35 keeps default gain=1 stable while still making the effect visible.
    enhanced_linear = np.clip(gray + float(detail_gain) * 0.35 * fine_detail, 0.0, 1.0)
    enhanced_gamma = _apply_gamma_float(enhanced_linear, gamma=float(gamma), c=float(gamma_c))

    enhanced_u8 = np.clip(enhanced_gamma * 255.0, 0, 255).astype(np.uint8)
    enhanced_rgb = cv2.cvtColor(enhanced_u8, cv2.COLOR_GRAY2RGB)

    if not return_debug:
        return enhanced_rgb

    debug = {
        "original_gray": gray_u8,
        "laplacian": laplacian,
        "laplacian_norm": laplacian_norm,
        "sobel_mag": sobel_mag,
        "sobel_lowpass": sobel_lowpass,
        "detail_mask": detail_mask,
        "fine_detail": fine_detail,
        "enhanced_linear": enhanced_linear,
        "enhanced_gamma": enhanced_gamma,
    }
    return enhanced_rgb, debug


def enhance_chest_xray_clahe(
    image_bgr: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: int = 8,
) -> np.ndarray:
    """
    CLAHE-based preprocessing for chest X-ray images.

    CLAHE (Contrast Limited Adaptive Histogram Equalization) is the
    standard preprocessing in medical imaging literature (CheXNet, etc.).
    It enhances local contrast while limiting noise amplification.
    """
    gray_u8 = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=(int(tile_grid_size), int(tile_grid_size)),
    )
    enhanced = clahe.apply(gray_u8)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)


def preprocess_chest_xray(image_bgr: np.ndarray, mode: str = "clahe", **kwargs) -> np.ndarray:
    """Dispatch preprocessing based on mode."""
    if mode == "clahe":
        return enhance_chest_xray_clahe(
            image_bgr,
            clip_limit=kwargs.get("clip_limit", 2.0),
            tile_grid_size=kwargs.get("tile_grid_size", 8),
        )
    elif mode == "laplacian_sobel":
        return enhance_chest_xray_bgr(
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
        raise ValueError(f"Unknown preprocessing mode: {mode}. Use 'clahe', 'laplacian_sobel', or 'none'.")


def _normalize_for_display(image: np.ndarray) -> np.ndarray:
    """Normalize float image for display as uint8."""
    arr = image.astype(np.float32)
    min_v = float(arr.min())
    max_v = float(arr.max())
    if max_v <= min_v:
        return np.zeros_like(arr, dtype=np.uint8)
    norm = (arr - min_v) / (max_v - min_v)
    return np.clip(norm * 255.0, 0, 255).astype(np.uint8)


def preview_enhancement_on_test_dataset(
    dataset,
    sample_count: int = 6,
    random_seed: int = 42,
    show: bool = True,
    show_debug_maps: bool = False,
    save_path: Path = None,
):
    """Preview original vs enhanced images on test split for quick visual validation."""
    if len(dataset) == 0:
        logger.warning("Enhancement preview skipped: empty dataset")
        return

    count = min(max(1, int(sample_count)), len(dataset))
    rng = np.random.default_rng(int(random_seed))
    indices = rng.choice(len(dataset), size=count, replace=False).tolist()

    num_cols = 4 if show_debug_maps else 2
    fig, axes = plt.subplots(
        count,
        num_cols,
        figsize=(12 if num_cols == 2 else 20, 4 * count),
        squeeze=False,
        constrained_layout=True,
    )

    for row_idx, ds_idx in enumerate(indices):
        row = dataset.df.iloc[ds_idx]
        image_name = row["Image Index"]
        img_path = os.path.join(dataset.image_dir, image_name)

        with open(img_path, "rb") as f:
            file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
            image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image_bgr is None:
            continue

        _, debug = enhance_chest_xray_bgr(
            image_bgr=image_bgr,
            force_grayscale=dataset.force_grayscale,
            laplacian_ksize=dataset.laplacian_ksize,
            sobel_ksize=dataset.sobel_ksize,
            lowpass_sigma=dataset.lowpass_sigma,
            detail_gain=dataset.detail_gain,
            gamma=dataset.gamma,
            gamma_c=dataset.gamma_c,
            return_debug=True,
        )

        original_gray = debug["original_gray"]
        enhanced_gray = np.clip(debug["enhanced_gamma"] * 255.0, 0, 255).astype(np.uint8)

        axes[row_idx, 0].imshow(original_gray, cmap="gray", vmin=0, vmax=255, aspect="auto")
        axes[row_idx, 0].set_title(f"Original: {image_name}")
        axes[row_idx, 0].axis("off")

        axes[row_idx, 1].imshow(enhanced_gray, cmap="gray", vmin=0, vmax=255, aspect="auto")
        axes[row_idx, 1].set_title(
            f"Enhanced (mask*lap + gamma)\n"
            f"gamma={dataset.gamma:.2f}, c={dataset.gamma_c:.2f}, gain={dataset.detail_gain:.2f}"
        )
        axes[row_idx, 1].axis("off")

        if show_debug_maps:
            mask_u8 = np.clip(debug["detail_mask"] * 255.0, 0, 255).astype(np.uint8)
            fine_detail_u8 = _normalize_for_display(debug["fine_detail"])

            axes[row_idx, 2].imshow(mask_u8, cmap="gray", vmin=0, vmax=255, aspect="auto")
            axes[row_idx, 2].set_title("Detail Mask (low-pass Sobel)")
            axes[row_idx, 2].axis("off")

            axes[row_idx, 3].imshow(fine_detail_u8, cmap="gray", vmin=0, vmax=255, aspect="auto")
            axes[row_idx, 3].set_title("Fine Detail (mask * Laplacian)")
            axes[row_idx, 3].axis("off")

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150)
        logger.info("Enhancement preview saved: %s", save_path)

    if show:
        plt.show()
    else:
        plt.close(fig)


# ============================================================
# Dataset
# ============================================================


class ChestXrayDataset(Dataset):
    """
    PyTorch Dataset for NIH Chest X-ray images.
    
    Multi-label classification: each image can have multiple disease findings.
    The labels are extracted from the 'Finding Labels' column in the metadata CSV.
    """
    
    def __init__(
        self,
        metadata_csv: str,
        image_dir: str,
        disease_labels: list = None,
        transform=None,
        sample_size: int = None,
        data_frame: pd.DataFrame = None,
        precomputed_labels: np.ndarray = None,
        preprocess_mode: str = "clahe",
        preprocess_kwargs: dict = None,
        # Legacy Laplacian+Sobel params (kept for backward compat)
        force_grayscale: bool = True,
        laplacian_ksize: int = 3,
        sobel_ksize: int = 3,
        lowpass_sigma: float = 1.2,
        detail_gain: float = 1.0,
        gamma: float = 0.5,
        gamma_c: float = 1.0,
    ):
        self.image_dir = image_dir
        self.transform = transform
        self.disease_labels = disease_labels or DISEASE_LABELS
        self._label_to_idx = {label: idx for idx, label in enumerate(self.disease_labels)}
        self.preprocess_mode = preprocess_mode
        self.preprocess_kwargs = preprocess_kwargs or {}
        # Legacy attrs kept for preview_enhancement_on_test_dataset compat
        self.force_grayscale = force_grayscale
        self.laplacian_ksize = laplacian_ksize
        self.sobel_ksize = sobel_ksize
        self.lowpass_sigma = lowpass_sigma
        self.detail_gain = detail_gain
        self.gamma = gamma
        self.gamma_c = gamma_c

        if data_frame is not None:
            self.df = data_frame.reset_index(drop=True).copy()
            logger.info(f"Loaded {len(self.df)} records from provided dataframe")
        else:
            # Load metadata
            self.df = pd.read_csv(metadata_csv)
            logger.info(f"Loaded {len(self.df)} records from {metadata_csv}")

            # Normalize column names and validate required columns
            rename_map = {}
            if "image_id" in self.df.columns:
                rename_map["image_id"] = "Image Index"
            if "labels" in self.df.columns:
                rename_map["labels"] = "Finding Labels"
            if rename_map:
                self.df.rename(columns=rename_map, inplace=True)

            required_cols = {"Image Index", "Finding Labels"}
            missing_cols = required_cols.difference(self.df.columns)
            if missing_cols:
                raise ValueError(
                    f"CSV missing required columns: {sorted(missing_cols)}. "
                    "Expected columns include 'Image Index' and 'Finding Labels'."
                )

            # Filter to only rows with existing images
            if Path(image_dir).exists():
                existing_images = set(f.name for f in Path(image_dir).iterdir() if f.is_file())
                self.df = self.df[self.df["Image Index"].isin(existing_images)].reset_index(drop=True)
                logger.info(f"After image matching: {len(self.df)} records")

            # Sample for quick testing
            if sample_size and sample_size < len(self.df):
                self.df = self.df.sample(n=sample_size, random_state=42).reset_index(drop=True)
                logger.info(f"Sampled {sample_size} records")

        if precomputed_labels is not None:
            if len(precomputed_labels) != len(self.df):
                raise ValueError("Length of precomputed_labels must match dataframe length")
            self.labels = precomputed_labels.astype(np.float32, copy=False)
        else:
            # Pre-compute multi-hot label vectors
            self.labels = self._compute_labels()
    
    def _compute_labels(self) -> np.ndarray:
        """Convert 'Finding Labels' strings to multi-hot vectors."""
        num_samples = len(self.df)
        num_classes = len(self.disease_labels)
        labels = np.zeros((num_samples, num_classes), dtype=np.float32)

        for row_idx, findings_str in enumerate(self.df["Finding Labels"].astype(str).tolist()):
            findings = findings_str.split("|")
            for finding in findings:
                finding = finding.strip()
                if finding in self._label_to_idx:
                    labels[row_idx, self._label_to_idx[finding]] = 1.0

        # Log label distribution
        if num_samples == 0:
            logger.warning("Dataset contains 0 samples after filtering")
        else:
            for i, label in enumerate(self.disease_labels):
                count = int(labels[:, i].sum())
                logger.info(f"  {label}: {count} ({count/num_samples*100:.1f}%)")
        
        return labels
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_name = row["Image Index"]
        
        # Load image with OpenCV (handle unicode paths correctly)
        img_path = os.path.join(self.image_dir, image_name)
        with open(img_path, "rb") as f:
            file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        image = preprocess_chest_xray(image, mode=self.preprocess_mode, **self.preprocess_kwargs)
        
        if self.transform:
            # torchvision transforms expect PIL or numpy array
            from PIL import Image as PILImage
            image = PILImage.fromarray(image)
            image = self.transform(image)
        
        # Multi-hot label vector
        label = torch.FloatTensor(self.labels[idx])
        
        return image, label


# ============================================================
# Training Logic
# ============================================================


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    epoch_idx: int,
    total_epochs: int,
    use_tqdm: bool = True,
    scaler=None,
    grad_clip: float = 1.0,
    label_smoothing: float = 0.0,
):
    """Train for one epoch with AMP, gradient clipping, and label smoothing."""
    model.train()
    if len(dataloader) == 0:
        raise ValueError("Training dataloader is empty")
    running_loss = 0.0
    use_amp = scaler is not None
    device_type = 'cuda' if device.type == 'cuda' else 'cpu'

    train_bar = tqdm(
        dataloader,
        total=len(dataloader),
        desc=f"Train {epoch_idx}/{total_epochs}",
        leave=False,
        dynamic_ncols=True,
        disable=not use_tqdm,
        position=1,
    )

    for batch_idx, (images, labels) in enumerate(train_bar, start=1):
        images = images.to(device)
        labels = labels.to(device)

        # Label smoothing for multi-label BCE
        if label_smoothing > 0:
            labels = labels * (1.0 - label_smoothing) + 0.5 * label_smoothing

        optimizer.zero_grad()

        with torch.amp.autocast(device_type=device_type, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        if use_amp:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        running_loss += loss.item()

        if use_tqdm:
            train_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                avg=f"{running_loss / batch_idx:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )
        elif batch_idx % 20 == 0:
            logger.info(f"  Batch {batch_idx}/{len(dataloader)} - Loss: {loss.item():.4f}")

    return running_loss / len(dataloader)


def evaluate(
    model,
    dataloader,
    criterion,
    device,
    disease_labels,
    epoch_idx: int,
    total_epochs: int,
    use_tqdm: bool = True,
    stage_name: str = "Val",
):
    """Evaluate the model."""
    model.eval()
    if len(dataloader) == 0:
        raise ValueError("Evaluation dataloader is empty")
    running_loss = 0.0
    all_preds = []
    all_labels = []

    eval_bar = tqdm(
        dataloader,
        total=len(dataloader),
        desc=f"{stage_name:<5} {epoch_idx}/{total_epochs}",
        leave=False,
        dynamic_ncols=True,
        disable=not use_tqdm,
        position=1,
    )

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(eval_bar, start=1):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            preds = torch.sigmoid(outputs)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            if use_tqdm:
                eval_bar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{running_loss / batch_idx:.4f}")
            elif batch_idx % 20 == 0:
                logger.info(f"  {stage_name} Batch {batch_idx}/{len(dataloader)} - Loss: {loss.item():.4f}")
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    avg_loss = running_loss / len(dataloader)
    
    # Compute AUC for each disease
    aucs = {}
    for i, label in enumerate(disease_labels):
        if all_labels[:, i].sum() > 0:
            try:
                auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
                aucs[label] = auc
            except ValueError:
                aucs[label] = 0.0
        else:
            aucs[label] = 0.0
    
    mean_auc = np.mean(list(aucs.values())) if aucs else 0.0
    
    return avg_loss, mean_auc, aucs, all_preds, all_labels


def _clone_with_subset(base_dataset: ChestXrayDataset, indices: np.ndarray, transform) -> ChestXrayDataset:
    """Create an in-memory dataset subset with its own transform."""
    subset = copy.copy(base_dataset)
    subset.transform = transform
    subset.df = base_dataset.df.iloc[indices].reset_index(drop=True).copy()
    subset.labels = base_dataset.labels[indices].copy()
    return subset


def _resolve_split_indices(
    dataset: ChestXrayDataset,
    split_dir: Path,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: int = 42,
    use_official_splits: bool = True,
):
    """
    Resolve train/val/test split indices.

    Preferred behavior:
    1) Use official NIH lists when available:
       - train_val_list.txt
       - test_list.txt
       Then split train_val into train/val using val_ratio.
    2) Fallback to deterministic random train/val/test split.
    """
    train_list_path = split_dir / "train_val_list.txt"
    test_list_path = split_dir / "test_list.txt"
    rng = np.random.default_rng(seed)

    if use_official_splits and train_list_path.exists() and test_list_path.exists():
        logger.info("Using official NIH train/test lists; deriving validation split from train_val")
        train_names = set(line.strip() for line in train_list_path.read_text().splitlines() if line.strip())
        test_names = set(line.strip() for line in test_list_path.read_text().splitlines() if line.strip())

        image_names = dataset.df["Image Index"]
        train_val_indices = np.where(image_names.isin(train_names).to_numpy())[0]
        test_indices = np.where(image_names.isin(test_names).to_numpy())[0]

        if len(train_val_indices) >= 2 and len(test_indices) >= 1:
            train_val_indices = train_val_indices.copy()
            rng.shuffle(train_val_indices)

            val_size = max(1, int(len(train_val_indices) * val_ratio))
            if val_size >= len(train_val_indices):
                val_size = len(train_val_indices) - 1

            val_indices = train_val_indices[:val_size]
            train_indices = train_val_indices[val_size:]

            if len(train_indices) > 0 and len(val_indices) > 0:
                return train_indices, val_indices, test_indices

        logger.warning(
            "Official split files found but could not produce non-empty train/val/test subsets. "
            "Falling back to random split."
        )

    total_samples = len(dataset)
    indices = np.arange(total_samples)
    rng.shuffle(indices)

    # Random fallback split (train/val/test)
    test_size = max(1, int(total_samples * test_ratio))
    if test_size >= total_samples:
        test_size = total_samples - 2

    remaining = total_samples - test_size
    val_size = max(1, int(remaining * val_ratio))
    if val_size >= remaining:
        val_size = remaining - 1

    train_size = remaining - val_size
    if train_size <= 0:
        train_size = 1
        val_size = max(1, remaining - train_size)

    train_indices = indices[:train_size]
    val_indices = indices[train_size: train_size + val_size]
    test_indices = indices[train_size + val_size:]

    return train_indices, val_indices, test_indices


def _compute_pos_weight(labels: np.ndarray, disease_labels: list, cap: float = 10.0) -> torch.Tensor:
    """Compute per-class positive weights: pos_weight = neg_count / pos_count."""
    if labels.ndim != 2:
        raise ValueError("labels must be a 2D array [num_samples, num_classes]")

    num_samples = labels.shape[0]
    positives = labels.sum(axis=0)
    negatives = num_samples - positives

    pos_weight = np.ones_like(positives, dtype=np.float32)
    valid = positives > 0
    pos_weight[valid] = negatives[valid] / positives[valid]
    pos_weight = np.maximum(pos_weight, 1.0)

    if cap is not None and cap > 0:
        pos_weight = np.minimum(pos_weight, float(cap))

    zero_pos_classes = [disease_labels[i] for i, p in enumerate(positives) if p == 0]
    if zero_pos_classes:
        logger.warning(
            "No positive samples in train split for classes: %s. Their pos_weight is kept at 1.0.",
            ", ".join(zero_pos_classes),
        )

    return torch.tensor(pos_weight, dtype=torch.float32)


class ChestXrayModel(nn.Module):
    """Multi-backbone model with dropout + BN classifier head."""

    SUPPORTED_BACKBONES = ("efficientnet_b4", "efficientnet_b3", "resnet50")

    def __init__(self, num_classes: int = 14, backbone: str = "efficientnet_b4",
                 pretrained: bool = True, dropout: float = 0.3, hidden_dim: int = 512):
        super().__init__()
        self.backbone_name = backbone
        self.num_classes = num_classes
        self._dropout_rate = dropout
        self._hidden_dim = hidden_dim

        if backbone == "efficientnet_b4":
            weights = models.EfficientNet_B4_Weights.DEFAULT if pretrained else None
            base = models.efficientnet_b4(weights=weights)
            num_features = base.classifier[1].in_features  # 1792
            base.classifier = nn.Identity()
            self.backbone = base
        elif backbone == "efficientnet_b3":
            weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
            base = models.efficientnet_b3(weights=weights)
            num_features = base.classifier[1].in_features  # 1536
            base.classifier = nn.Identity()
            self.backbone = base
        elif backbone == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            base = models.resnet50(weights=weights)
            num_features = base.fc.in_features  # 2048
            base.fc = nn.Identity()
            self.backbone = base
        else:
            raise ValueError(f"Unsupported backbone: {backbone}. Choose from {self.SUPPORTED_BACKBONES}")

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        if features.dim() > 2:
            features = features.view(features.size(0), -1)
        return self.classifier(features)


def _create_model(num_classes: int, backbone: str = "efficientnet_b4",
                  pretrained: bool = True, dropout: float = 0.3,
                  hidden_dim: int = 512) -> nn.Module:
    """Factory: create ChestXrayModel with the specified backbone."""
    return ChestXrayModel(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=pretrained,
        dropout=dropout,
        hidden_dim=hidden_dim,
    )


def _compute_sample_weights(labels: np.ndarray, disease_labels: list) -> np.ndarray:
    """Compute per-sample weights for WeightedRandomSampler (multi-label aware).

    Samples containing rarer diseases get higher weight so the
    sampler oversamples them, effectively augmenting rare classes more.
    """
    n_samples = labels.shape[0]
    class_counts = labels.sum(axis=0)  # per-class positive count

    class_weights = np.ones_like(class_counts, dtype=np.float64)
    valid = class_counts > 0
    class_weights[valid] = n_samples / (class_counts[valid] * len(disease_labels))

    sample_weights = np.ones(n_samples, dtype=np.float64)
    for i in range(n_samples):
        pos_classes = np.where(labels[i] > 0)[0]
        if len(pos_classes) > 0:
            # Weight driven by rarest positive class
            sample_weights[i] = float(np.max(class_weights[pos_classes]))

    return sample_weights


def _create_cosine_warmup_scheduler(optimizer, warmup_epochs: int, total_epochs: int,
                                     min_lr_ratio: float = 0.01):
    """Linear warmup + cosine decay scheduler."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return max(min_lr_ratio, (epoch + 1) / max(1, warmup_epochs))
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    parser = argparse.ArgumentParser(description="Train Chest X-ray CNN Model (v2 — optimized)")
    parser.add_argument("--epochs", type=int, default=120, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (16 for EfficientNet-B4+AMP on 6GB)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (head; backbone=lr*0.1)")
    parser.add_argument("--image-size", type=int, default=300, help="Input image size")
    parser.add_argument("--sample-size", type=int, default=None, help="Subset for quick testing")
    parser.add_argument("--output", type=str, default="./models/chest_xray_model.pth")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data-dir", type=str, default="./data/chest_xray")
    parser.add_argument("--metadata-csv", type=str, default=None)
    parser.add_argument("--image-dir", type=str, default=None)
    parser.add_argument("--no-tqdm", action="store_true")
    parser.add_argument("--save-top-k", type=int, default=3)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--disable-pos-weight", action="store_true")
    parser.add_argument("--pos-weight-cap", type=float, default=10.0)
    parser.add_argument("--lr-patience", type=int, default=4, help="ReduceLROnPlateau patience")
    parser.add_argument("--lr-factor", type=float, default=0.3, help="ReduceLROnPlateau factor")
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--early-stopping-patience", type=int, default=15)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--min-epochs", type=int, default=20)
    parser.add_argument("--disable-early-stopping", action="store_true")
    parser.add_argument("--eval-test-every", type=int, default=5)
    # --- Model architecture ---
    parser.add_argument("--backbone", type=str, default="efficientnet_b4",
                        choices=["efficientnet_b4", "efficientnet_b3", "resnet50"])
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--hidden-dim", type=int, default=512)
    # --- Preprocessing ---
    parser.add_argument("--preprocess-mode", type=str, default="clahe",
                        choices=["clahe", "laplacian_sobel", "none"])
    parser.add_argument("--clahe-clip-limit", type=float, default=2.0)
    parser.add_argument("--clahe-tile-size", type=int, default=8)
    # --- Training optimizations ---
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--enable-amp", action="store_true", default=True)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--scheduler", type=str, default="cosine_warmup",
                        choices=["cosine_warmup", "reduce_on_plateau"])
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--disable-weighted-sampler", action="store_true")
    # --- Legacy Laplacian+Sobel args ---
    parser.add_argument("--disable-force-grayscale", action="store_true")
    parser.add_argument("--laplacian-ksize", type=int, default=3)
    parser.add_argument("--sobel-ksize", type=int, default=3)
    parser.add_argument("--lowpass-sigma", type=float, default=1.2)
    parser.add_argument("--detail-gain", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--gamma-c", type=float, default=1.0)
    # --- Preview ---
    parser.add_argument("--preview-enhancement-test", action="store_true")
    parser.add_argument("--preview-enhancement-show", action="store_true")
    parser.add_argument("--preview-enhancement-debug", action="store_true")
    parser.add_argument("--preview-enhancement-count", type=int, default=6)
    parser.add_argument("--preview-enhancement-seed", type=int, default=42)
    parser.add_argument("--preview-enhancement-save-path", type=str, default=None)
    parser.add_argument("--preview-enhancement-only", action="store_true")
    args = parser.parse_args()
    if args.disable_amp:
        args.enable_amp = False

    if args.save_top_k < 0:
        logger.error("--save-top-k must be >= 0")
        sys.exit(1)
    if args.checkpoint_every < 0:
        logger.error("--checkpoint-every must be >= 0")
        sys.exit(1)
    if not (0.0 < args.val_ratio < 0.5):
        logger.error("--val-ratio must be between 0 and 0.5")
        sys.exit(1)
    if not (0.05 <= args.test_ratio < 0.5):
        logger.error("--test-ratio must be in [0.05, 0.5)")
        sys.exit(1)
    if args.pos_weight_cap <= 0:
        logger.error("--pos-weight-cap must be > 0")
        sys.exit(1)
    if args.lr_patience < 0:
        logger.error("--lr-patience must be >= 0")
        sys.exit(1)
    if not (0.0 < args.lr_factor < 1.0):
        logger.error("--lr-factor must be in (0, 1)")
        sys.exit(1)
    if args.min_lr < 0:
        logger.error("--min-lr must be >= 0")
        sys.exit(1)
    if args.early_stopping_patience < 0:
        logger.error("--early-stopping-patience must be >= 0")
        sys.exit(1)
    if args.early_stopping_min_delta < 0:
        logger.error("--early-stopping-min-delta must be >= 0")
        sys.exit(1)
    if args.min_epochs < 1:
        logger.error("--min-epochs must be >= 1")
        sys.exit(1)
    if args.eval_test_every < 0:
        logger.error("--eval-test-every must be >= 0")
        sys.exit(1)
    if args.laplacian_ksize < 1 or args.laplacian_ksize % 2 == 0:
        logger.error("--laplacian-ksize must be a positive odd integer")
        sys.exit(1)
    if args.sobel_ksize < 1 or args.sobel_ksize % 2 == 0:
        logger.error("--sobel-ksize must be a positive odd integer")
        sys.exit(1)
    if args.lowpass_sigma < 0:
        logger.error("--lowpass-sigma must be >= 0")
        sys.exit(1)
    if args.detail_gain < 0:
        logger.error("--detail-gain must be >= 0")
        sys.exit(1)
    if args.gamma <= 0:
        logger.error("--gamma must be > 0")
        sys.exit(1)
    if args.gamma_c <= 0:
        logger.error("--gamma-c must be > 0")
        sys.exit(1)
    if args.preview_enhancement_only and not args.preview_enhancement_test:
        logger.info(
            "--preview-enhancement-only enabled; auto-enabling --preview-enhancement-test."
        )
        args.preview_enhancement_test = True
    if args.preview_enhancement_count < 1:
        logger.error("--preview-enhancement-count must be >= 1")
        sys.exit(1)
    
    # Resolve paths
    data_dir = Path(args.data_dir)
    data_roots = [data_dir]
    archive_dir = data_dir / "archive"
    if archive_dir.exists():
        data_roots.append(archive_dir)
    
    # Try to find metadata CSV
    metadata_csv = args.metadata_csv
    if metadata_csv is None:
        for root in data_roots:
            candidates = [
                root / "Data_Entry_2017.csv",
                root / "Data_Entry_2017_v2020.csv",
                root / "metadata.csv",
                root / "chest_xray_metadata.csv",
            ]
            for candidate in candidates:
                if candidate.exists():
                    metadata_csv = str(candidate)
                    break
            if metadata_csv:
                break
    
    if metadata_csv is None or not Path(metadata_csv).exists():
        logger.error(
            f"Metadata CSV not found. Please download the NIH Chest X-ray dataset:\n"
            f"  kaggle datasets download -d nih-chest-xrays/data -p {data_dir}\n"
            f"Or specify --metadata-csv path"
        )
        sys.exit(1)
    
    # Try to find image directory
    image_dir = args.image_dir
    if image_dir is None:
        for root in data_roots:
            candidates = [
                root / "all_images",
                root / "images",
                root / "images_001" / "images",
                root / "sample",
            ]
            for candidate in candidates:
                if candidate.exists():
                    image_dir = str(candidate)
                    break
            if image_dir:
                break
    
    if image_dir is None or not Path(image_dir).exists():
        logger.error(
            f"Image directory not found at {data_dir}. "
            "Please download the dataset and specify --image-dir."
        )
        sys.exit(1)
    
    logger.info(f"Metadata CSV: {metadata_csv}")
    logger.info(f"Image directory: {image_dir}")
    logger.info(
        "Config -> backbone=%s, preprocess=%s, image_size=%d, dropout=%.2f, "
        "label_smooth=%.3f, AMP=%s, grad_clip=%.1f, scheduler=%s, weight_decay=%.1e",
        args.backbone, args.preprocess_mode, args.image_size, args.dropout,
        args.label_smoothing, args.enable_amp, args.grad_clip,
        args.scheduler, args.weight_decay,
    )

    # Build preprocessing kwargs based on mode
    preprocess_kwargs = {}
    if args.preprocess_mode == "clahe":
        preprocess_kwargs = {
            "clip_limit": args.clahe_clip_limit,
            "tile_grid_size": args.clahe_tile_size,
        }
    elif args.preprocess_mode == "laplacian_sobel":
        preprocess_kwargs = {
            "force_grayscale": not args.disable_force_grayscale,
            "laplacian_ksize": args.laplacian_ksize,
            "sobel_ksize": args.sobel_ksize,
            "lowpass_sigma": args.lowpass_sigma,
            "detail_gain": args.detail_gain,
            "gamma": args.gamma,
            "gamma_c": args.gamma_c,
        }

    # Transforms — aggressive augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=15, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 0.5))], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create base dataset
    full_dataset = ChestXrayDataset(
        metadata_csv=metadata_csv,
        image_dir=image_dir,
        disease_labels=DISEASE_LABELS,
        transform=None,
        sample_size=args.sample_size,
        preprocess_mode=args.preprocess_mode,
        preprocess_kwargs=preprocess_kwargs,
        force_grayscale=not args.disable_force_grayscale,
        laplacian_ksize=args.laplacian_ksize,
        sobel_ksize=args.sobel_ksize,
        lowpass_sigma=args.lowpass_sigma,
        detail_gain=args.detail_gain,
        gamma=args.gamma,
        gamma_c=args.gamma_c,
    )

    if len(full_dataset) < 3:
        logger.error("Need at least 3 samples to create train/val/test split.")
        sys.exit(1)

    split_dir = Path(metadata_csv).parent
    use_official_splits = args.sample_size is None
    if not use_official_splits:
        logger.info("sample-size is set: using random split instead of official NIH lists.")

    train_indices, val_indices, test_indices = _resolve_split_indices(
        full_dataset,
        split_dir=split_dir,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.split_seed,
        use_official_splits=use_official_splits,
    )

    if len(train_indices) == 0 or len(val_indices) == 0 or len(test_indices) == 0:
        logger.error(
            "Invalid split produced empty subset(s): "
            f"train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}"
        )
        sys.exit(1)

    train_dataset = _clone_with_subset(full_dataset, train_indices, train_transform)
    val_dataset = _clone_with_subset(full_dataset, val_indices, test_transform)
    test_dataset = _clone_with_subset(full_dataset, test_indices, test_transform)

    logger.info(
        f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples, "
        f"Test: {len(test_dataset)} samples"
    )

    if args.preview_enhancement_test:
        preview_save_path = (
            Path(args.preview_enhancement_save_path)
            if args.preview_enhancement_save_path
            else (Path(args.output).parent / "chest_xray_enhancement_preview.png")
        )
        preview_enhancement_on_test_dataset(
            dataset=test_dataset,
            sample_count=args.preview_enhancement_count,
            random_seed=args.preview_enhancement_seed,
            show=args.preview_enhancement_show,
            show_debug_maps=args.preview_enhancement_debug,
            save_path=preview_save_path,
        )

        if args.preview_enhancement_only:
            logger.info("Preview-only mode enabled. Exiting before training.")
            return

    # --- DataLoaders with optional WeightedRandomSampler ---
    pin_memory = args.device == "cuda" and torch.cuda.is_available()
    train_sampler = None
    shuffle_train = True
    if not args.disable_weighted_sampler:
        sample_weights = _compute_sample_weights(train_dataset.labels, DISEASE_LABELS)
        train_sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights).double(),
            num_samples=len(train_dataset),
            replacement=True,
        )
        shuffle_train = False  # sampler replaces shuffle
        logger.info(
            "WeightedRandomSampler enabled (min_w=%.4f, max_w=%.4f, mean_w=%.4f)",
            sample_weights.min(), sample_weights.max(), sample_weights.mean(),
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle_train,
        sampler=train_sampler,
        num_workers=2,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=pin_memory,
    )

    # --- Device ---
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    logger.info(f"Using device: {device}")

    # --- Model ---
    model = _create_model(
        num_classes=len(DISEASE_LABELS),
        backbone=args.backbone,
        pretrained=True,
        dropout=args.dropout,
        hidden_dim=args.hidden_dim,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {args.backbone} - {total_params:,} total params, {trainable_params:,} trainable")

    # --- Loss ---
    if args.disable_pos_weight:
        pos_weight = None
        logger.info("Class imbalance weighting disabled (plain BCEWithLogitsLoss).")
    else:
        pos_weight = _compute_pos_weight(
            labels=train_dataset.labels,
            disease_labels=DISEASE_LABELS,
            cap=args.pos_weight_cap,
        ).to(device)
        logger.info(
            "Using pos_weight with cap=%.2f (min=%.3f, max=%.3f, mean=%.3f)",
            args.pos_weight_cap,
            float(pos_weight.min().item()),
            float(pos_weight.max().item()),
            float(pos_weight.mean().item()),
        )

    train_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    eval_criterion = nn.BCEWithLogitsLoss()

    # --- Optimizer: Differential LR (backbone slow, head fast) + AdamW ---
    backbone_params = list(model.backbone.parameters())
    head_params = list(model.classifier.parameters())
    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": args.lr * 0.1},
        {"params": head_params, "lr": args.lr},
    ], weight_decay=args.weight_decay)
    logger.info(
        "AdamW optimizer: backbone_lr=%.2e, head_lr=%.2e, weight_decay=%.1e",
        args.lr * 0.1, args.lr, args.weight_decay,
    )

    # --- Scheduler ---
    if args.scheduler == "cosine_warmup":
        scheduler = _create_cosine_warmup_scheduler(
            optimizer, warmup_epochs=args.warmup_epochs,
            total_epochs=args.epochs, min_lr_ratio=args.min_lr / args.lr,
        )
        logger.info("Scheduler: cosine_warmup (warmup=%d epochs)", args.warmup_epochs)
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=args.lr_factor,
            patience=args.lr_patience, min_lr=args.min_lr,
        )
        logger.info("Scheduler: ReduceLROnPlateau (patience=%d, factor=%.2f)", args.lr_patience, args.lr_factor)

    # --- AMP Scaler ---
    use_amp = args.enable_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None
    if use_amp:
        logger.info("Mixed precision training (AMP) enabled")
    
    # Training loop
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    last_ckpt_path = output_path.parent / f"{output_path.stem}_last.pth"
    history_csv_path = output_path.parent / "chest_xray_training_history.csv"
    report_path = output_path.parent / "chest_xray_training_report.json"
    plot_path = output_path.parent / "chest_xray_training_curves.png"
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_mean_auc": [],
        "test_loss_monitor": [],
        "test_mean_auc_monitor": [],
        "lr": [],
    }
    top_k_checkpoints = []
    best_auc = float("-inf")  # Selection metric: best validation Mean AUC
    best_epoch = None
    best_per_disease_auc = {}
    stop_reason = "max_epochs_completed"
    epochs_without_improve = 0
    run_start_time = time.time()
    preprocessing_signature = {
        "pipeline": args.preprocess_mode,
        "image_size": int(args.image_size),
    }
    preprocessing_signature.update(preprocess_kwargs)
    model_config = {
        "backbone": args.backbone,
        "dropout": args.dropout,
        "hidden_dim": args.hidden_dim,
    }

    def _build_checkpoint_payload(
        epoch,
        train_loss,
        val_loss,
        val_mean_auc,
        val_aucs,
        test_loss_monitor=None,
        test_mean_auc_monitor=None,
    ):
        return {
            "model_state_dict": model.state_dict(),
            "class_labels": DISEASE_LABELS,
            "backbone": args.backbone,
            "model_config": model_config,
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_mean_auc": float(val_mean_auc),
            "per_disease_val_auc": {k: float(v) for k, v in val_aucs.items()},
            "test_loss_monitor": float(test_loss_monitor) if test_loss_monitor is not None else None,
            "test_mean_auc_monitor": (
                float(test_mean_auc_monitor) if test_mean_auc_monitor is not None else None
            ),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "selection_metric": "val_mean_auc",
            "best_val_mean_auc_so_far": float(best_auc if best_auc != float("-inf") else val_mean_auc),
            "preprocessing": preprocessing_signature,
            "args": vars(args),
        }

    def _save_epoch_report(status: str, final_test_results: dict = None):
        pd.DataFrame(history).to_csv(history_csv_path, index=False)

        run_duration_sec = time.time() - run_start_time
        final_epoch_idx = history["epoch"][-1] if history["epoch"] else None
        final_train_loss = history["train_loss"][-1] if history["train_loss"] else None
        final_val_loss = history["val_loss"][-1] if history["val_loss"] else None
        final_val_mean_auc = history["val_mean_auc"][-1] if history["val_mean_auc"] else None
        final_test_loss_monitor = history["test_loss_monitor"][-1] if history["test_loss_monitor"] else None
        final_test_auc_monitor = (
            history["test_mean_auc_monitor"][-1] if history["test_mean_auc_monitor"] else None
        )

        report = {
            "status": status,
            "last_updated_local": time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": round(run_duration_sec, 2),
            "epochs_requested": int(args.epochs),
            "epochs_completed": len(history["epoch"]),
            "stop_reason": stop_reason,
            "selection_metric": "val_mean_auc",
            "preprocessing": preprocessing_signature,
            "best_model": {
                "path": str(output_path),
                "epoch": best_epoch,
                "val_mean_auc": float(best_auc) if best_auc != float("-inf") else None,
                "per_disease_val_auc": best_per_disease_auc,
            },
            "last_checkpoint": str(last_ckpt_path),
            "top_k_checkpoints": top_k_checkpoints,
            "final_epoch_metrics": {
                "epoch": final_epoch_idx,
                "train_loss": float(final_train_loss) if final_train_loss is not None else None,
                "val_loss": float(final_val_loss) if final_val_loss is not None else None,
                "val_mean_auc": float(final_val_mean_auc) if final_val_mean_auc is not None else None,
                "test_loss_monitor": (
                    float(final_test_loss_monitor) if final_test_loss_monitor is not None else None
                ),
                "test_mean_auc_monitor": (
                    float(final_test_auc_monitor) if final_test_auc_monitor is not None else None
                ),
                "lr": float(history["lr"][-1]) if history["lr"] else None,
            },
            "final_test_evaluation": final_test_results,
            "artifacts": {
                "training_curves_png": str(plot_path),
                "history_csv": str(history_csv_path),
            },
        }

        report_tmp_path = report_path.with_suffix(report_path.suffix + ".tmp")
        report_tmp_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        report_tmp_path.replace(report_path)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting training: {args.epochs} epochs, batch_size={args.batch_size}")
    logger.info(f"{'='*60}\n")
    use_tqdm = (not args.no_tqdm) and sys.stdout.isatty() and TQDM_AVAILABLE
    if (not args.no_tqdm) and (not TQDM_AVAILABLE):
        logger.warning("tqdm is not installed. Falling back to standard logging.")

    epoch_bar = tqdm(
        range(1, args.epochs + 1),
        desc="Epochs",
        dynamic_ncols=True,
        disable=not use_tqdm,
        position=0,
        leave=True,
    )

    for epoch in epoch_bar:
        # Train
        train_loss = train_one_epoch(
            model,
            train_loader,
            train_criterion,
            optimizer,
            device,
            epoch_idx=epoch,
            total_epochs=args.epochs,
            use_tqdm=use_tqdm,
            scaler=scaler,
            grad_clip=args.grad_clip,
            label_smoothing=args.label_smoothing,
        )
        
        # Evaluate on validation set (used for model selection)
        val_loss, val_mean_auc, val_aucs, _, _ = evaluate(
            model,
            val_loader,
            eval_criterion,
            device,
            DISEASE_LABELS,
            epoch_idx=epoch,
            total_epochs=args.epochs,
            use_tqdm=use_tqdm,
            stage_name="Val",
        )

        test_loss_monitor = None
        test_mean_auc_monitor = None
        if args.eval_test_every > 0 and epoch % args.eval_test_every == 0:
            test_loss_monitor, test_mean_auc_monitor, _, _, _ = evaluate(
                model,
                test_loader,
                eval_criterion,
                device,
                DISEASE_LABELS,
                epoch_idx=epoch,
                total_epochs=args.epochs,
                use_tqdm=use_tqdm,
                stage_name="Test",
            )

        if args.scheduler == "cosine_warmup":
            scheduler.step()
        else:
            scheduler.step(val_mean_auc)
        current_lr = optimizer.param_groups[0]["lr"]
        if use_tqdm:
            epoch_bar.set_postfix(
                train_loss=f"{train_loss:.4f}",
                val_loss=f"{val_loss:.4f}",
                val_auc=f"{val_mean_auc:.4f}",
                lr=f"{current_lr:.2e}",
            )
        
        logger.info(
            f"Epoch {epoch}: Train Loss={train_loss:.4f} | "
            f"Val Loss={val_loss:.4f}, Val Mean AUC={val_mean_auc:.4f}, LR={current_lr:.2e}"
        )
        if test_mean_auc_monitor is not None:
            logger.info(
                f"  Test monitor: Loss={test_loss_monitor:.4f}, Mean AUC={test_mean_auc_monitor:.4f}"
            )

        # Log per-disease validation AUC
        for disease, auc in val_aucs.items():
            if auc > 0:
                logger.info(f"  {disease}: Val AUC={auc:.4f}")
        
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mean_auc"].append(val_mean_auc)
        history["test_loss_monitor"].append(test_loss_monitor)
        history["test_mean_auc_monitor"].append(test_mean_auc_monitor)
        history["lr"].append(current_lr)

        checkpoint_payload = _build_checkpoint_payload(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            val_mean_auc=val_mean_auc,
            val_aucs=val_aucs,
            test_loss_monitor=test_loss_monitor,
            test_mean_auc_monitor=test_mean_auc_monitor,
        )

        # Always keep the latest state (safe for long unattended runs)
        torch.save(checkpoint_payload, str(last_ckpt_path))

        # Optional periodic checkpoint snapshots
        if args.checkpoint_every > 0 and epoch % args.checkpoint_every == 0:
            periodic_path = output_path.parent / f"{output_path.stem}_epoch_{epoch:03d}.pth"
            torch.save(checkpoint_payload, str(periodic_path))
            logger.info(f"  Periodic checkpoint saved: {periodic_path}")

        # Save/refresh main best model path (selected by validation Mean AUC)
        improved = val_mean_auc > (best_auc + args.early_stopping_min_delta)
        if improved:
            best_auc = val_mean_auc
            best_epoch = epoch
            best_per_disease_auc = {k: float(v) for k, v in val_aucs.items()}
            checkpoint_payload["best_val_mean_auc_so_far"] = float(best_auc)
            torch.save(checkpoint_payload, str(output_path))
            logger.info(f"  Best model updated: {output_path} (Val Mean AUC: {best_auc:.4f})")
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        # Keep top-K best checkpoints by validation Mean AUC
        if args.save_top_k > 0:
            qualifies_for_top_k = (
                len(top_k_checkpoints) < args.save_top_k
                or val_mean_auc > top_k_checkpoints[-1]["val_mean_auc"]
            )
            if qualifies_for_top_k:
                top_k_path = (
                    output_path.parent / f"{output_path.stem}_top_e{epoch:03d}_valauc_{val_mean_auc:.4f}.pth"
                )
                torch.save(checkpoint_payload, str(top_k_path))
                top_k_checkpoints.append(
                    {"epoch": epoch, "val_mean_auc": float(val_mean_auc), "path": str(top_k_path)}
                )
                top_k_checkpoints.sort(key=lambda x: x["val_mean_auc"], reverse=True)

                while len(top_k_checkpoints) > args.save_top_k:
                    removed = top_k_checkpoints.pop()
                    removed_path = Path(removed["path"])
                    if removed_path.exists():
                        removed_path.unlink()

                logger.info(f"  Top-K checkpoint saved: {top_k_path}")

        # Update report artifacts each epoch for unattended runs
        _save_epoch_report(status="running")

        if (
            not args.disable_early_stopping
            and epoch >= args.min_epochs
            and epochs_without_improve >= args.early_stopping_patience
        ):
            stop_reason = (
                f"early_stopping(no val_mean_auc improvement > {args.early_stopping_min_delta} "
                f"for {epochs_without_improve} epochs)"
            )
            logger.info("Early stopping triggered at epoch %d. %s", epoch, stop_reason)
            break
    
    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"], label="Val Loss")
    if any(v is not None for v in history["test_loss_monitor"]):
        test_loss_monitor = [np.nan if v is None else v for v in history["test_loss_monitor"]]
        axes[0].plot(test_loss_monitor, label="Test Loss (Monitor)", linestyle="--", alpha=0.7)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Train/Val Loss")
    axes[0].legend()

    axes[1].plot(history["val_mean_auc"], label="Val Mean AUC", color="green")
    if any(v is not None for v in history["test_mean_auc_monitor"]):
        test_auc_monitor = [np.nan if v is None else v for v in history["test_mean_auc_monitor"]]
        axes[1].plot(test_auc_monitor, label="Test Mean AUC (Monitor)", linestyle="--", alpha=0.7)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUC")
    axes[1].set_title("Validation Mean AUC")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(str(plot_path), dpi=150)
    logger.info(f"Training curves saved to {plot_path}")

    # Final test evaluation: run once for best and last checkpoints
    final_eval_epoch = history["epoch"][-1] if history["epoch"] else 0

    def _evaluate_checkpoint_on_test(checkpoint_path: Path):
        if not checkpoint_path.exists():
            return None
        checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
        eval_model = _create_model(
            num_classes=len(DISEASE_LABELS), backbone=args.backbone,
            pretrained=False, dropout=args.dropout, hidden_dim=args.hidden_dim,
        ).to(device)
        eval_model.load_state_dict(checkpoint["model_state_dict"])
        test_loss, test_mean_auc, test_aucs, _, _ = evaluate(
            eval_model,
            test_loader,
            eval_criterion,
            device,
            DISEASE_LABELS,
            epoch_idx=final_eval_epoch,
            total_epochs=args.epochs,
            use_tqdm=use_tqdm,
            stage_name="Test",
        )
        return {
            "checkpoint_path": str(checkpoint_path),
            "test_loss": float(test_loss),
            "test_mean_auc": float(test_mean_auc),
            "per_disease_test_auc": {k: float(v) for k, v in test_aucs.items()},
        }

    final_test_results = {
        "best_model": _evaluate_checkpoint_on_test(output_path),
        "last_model": _evaluate_checkpoint_on_test(last_ckpt_path),
    }

    _save_epoch_report(status="completed", final_test_results=final_test_results)
    logger.info(f"Training history saved to {history_csv_path}")
    logger.info(f"Training report saved to {report_path}")

    logger.info(f"\nTraining complete! Best Val Mean AUC: {best_auc:.4f} (epoch={best_epoch})")
    logger.info(f"Best model path: {output_path}")
    logger.info(f"Last checkpoint path: {last_ckpt_path}")
    if final_test_results.get("best_model") is not None:
        logger.info(
            "Final test (best model): loss=%.4f, mean_auc=%.4f",
            final_test_results["best_model"]["test_loss"],
            final_test_results["best_model"]["test_mean_auc"],
        )
    if top_k_checkpoints:
        logger.info("Top-K checkpoints by val_mean_auc:")
        for rank, ckpt in enumerate(top_k_checkpoints, start=1):
            logger.info(
                f"  {rank}. epoch={ckpt['epoch']}, val_mean_auc={ckpt['val_mean_auc']:.4f}, path={ckpt['path']}"
            )


if __name__ == "__main__":
    main()

