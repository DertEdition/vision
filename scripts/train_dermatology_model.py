"""
Dermatology Model Training Script

Train a dual-head CNN model on the DERM12345 dataset for:
1. Malignancy classification
2. Disease sub-type classification

This script is designed for robust long-running training:
- Train/Val/Test protocol (model selection on validation only)
- Class imbalance handling via weighted losses
- ReduceLROnPlateau + optional early stopping
- Best/last/top-k checkpointing
- Epoch-level report persistence (JSON + CSV)
- Terminal monitoring with tqdm progress bars
"""

import argparse
import copy
import json
import logging
import os
import sys
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

try:
    from tqdm.auto import tqdm

    TQDM_AVAILABLE = True
except ImportError:  # pragma: no cover - fallback for minimal environments
    TQDM_AVAILABLE = False

    def tqdm(iterable=None, *args, **kwargs):
        return iterable


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DERM12345Dataset(Dataset):
    """Dataset for DERM12345 skin lesion images."""

    def __init__(
        self,
        metadata_csv: str = None,
        image_dirs: list = None,
        malignancy_classes: list = None,
        disease_classes: list = None,
        transform=None,
        sample_size: int = None,
        data_frame: pd.DataFrame = None,
        image_paths: dict = None,
    ):
        self.transform = transform
        self.malignancy_classes = malignancy_classes or []
        self.disease_classes = disease_classes or []
        self.malignancy_to_idx = {c: i for i, c in enumerate(self.malignancy_classes)}
        self.disease_to_idx = {c: i for i, c in enumerate(self.disease_classes)}

        if data_frame is not None:
            self.df = data_frame.reset_index(drop=True).copy()
            logger.info("Loaded %d records from provided dataframe", len(self.df))
        else:
            if metadata_csv is None:
                raise ValueError("metadata_csv is required when data_frame is not provided")
            self.df = pd.read_csv(metadata_csv)
            logger.info("Loaded %d records from %s", len(self.df), metadata_csv)

        required_cols = {"image_id", "malignancy", "sub_class"}
        missing_cols = required_cols.difference(self.df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns in metadata: {sorted(missing_cols)}")

        self.df = self.df[self.df["malignancy"].isin(self.malignancy_classes)]
        self.df = self.df[self.df["sub_class"].isin(self.disease_classes)]
        logger.info("After class filtering: %d records", len(self.df))

        if image_paths is not None:
            self.image_paths = dict(image_paths)
        else:
            self.image_paths = {}
            for img_dir in image_dirs or []:
                p = Path(img_dir)
                if not p.exists():
                    continue
                for f in p.rglob("*"):
                    if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                        self.image_paths[f.stem] = str(f)

        self.df = self.df[self.df["image_id"].isin(self.image_paths.keys())].reset_index(drop=True)
        logger.info("After image matching: %d records", len(self.df))

        if sample_size and sample_size < len(self.df):
            self.df = self.df.sample(n=sample_size, random_state=42).reset_index(drop=True)
            logger.info("Sampled %d records", sample_size)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row["image_id"]
        img_path = self.image_paths[image_id]

        with open(img_path, "rb") as f:
            file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            from PIL import Image as PILImage

            image = PILImage.fromarray(image)
            image = self.transform(image)

        mal_label = self.malignancy_to_idx[row["malignancy"]]
        dis_label = self.disease_to_idx[row["sub_class"]]
        return image, mal_label, dis_label


class DermatologyNet(nn.Module):
    """ResNet18 backbone with dual heads."""

    def __init__(self, num_malignancy_classes: int, num_disease_classes: int, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

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

    def forward(self, x):
        features = self.backbone(x)
        malignancy_out = self.malignancy_head(features)
        disease_out = self.disease_head(features)
        return malignancy_out, disease_out


def _create_model(num_malignancy_classes: int, num_disease_classes: int, pretrained: bool = True) -> nn.Module:
    return DermatologyNet(
        num_malignancy_classes=num_malignancy_classes,
        num_disease_classes=num_disease_classes,
        pretrained=pretrained,
    )


def _clone_with_subset(base_dataset: DERM12345Dataset, indices: np.ndarray, transform) -> DERM12345Dataset:
    subset = copy.copy(base_dataset)
    subset.transform = transform
    subset.df = base_dataset.df.iloc[indices].reset_index(drop=True).copy()
    return subset


def _split_train_val_indices(dataset: DERM12345Dataset, val_ratio: float, seed: int):
    indices = np.arange(len(dataset))
    if len(indices) < 2:
        raise ValueError("Need at least 2 samples to split into train/val")

    stratify_values = dataset.df["malignancy"].to_numpy()
    label_counts = pd.Series(stratify_values).value_counts()
    can_stratify = (label_counts >= 2).all()

    try:
        train_idx, val_idx = train_test_split(
            indices,
            test_size=val_ratio,
            random_state=seed,
            shuffle=True,
            stratify=stratify_values if can_stratify else None,
        )
    except ValueError:
        logger.warning("Stratified split failed. Falling back to random split.")
        rng = np.random.default_rng(seed)
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        val_size = max(1, int(len(shuffled) * val_ratio))
        if val_size >= len(shuffled):
            val_size = len(shuffled) - 1
        val_idx = shuffled[:val_size]
        train_idx = shuffled[val_size:]

    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError(f"Invalid split produced empty subset(s): train={len(train_idx)}, val={len(val_idx)}")

    return np.array(train_idx), np.array(val_idx)


def _compute_class_weights(label_indices: np.ndarray, class_names: list, cap: float = 10.0) -> torch.Tensor:
    num_classes = len(class_names)
    counts = np.bincount(label_indices, minlength=num_classes).astype(np.float32)
    total = float(np.sum(counts))

    weights = np.ones_like(counts, dtype=np.float32)
    valid = counts > 0
    weights[valid] = total / (num_classes * counts[valid])

    if cap and cap > 0:
        weights = np.minimum(weights, float(cap))

    missing_classes = [class_names[i] for i, c in enumerate(counts) if c == 0]
    if missing_classes:
        logger.warning(
            "No training samples for classes: %s. Weight kept at 1.0 for those classes.",
            ", ".join(missing_classes),
        )

    return torch.tensor(weights, dtype=torch.float32)


def _compute_metrics(labels: np.ndarray, preds: np.ndarray):
    acc = float(np.mean(preds == labels)) if len(labels) > 0 else 0.0
    macro_f1 = float(f1_score(labels, preds, average="macro", zero_division=0)) if len(labels) > 0 else 0.0
    return acc, macro_f1


def train_one_epoch(
    model,
    dataloader,
    criterion_mal,
    criterion_dis,
    optimizer,
    device,
    epoch_idx: int,
    total_epochs: int,
    mal_loss_weight: float = 1.0,
    dis_loss_weight: float = 1.0,
    use_tqdm: bool = True,
):
    model.train()
    if len(dataloader) == 0:
        raise ValueError("Training dataloader is empty")

    running_loss = 0.0
    mal_preds = []
    mal_labels_all = []
    dis_preds = []
    dis_labels_all = []

    train_bar = tqdm(
        dataloader,
        total=len(dataloader),
        desc=f"Train {epoch_idx}/{total_epochs}",
        leave=False,
        dynamic_ncols=True,
        disable=not use_tqdm,
        position=1,
    )

    for batch_idx, (images, mal_labels, dis_labels) in enumerate(train_bar, start=1):
        images = images.to(device)
        mal_labels = mal_labels.to(device)
        dis_labels = dis_labels.to(device)

        optimizer.zero_grad()
        mal_out, dis_out = model(images)

        loss_mal = criterion_mal(mal_out, mal_labels)
        loss_dis = criterion_dis(dis_out, dis_labels)
        loss = mal_loss_weight * loss_mal + dis_loss_weight * loss_dis

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        mal_pred = mal_out.argmax(1)
        dis_pred = dis_out.argmax(1)
        mal_preds.extend(mal_pred.detach().cpu().numpy())
        dis_preds.extend(dis_pred.detach().cpu().numpy())
        mal_labels_all.extend(mal_labels.detach().cpu().numpy())
        dis_labels_all.extend(dis_labels.detach().cpu().numpy())

        if use_tqdm:
            train_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                avg=f"{running_loss / batch_idx:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )
        elif batch_idx % 20 == 0:
            logger.info("  Batch %d/%d - Loss: %.4f", batch_idx, len(dataloader), loss.item())

    mal_acc, mal_macro_f1 = _compute_metrics(np.array(mal_labels_all), np.array(mal_preds))
    dis_acc, dis_macro_f1 = _compute_metrics(np.array(dis_labels_all), np.array(dis_preds))
    selection_score = 0.5 * (mal_macro_f1 + dis_macro_f1)

    return {
        "loss": running_loss / len(dataloader),
        "mal_acc": mal_acc,
        "dis_acc": dis_acc,
        "mal_macro_f1": mal_macro_f1,
        "dis_macro_f1": dis_macro_f1,
        "selection_score": selection_score,
    }


def evaluate(
    model,
    dataloader,
    criterion_mal,
    criterion_dis,
    device,
    epoch_idx: int,
    total_epochs: int,
    stage_name: str = "Val",
    mal_loss_weight: float = 1.0,
    dis_loss_weight: float = 1.0,
    use_tqdm: bool = True,
):
    model.eval()
    if len(dataloader) == 0:
        raise ValueError(f"{stage_name} dataloader is empty")

    running_loss = 0.0
    all_mal_preds = []
    all_mal_labels = []
    all_dis_preds = []
    all_dis_labels = []

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
        for batch_idx, (images, mal_labels, dis_labels) in enumerate(eval_bar, start=1):
            images = images.to(device)
            mal_labels = mal_labels.to(device)
            dis_labels = dis_labels.to(device)

            mal_out, dis_out = model(images)
            loss_mal = criterion_mal(mal_out, mal_labels)
            loss_dis = criterion_dis(dis_out, dis_labels)
            loss = mal_loss_weight * loss_mal + dis_loss_weight * loss_dis
            running_loss += loss.item()

            mal_pred = mal_out.argmax(1)
            dis_pred = dis_out.argmax(1)
            all_mal_preds.extend(mal_pred.cpu().numpy())
            all_mal_labels.extend(mal_labels.cpu().numpy())
            all_dis_preds.extend(dis_pred.cpu().numpy())
            all_dis_labels.extend(dis_labels.cpu().numpy())

            if use_tqdm:
                eval_bar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{running_loss / batch_idx:.4f}")
            elif batch_idx % 20 == 0:
                logger.info("  %s Batch %d/%d - Loss: %.4f", stage_name, batch_idx, len(dataloader), loss.item())

    mal_labels_np = np.array(all_mal_labels)
    mal_preds_np = np.array(all_mal_preds)
    dis_labels_np = np.array(all_dis_labels)
    dis_preds_np = np.array(all_dis_preds)

    mal_acc, mal_macro_f1 = _compute_metrics(mal_labels_np, mal_preds_np)
    dis_acc, dis_macro_f1 = _compute_metrics(dis_labels_np, dis_preds_np)
    selection_score = 0.5 * (mal_macro_f1 + dis_macro_f1)

    return {
        "loss": running_loss / len(dataloader),
        "mal_acc": mal_acc,
        "dis_acc": dis_acc,
        "mal_macro_f1": mal_macro_f1,
        "dis_macro_f1": dis_macro_f1,
        "selection_score": selection_score,
        "mal_preds": mal_preds_np,
        "mal_labels": mal_labels_np,
        "dis_preds": dis_preds_np,
        "dis_labels": dis_labels_np,
    }


def main():
    parser = argparse.ArgumentParser(description="Train Dermatology CNN Model (robust mode)")
    parser.add_argument("--epochs", type=int, default=120, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--image-size", type=int, default=224, help="Input image size")
    parser.add_argument("--sample-size", type=int, default=None, help="Use a subset of training samples")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--data-dir", type=str, default="./data/DERM12345", help="DERM12345 base directory")
    parser.add_argument("--train-csv", type=str, default=None, help="Optional train metadata CSV path")
    parser.add_argument("--test-csv", type=str, default=None, help="Optional test metadata CSV path")
    parser.add_argument("--output", type=str, default="./models/dermatology_model.pth", help="Best model output path")

    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation ratio from train split")
    parser.add_argument("--split-seed", type=int, default=42, help="Split random seed")
    parser.add_argument("--save-top-k", type=int, default=5, help="Keep top-K checkpoints by val selection metric")
    parser.add_argument("--checkpoint-every", type=int, default=5, help="Save periodic checkpoint every N epochs")

    parser.add_argument("--disable-class-weight", action="store_true", help="Disable class-weighted losses")
    parser.add_argument("--class-weight-cap", type=float, default=10.0, help="Upper cap for computed class weights")
    parser.add_argument("--mal-loss-weight", type=float, default=1.0, help="Malignancy loss multiplier")
    parser.add_argument("--dis-loss-weight", type=float, default=1.0, help="Disease loss multiplier")

    parser.add_argument("--lr-patience", type=int, default=2, help="ReduceLROnPlateau patience")
    parser.add_argument("--lr-factor", type=float, default=0.5, help="ReduceLROnPlateau factor")
    parser.add_argument("--min-lr", type=float, default=1e-6, help="Minimum LR for scheduler")

    parser.add_argument("--early-stopping-patience", type=int, default=8, help="Early stopping patience")
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=1e-4,
        help="Minimum val score gain to reset patience",
    )
    parser.add_argument("--min-epochs", type=int, default=10, help="Minimum epochs before early stopping")
    parser.add_argument("--disable-early-stopping", action="store_true", help="Disable early stopping")
    parser.add_argument("--eval-test-every", type=int, default=0, help="Evaluate test every N epochs (0=off)")
    parser.add_argument("--no-tqdm", action="store_true", help="Disable tqdm progress bars")
    args = parser.parse_args()

    if args.save_top_k < 0:
        logger.error("--save-top-k must be >= 0")
        sys.exit(1)
    if args.checkpoint_every < 0:
        logger.error("--checkpoint-every must be >= 0")
        sys.exit(1)
    if not (0.0 < args.val_ratio < 0.5):
        logger.error("--val-ratio must be between 0 and 0.5")
        sys.exit(1)
    if args.class_weight_cap <= 0:
        logger.error("--class-weight-cap must be > 0")
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
    if args.mal_loss_weight <= 0 or args.dis_loss_weight <= 0:
        logger.error("--mal-loss-weight and --dis-loss-weight must be > 0")
        sys.exit(1)

    def _resolve_from_project(path_value: str) -> Path:
        candidate = Path(path_value)
        if candidate.is_absolute():
            return candidate.resolve()
        return (PROJECT_ROOT / candidate).resolve()

    data_dir = _resolve_from_project(args.data_dir)
    train_csv = (
        _resolve_from_project(args.train_csv) if args.train_csv else (data_dir / "derm12345_metadata_train.csv")
    )
    test_csv = _resolve_from_project(args.test_csv) if args.test_csv else (data_dir / "derm12345_metadata_test.csv")
    train_image_dirs = [str(data_dir / "derm12345_train_part_1"), str(data_dir / "derm12345_train_part_2")]
    test_image_dirs = [str(data_dir / "derm12345_test")]

    if not train_csv.exists():
        logger.error("Train CSV not found: %s", train_csv)
        sys.exit(1)
    if not test_csv.exists():
        logger.error("Test CSV not found: %s", test_csv)
        sys.exit(1)

    train_df = pd.read_csv(train_csv)
    malignancy_classes = sorted(train_df["malignancy"].dropna().unique().tolist())
    disease_classes = sorted(train_df["sub_class"].dropna().unique().tolist())
    logger.info("Malignancy classes (%d): %s", len(malignancy_classes), malignancy_classes)
    logger.info("Disease classes (%d): %s", len(disease_classes), disease_classes)

    train_transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    base_train_dataset = DERM12345Dataset(
        metadata_csv=str(train_csv),
        image_dirs=train_image_dirs,
        malignancy_classes=malignancy_classes,
        disease_classes=disease_classes,
        transform=None,
        sample_size=args.sample_size,
    )
    if len(base_train_dataset) < 3:
        logger.error("Need at least 3 samples in training pool for train/val split.")
        sys.exit(1)

    train_indices, val_indices = _split_train_val_indices(
        dataset=base_train_dataset,
        val_ratio=args.val_ratio,
        seed=args.split_seed,
    )
    train_dataset = _clone_with_subset(base_train_dataset, train_indices, train_transform)
    val_dataset = _clone_with_subset(base_train_dataset, val_indices, eval_transform)

    test_sample_size = max(1, args.sample_size // 5) if args.sample_size else None
    test_dataset = DERM12345Dataset(
        metadata_csv=str(test_csv),
        image_dirs=test_image_dirs,
        malignancy_classes=malignancy_classes,
        disease_classes=disease_classes,
        transform=eval_transform,
        sample_size=test_sample_size,
    )

    if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
        logger.error(
            "Invalid dataset sizes: train=%d, val=%d, test=%d",
            len(train_dataset),
            len(val_dataset),
            len(test_dataset),
        )
        sys.exit(1)

    logger.info(
        "Dataset split sizes -> train=%d, val=%d, test=%d",
        len(train_dataset),
        len(val_dataset),
        len(test_dataset),
    )

    pin_memory = args.device == "cuda" and torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin_memory,
    )

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    logger.info("Using device: %s", device)

    model = _create_model(
        num_malignancy_classes=len(malignancy_classes),
        num_disease_classes=len(disease_classes),
        pretrained=True,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model params: %s total, %s trainable", f"{total_params:,}", f"{trainable_params:,}")

    if args.disable_class_weight:
        mal_class_weights = None
        dis_class_weights = None
        logger.info("Class-weighted losses disabled.")
    else:
        train_mal = train_dataset.df["malignancy"].map(train_dataset.malignancy_to_idx).to_numpy(dtype=np.int64)
        train_dis = train_dataset.df["sub_class"].map(train_dataset.disease_to_idx).to_numpy(dtype=np.int64)
        mal_class_weights = _compute_class_weights(train_mal, malignancy_classes, cap=args.class_weight_cap).to(device)
        dis_class_weights = _compute_class_weights(train_dis, disease_classes, cap=args.class_weight_cap).to(device)
        logger.info(
            "Malignancy class weights -> min=%.3f max=%.3f mean=%.3f",
            float(mal_class_weights.min().item()),
            float(mal_class_weights.max().item()),
            float(mal_class_weights.mean().item()),
        )
        logger.info(
            "Disease class weights -> min=%.3f max=%.3f mean=%.3f",
            float(dis_class_weights.min().item()),
            float(dis_class_weights.max().item()),
            float(dis_class_weights.mean().item()),
        )

    train_criterion_mal = nn.CrossEntropyLoss(weight=mal_class_weights)
    train_criterion_dis = nn.CrossEntropyLoss(weight=dis_class_weights)
    eval_criterion_mal = nn.CrossEntropyLoss()
    eval_criterion_dis = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=args.min_lr,
    )

    output_path = _resolve_from_project(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    last_ckpt_path = output_path.parent / f"{output_path.stem}_last.pth"
    history_csv_path = output_path.parent / "dermatology_training_history.csv"
    report_path = output_path.parent / "dermatology_training_report.json"
    plot_path = output_path.parent / "dermatology_training_curves.png"

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_mal_acc": [],
        "val_dis_acc": [],
        "val_mal_macro_f1": [],
        "val_dis_macro_f1": [],
        "val_selection_score": [],
        "test_loss_monitor": [],
        "test_mal_acc_monitor": [],
        "test_dis_acc_monitor": [],
        "test_mal_macro_f1_monitor": [],
        "test_dis_macro_f1_monitor": [],
        "test_selection_score_monitor": [],
        "lr": [],
    }
    top_k_checkpoints = []
    best_score = float("-inf")
    best_epoch = None
    best_metrics = {}
    stop_reason = "max_epochs_completed"
    epochs_without_improve = 0
    run_start_time = time.time()

    def _build_checkpoint_payload(epoch, train_metrics, val_metrics, test_metrics_monitor=None):
        return {
            "model_state_dict": model.state_dict(),
            "malignancy_classes": malignancy_classes,
            "disease_classes": disease_classes,
            "epoch": epoch,
            "train_metrics": {k: float(v) for k, v in train_metrics.items()},
            "val_metrics": {
                "loss": float(val_metrics["loss"]),
                "mal_acc": float(val_metrics["mal_acc"]),
                "dis_acc": float(val_metrics["dis_acc"]),
                "mal_macro_f1": float(val_metrics["mal_macro_f1"]),
                "dis_macro_f1": float(val_metrics["dis_macro_f1"]),
                "selection_score": float(val_metrics["selection_score"]),
            },
            "test_metrics_monitor": None
            if test_metrics_monitor is None
            else {
                "loss": float(test_metrics_monitor["loss"]),
                "mal_acc": float(test_metrics_monitor["mal_acc"]),
                "dis_acc": float(test_metrics_monitor["dis_acc"]),
                "mal_macro_f1": float(test_metrics_monitor["mal_macro_f1"]),
                "dis_macro_f1": float(test_metrics_monitor["dis_macro_f1"]),
                "selection_score": float(test_metrics_monitor["selection_score"]),
            },
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "selection_metric": "val_combined_macro_f1",
            "best_val_selection_score_so_far": float(
                best_score if best_score != float("-inf") else val_metrics["selection_score"]
            ),
            "args": vars(args),
        }

    def _save_epoch_report(status: str, final_test_results: dict = None):
        pd.DataFrame(history).to_csv(history_csv_path, index=False)

        duration_sec = time.time() - run_start_time
        final_epoch = history["epoch"][-1] if history["epoch"] else None
        final_train_loss = history["train_loss"][-1] if history["train_loss"] else None
        final_val_loss = history["val_loss"][-1] if history["val_loss"] else None
        final_val_mal_f1 = history["val_mal_macro_f1"][-1] if history["val_mal_macro_f1"] else None
        final_val_dis_f1 = history["val_dis_macro_f1"][-1] if history["val_dis_macro_f1"] else None
        final_val_score = history["val_selection_score"][-1] if history["val_selection_score"] else None
        final_lr = history["lr"][-1] if history["lr"] else None

        report = {
            "status": status,
            "last_updated_local": time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": round(duration_sec, 2),
            "epochs_requested": int(args.epochs),
            "epochs_completed": len(history["epoch"]),
            "stop_reason": stop_reason,
            "selection_metric": "val_combined_macro_f1",
            "best_model": {
                "path": str(output_path),
                "epoch": best_epoch,
                "val_selection_score": float(best_score) if best_score != float("-inf") else None,
                "val_metrics": best_metrics,
            },
            "last_checkpoint": str(last_ckpt_path),
            "top_k_checkpoints": top_k_checkpoints,
            "final_epoch_metrics": {
                "epoch": final_epoch,
                "train_loss": float(final_train_loss) if final_train_loss is not None else None,
                "val_loss": float(final_val_loss) if final_val_loss is not None else None,
                "val_mal_macro_f1": float(final_val_mal_f1) if final_val_mal_f1 is not None else None,
                "val_dis_macro_f1": float(final_val_dis_f1) if final_val_dis_f1 is not None else None,
                "val_selection_score": float(final_val_score) if final_val_score is not None else None,
                "lr": float(final_lr) if final_lr is not None else None,
            },
            "final_test_evaluation": final_test_results,
            "artifacts": {
                "training_curves_png": str(plot_path),
                "history_csv": str(history_csv_path),
            },
        }

        report_tmp = report_path.with_suffix(report_path.suffix + ".tmp")
        report_tmp.write_text(json.dumps(report, indent=2), encoding="utf-8")
        report_tmp.replace(report_path)

    use_tqdm = (not args.no_tqdm) and sys.stdout.isatty() and TQDM_AVAILABLE
    if (not args.no_tqdm) and (not TQDM_AVAILABLE):
        logger.warning("tqdm is not installed. Falling back to standard logging.")

    logger.info("%s", "=" * 70)
    logger.info(
        "Starting dermatology training: epochs=%d, batch_size=%d, val_ratio=%.2f",
        args.epochs,
        args.batch_size,
        args.val_ratio,
    )
    logger.info("%s", "=" * 70)

    epoch_bar = tqdm(
        range(1, args.epochs + 1),
        desc="Epochs",
        dynamic_ncols=True,
        disable=not use_tqdm,
        position=0,
        leave=True,
    )

    for epoch in epoch_bar:
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion_mal=train_criterion_mal,
            criterion_dis=train_criterion_dis,
            optimizer=optimizer,
            device=device,
            epoch_idx=epoch,
            total_epochs=args.epochs,
            mal_loss_weight=args.mal_loss_weight,
            dis_loss_weight=args.dis_loss_weight,
            use_tqdm=use_tqdm,
        )

        val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            criterion_mal=eval_criterion_mal,
            criterion_dis=eval_criterion_dis,
            device=device,
            epoch_idx=epoch,
            total_epochs=args.epochs,
            stage_name="Val",
            mal_loss_weight=args.mal_loss_weight,
            dis_loss_weight=args.dis_loss_weight,
            use_tqdm=use_tqdm,
        )

        test_monitor_metrics = None
        if args.eval_test_every > 0 and epoch % args.eval_test_every == 0:
            test_monitor_metrics = evaluate(
                model=model,
                dataloader=test_loader,
                criterion_mal=eval_criterion_mal,
                criterion_dis=eval_criterion_dis,
                device=device,
                epoch_idx=epoch,
                total_epochs=args.epochs,
                stage_name="Test",
                mal_loss_weight=args.mal_loss_weight,
                dis_loss_weight=args.dis_loss_weight,
                use_tqdm=use_tqdm,
            )

        scheduler.step(val_metrics["selection_score"])
        current_lr = optimizer.param_groups[0]["lr"]

        if use_tqdm:
            epoch_bar.set_postfix(
                train_loss=f"{train_metrics['loss']:.4f}",
                val_score=f"{val_metrics['selection_score']:.4f}",
                lr=f"{current_lr:.2e}",
            )

        logger.info(
            "Epoch %d: Train Loss=%.4f | Val Loss=%.4f | "
            "Val Mal Acc=%.3f, Val Dis Acc=%.3f | Val Mal F1=%.3f, Val Dis F1=%.3f | Val Score=%.4f | LR=%.2e",
            epoch,
            train_metrics["loss"],
            val_metrics["loss"],
            val_metrics["mal_acc"],
            val_metrics["dis_acc"],
            val_metrics["mal_macro_f1"],
            val_metrics["dis_macro_f1"],
            val_metrics["selection_score"],
            current_lr,
        )
        if test_monitor_metrics is not None:
            logger.info(
                "  Test monitor -> Loss=%.4f, Mal Acc=%.3f, Dis Acc=%.3f, Score=%.4f",
                test_monitor_metrics["loss"],
                test_monitor_metrics["mal_acc"],
                test_monitor_metrics["dis_acc"],
                test_monitor_metrics["selection_score"],
            )

        history["epoch"].append(epoch)
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_mal_acc"].append(val_metrics["mal_acc"])
        history["val_dis_acc"].append(val_metrics["dis_acc"])
        history["val_mal_macro_f1"].append(val_metrics["mal_macro_f1"])
        history["val_dis_macro_f1"].append(val_metrics["dis_macro_f1"])
        history["val_selection_score"].append(val_metrics["selection_score"])
        history["lr"].append(current_lr)

        if test_monitor_metrics is None:
            history["test_loss_monitor"].append(None)
            history["test_mal_acc_monitor"].append(None)
            history["test_dis_acc_monitor"].append(None)
            history["test_mal_macro_f1_monitor"].append(None)
            history["test_dis_macro_f1_monitor"].append(None)
            history["test_selection_score_monitor"].append(None)
        else:
            history["test_loss_monitor"].append(test_monitor_metrics["loss"])
            history["test_mal_acc_monitor"].append(test_monitor_metrics["mal_acc"])
            history["test_dis_acc_monitor"].append(test_monitor_metrics["dis_acc"])
            history["test_mal_macro_f1_monitor"].append(test_monitor_metrics["mal_macro_f1"])
            history["test_dis_macro_f1_monitor"].append(test_monitor_metrics["dis_macro_f1"])
            history["test_selection_score_monitor"].append(test_monitor_metrics["selection_score"])

        checkpoint_payload = _build_checkpoint_payload(
            epoch=epoch,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics_monitor=test_monitor_metrics,
        )

        torch.save(checkpoint_payload, str(last_ckpt_path))

        if args.checkpoint_every > 0 and epoch % args.checkpoint_every == 0:
            periodic_path = output_path.parent / f"{output_path.stem}_epoch_{epoch:03d}.pth"
            torch.save(checkpoint_payload, str(periodic_path))
            logger.info("  Periodic checkpoint saved: %s", periodic_path)

        improved = val_metrics["selection_score"] > (best_score + args.early_stopping_min_delta)
        if improved:
            best_score = val_metrics["selection_score"]
            best_epoch = epoch
            best_metrics = {
                "loss": float(val_metrics["loss"]),
                "mal_acc": float(val_metrics["mal_acc"]),
                "dis_acc": float(val_metrics["dis_acc"]),
                "mal_macro_f1": float(val_metrics["mal_macro_f1"]),
                "dis_macro_f1": float(val_metrics["dis_macro_f1"]),
                "selection_score": float(val_metrics["selection_score"]),
            }
            checkpoint_payload["best_val_selection_score_so_far"] = float(best_score)
            torch.save(checkpoint_payload, str(output_path))
            logger.info("  Best model updated: %s (Val Score: %.4f)", output_path, best_score)
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        if args.save_top_k > 0:
            qualifies_for_top_k = (
                len(top_k_checkpoints) < args.save_top_k
                or val_metrics["selection_score"] > top_k_checkpoints[-1]["val_selection_score"]
            )
            if qualifies_for_top_k:
                top_k_path = (
                    output_path.parent
                    / f"{output_path.stem}_top_e{epoch:03d}_valscore_{val_metrics['selection_score']:.4f}.pth"
                )
                torch.save(checkpoint_payload, str(top_k_path))
                top_k_checkpoints.append(
                    {
                        "epoch": epoch,
                        "val_selection_score": float(val_metrics["selection_score"]),
                        "path": str(top_k_path),
                    }
                )
                top_k_checkpoints.sort(key=lambda x: x["val_selection_score"], reverse=True)

                while len(top_k_checkpoints) > args.save_top_k:
                    removed = top_k_checkpoints.pop()
                    removed_path = Path(removed["path"])
                    if removed_path.exists():
                        removed_path.unlink()

                logger.info("  Top-K checkpoint saved: %s", top_k_path)

        _save_epoch_report(status="running")

        if (
            not args.disable_early_stopping
            and epoch >= args.min_epochs
            and epochs_without_improve >= args.early_stopping_patience
        ):
            stop_reason = (
                "early_stopping(no val_selection_score improvement > "
                f"{args.early_stopping_min_delta} for {epochs_without_improve} epochs)"
            )
            logger.info("Early stopping triggered at epoch %d. %s", epoch, stop_reason)
            break

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"], label="Val Loss")
    if any(v is not None for v in history["test_loss_monitor"]):
        test_loss_monitor = [np.nan if v is None else v for v in history["test_loss_monitor"]]
        axes[0].plot(test_loss_monitor, label="Test Loss (Monitor)", linestyle="--", alpha=0.7)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["val_mal_macro_f1"], label="Val Mal Macro F1")
    axes[1].plot(history["val_dis_macro_f1"], label="Val Dis Macro F1")
    axes[1].set_title("Validation Macro F1")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    axes[2].plot(history["val_selection_score"], label="Val Selection Score", color="green")
    if any(v is not None for v in history["test_selection_score_monitor"]):
        test_score_monitor = [np.nan if v is None else v for v in history["test_selection_score_monitor"]]
        axes[2].plot(test_score_monitor, label="Test Selection Score (Monitor)", linestyle="--", alpha=0.7)
    axes[2].set_title("Selection Metric")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(str(plot_path), dpi=150)
    logger.info("Training curves saved to %s", plot_path)

    final_epoch = history["epoch"][-1] if history["epoch"] else 0

    def _evaluate_checkpoint_on_test(checkpoint_path: Path):
        if not checkpoint_path.exists():
            return None

        checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
        eval_model = _create_model(
            num_malignancy_classes=len(checkpoint["malignancy_classes"]),
            num_disease_classes=len(checkpoint["disease_classes"]),
            pretrained=False,
        ).to(device)
        eval_model.load_state_dict(checkpoint["model_state_dict"])

        metrics = evaluate(
            model=eval_model,
            dataloader=test_loader,
            criterion_mal=eval_criterion_mal,
            criterion_dis=eval_criterion_dis,
            device=device,
            epoch_idx=final_epoch,
            total_epochs=args.epochs,
            stage_name="Test",
            mal_loss_weight=args.mal_loss_weight,
            dis_loss_weight=args.dis_loss_weight,
            use_tqdm=use_tqdm,
        )

        mal_report = classification_report(
            metrics["mal_labels"],
            metrics["mal_preds"],
            labels=list(range(len(malignancy_classes))),
            target_names=malignancy_classes,
            zero_division=0,
            output_dict=True,
        )
        dis_report = classification_report(
            metrics["dis_labels"],
            metrics["dis_preds"],
            labels=list(range(len(disease_classes))),
            target_names=disease_classes,
            zero_division=0,
            output_dict=True,
        )

        return {
            "checkpoint_path": str(checkpoint_path),
            "loss": float(metrics["loss"]),
            "mal_acc": float(metrics["mal_acc"]),
            "dis_acc": float(metrics["dis_acc"]),
            "mal_macro_f1": float(metrics["mal_macro_f1"]),
            "dis_macro_f1": float(metrics["dis_macro_f1"]),
            "selection_score": float(metrics["selection_score"]),
            "malignancy_report": mal_report,
            "disease_report": dis_report,
        }

    final_test_results = {
        "best_model": _evaluate_checkpoint_on_test(output_path),
        "last_model": _evaluate_checkpoint_on_test(last_ckpt_path),
    }

    _save_epoch_report(status="completed", final_test_results=final_test_results)
    logger.info("Training history saved to %s", history_csv_path)
    logger.info("Training report saved to %s", report_path)

    logger.info("\nTraining complete! Best val selection score: %.4f (epoch=%s)", best_score, best_epoch)
    logger.info("Best model path: %s", output_path)
    logger.info("Last checkpoint path: %s", last_ckpt_path)
    if final_test_results.get("best_model") is not None:
        logger.info(
            "Final test (best model): loss=%.4f, mal_acc=%.3f, dis_acc=%.3f, score=%.4f",
            final_test_results["best_model"]["loss"],
            final_test_results["best_model"]["mal_acc"],
            final_test_results["best_model"]["dis_acc"],
            final_test_results["best_model"]["selection_score"],
        )
    if top_k_checkpoints:
        logger.info("Top-K checkpoints by val selection score:")
        for rank, ckpt in enumerate(top_k_checkpoints, start=1):
            logger.info(
                "  %d. epoch=%d, val_score=%.4f, path=%s",
                rank,
                ckpt["epoch"],
                ckpt["val_selection_score"],
                ckpt["path"],
            )


if __name__ == "__main__":
    main()
