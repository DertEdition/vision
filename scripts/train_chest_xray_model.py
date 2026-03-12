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
import os
import sys
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import cv2
from sklearn.metrics import roc_auc_score, classification_report

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
    ):
        self.image_dir = image_dir
        self.transform = transform
        self.disease_labels = disease_labels or DISEASE_LABELS
        
        # Load metadata
        self.df = pd.read_csv(metadata_csv)
        logger.info(f"Loaded {len(self.df)} records from {metadata_csv}")
        
        # Check for required columns
        if "Image Index" not in self.df.columns or "Finding Labels" not in self.df.columns:
            logger.error("CSV must contain 'Image Index' and 'Finding Labels' columns")
            # Try alternative column names
            if "image_id" in self.df.columns:
                self.df.rename(columns={"image_id": "Image Index"}, inplace=True)
            if "labels" in self.df.columns:
                self.df.rename(columns={"labels": "Finding Labels"}, inplace=True)
        
        # Filter to only rows with existing images
        if Path(image_dir).exists():
            existing_images = set(f.name for f in Path(image_dir).iterdir() if f.is_file())
            self.df = self.df[self.df["Image Index"].isin(existing_images)]
            logger.info(f"After image matching: {len(self.df)} records")
        
        # Sample for quick testing
        if sample_size and sample_size < len(self.df):
            self.df = self.df.sample(n=sample_size, random_state=42).reset_index(drop=True)
            logger.info(f"Sampled {sample_size} records")
        
        # Pre-compute multi-hot label vectors
        self.labels = self._compute_labels()
    
    def _compute_labels(self) -> np.ndarray:
        """Convert 'Finding Labels' strings to multi-hot vectors."""
        num_samples = len(self.df)
        num_classes = len(self.disease_labels)
        labels = np.zeros((num_samples, num_classes), dtype=np.float32)
        
        for i, row in self.df.iterrows():
            findings = str(row["Finding Labels"]).split("|")
            for finding in findings:
                finding = finding.strip()
                if finding in self.disease_labels:
                    idx = self.disease_labels.index(finding)
                    labels[self.df.index.get_loc(i), idx] = 1.0
        
        # Log label distribution
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
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
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


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if (batch_idx + 1) % 20 == 0:
            logger.info(f"  Batch {batch_idx+1}/{len(dataloader)} - Loss: {loss.item():.4f}")
    
    return running_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device, disease_labels):
    """Evaluate the model."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            preds = torch.sigmoid(outputs)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
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


def main():
    parser = argparse.ArgumentParser(description="Train Chest X-ray CNN Model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--image-size", type=int, default=224, help="Input image size")
    parser.add_argument("--sample-size", type=int, default=None, help="Use a subset for quick testing")
    parser.add_argument("--output", type=str, default="./models/chest_xray_model.pth", help="Output model path")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--data-dir", type=str, default="./data/chest_xray", help="Dataset directory")
    parser.add_argument("--metadata-csv", type=str, default=None, help="Path to metadata CSV")
    parser.add_argument("--image-dir", type=str, default=None, help="Path to image directory")
    args = parser.parse_args()
    
    # Resolve paths
    data_dir = Path(args.data_dir)
    
    # Try to find metadata CSV
    metadata_csv = args.metadata_csv
    if metadata_csv is None:
        candidates = [
            data_dir / "Data_Entry_2017.csv",
            data_dir / "Data_Entry_2017_v2020.csv",
            data_dir / "metadata.csv",
            data_dir / "chest_xray_metadata.csv",
        ]
        for candidate in candidates:
            if candidate.exists():
                metadata_csv = str(candidate)
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
        candidates = [
            data_dir / "images",
            data_dir / "images_001" / "images",
            data_dir / "sample",
        ]
        for candidate in candidates:
            if candidate.exists():
                image_dir = str(candidate)
                break
    
    if image_dir is None or not Path(image_dir).exists():
        logger.error(
            f"Image directory not found at {data_dir}. "
            "Please download the dataset and specify --image-dir."
        )
        sys.exit(1)
    
    logger.info(f"Metadata CSV: {metadata_csv}")
    logger.info(f"Image directory: {image_dir}")
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Create dataset
    full_dataset = ChestXrayDataset(
        metadata_csv=metadata_csv,
        image_dir=image_dir,
        disease_labels=DISEASE_LABELS,
        transform=train_transform,
        sample_size=args.sample_size,
    )
    
    # Split into train/test (80/20)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Override transform for test split
    # Note: Since we use random_split, the test set will use train transforms
    # For a production setup, use separate Dataset instances
    
    logger.info(f"Train: {len(train_dataset)} samples, Test: {len(test_dataset)} samples")
    
    if len(train_dataset) == 0:
        logger.error("No training samples found!")
        sys.exit(1)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    logger.info(f"Using device: {device}")
    
    # Model: ResNet50
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(DISEASE_LABELS))
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: ResNet50 - {total_params:,} total params, {trainable_params:,} trainable")
    
    # Loss & Optimizer
    criterion = nn.BCEWithLogitsLoss()  # Multi-label classification
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # Training loop
    best_auc = 0.0
    history = {"train_loss": [], "test_loss": [], "mean_auc": []}
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting training: {args.epochs} epochs, batch_size={args.batch_size}")
    logger.info(f"{'='*60}\n")
    
    for epoch in range(1, args.epochs + 1):
        logger.info(f"Epoch {epoch}/{args.epochs}")
        
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Evaluate
        test_loss, mean_auc, aucs, _, _ = evaluate(
            model, test_loader, criterion, device, DISEASE_LABELS
        )
        
        scheduler.step()
        
        logger.info(
            f"Epoch {epoch}: Train Loss={train_loss:.4f} | "
            f"Test Loss={test_loss:.4f}, Mean AUC={mean_auc:.4f}"
        )
        
        # Log per-disease AUC
        for disease, auc in aucs.items():
            if auc > 0:
                logger.info(f"  {disease}: AUC={auc:.4f}")
        
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["mean_auc"].append(mean_auc)
        
        # Save best model
        if mean_auc > best_auc:
            best_auc = mean_auc
            
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_labels": DISEASE_LABELS,
                "epoch": epoch,
                "best_mean_auc": best_auc,
                "per_disease_auc": aucs,
                "args": vars(args),
            }, str(output_path))
            
            logger.info(f"  ✓ Best model saved to {output_path} (Mean AUC: {best_auc:.4f})")
    
    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["test_loss"], label="Test Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Test Loss")
    axes[0].legend()
    
    axes[1].plot(history["mean_auc"], label="Mean AUC", color="green")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUC")
    axes[1].set_title("Mean AUC Score")
    axes[1].legend()
    
    plt.tight_layout()
    plot_path = Path(args.output).parent / "chest_xray_training_curves.png"
    plt.savefig(str(plot_path), dpi=150)
    logger.info(f"Training curves saved to {plot_path}")
    
    logger.info(f"\nTraining complete! Best Mean AUC: {best_auc:.4f}")
    logger.info(f"Model saved to: {args.output}")


if __name__ == "__main__":
    main()
