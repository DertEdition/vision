"""
Dermatology Model Training Script

Train a CNN model on the DERM12345 dataset for skin lesion classification.
Dual-head architecture:
  1. Malignancy classification: benign / malignant / indeterminate
  2. Disease type classification: specific sub-class

Usage:
    python scripts/train_dermatology_model.py --epochs 10 --batch-size 32
    python scripts/train_dermatology_model.py --epochs 5 --sample-size 500  # Quick test
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
from sklearn.metrics import classification_report, confusion_matrix

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# Dataset
# ============================================================


class DERM12345Dataset(Dataset):
    """
    PyTorch Dataset for DERM12345 skin lesion images.
    
    Loads images and labels from the DERM12345 metadata CSV.
    """
    
    def __init__(
        self,
        metadata_csv: str,
        image_dirs: list,
        malignancy_classes: list,
        disease_classes: list,
        transform=None,
        sample_size: int = None,
    ):
        self.transform = transform
        self.malignancy_classes = malignancy_classes
        self.disease_classes = disease_classes
        
        # Load metadata
        self.df = pd.read_csv(metadata_csv)
        logger.info(f"Loaded {len(self.df)} records from {metadata_csv}")
        
        # Filter to only rows with known malignancy and sub_class
        self.df = self.df[self.df["malignancy"].isin(malignancy_classes)]
        self.df = self.df[self.df["sub_class"].isin(disease_classes)]
        logger.info(f"After filtering: {len(self.df)} records")
        
        # Build image path lookup (search recursively in subdirectories)
        self.image_paths = {}
        for img_dir in image_dirs:
            if Path(img_dir).exists():
                # Search recursively for images in subdirectories
                for f in Path(img_dir).rglob("*"):
                    if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                        self.image_paths[f.stem] = str(f)
        
        # Filter to only rows with existing images
        self.df = self.df[self.df["image_id"].isin(self.image_paths.keys())]
        logger.info(f"After image matching: {len(self.df)} records")
        
        # Optional sampling for quick tests
        if sample_size and sample_size < len(self.df):
            self.df = self.df.sample(n=sample_size, random_state=42).reset_index(drop=True)
            logger.info(f"Sampled {sample_size} records for training")
        
        # Create label encodings
        self.malignancy_to_idx = {c: i for i, c in enumerate(malignancy_classes)}
        self.disease_to_idx = {c: i for i, c in enumerate(disease_classes)}
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row["image_id"]
        
        # Load image with OpenCV (handle unicode paths correctly)
        img_path = self.image_paths[image_id]
        with open(img_path, "rb") as f:
            file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            # torchvision transforms expect PIL or numpy array
            # We pass numpy array (H, W, C) in uint8 format
            from PIL import Image as PILImage
            image = PILImage.fromarray(image)
            image = self.transform(image)
        
        # Labels
        mal_label = self.malignancy_to_idx[row["malignancy"]]
        dis_label = self.disease_to_idx[row["sub_class"]]
        
        return image, mal_label, dis_label


# ============================================================
# Model Architecture
# ============================================================


class DermatologyNet(nn.Module):
    """
    ResNet18-based model with dual classification heads.
    
    Head 1: Malignancy (benign/malignant/indeterminate)
    Head 2: Disease type (sub_class from DERM12345)
    """
    
    def __init__(self, num_malignancy_classes: int, num_disease_classes: int, pretrained: bool = True):
        super().__init__()
        
        # Load pretrained ResNet18
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # Remove original classification head
        
        # Custom dual heads
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


# ============================================================
# Training Logic
# ============================================================


def train_one_epoch(model, dataloader, criterion_mal, criterion_dis, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct_mal = 0
    correct_dis = 0
    total = 0
    
    for batch_idx, (images, mal_labels, dis_labels) in enumerate(dataloader):
        images = images.to(device)
        mal_labels = mal_labels.to(device)
        dis_labels = dis_labels.to(device)
        
        optimizer.zero_grad()
        
        mal_out, dis_out = model(images)
        
        loss_mal = criterion_mal(mal_out, mal_labels)
        loss_dis = criterion_dis(dis_out, dis_labels)
        loss = loss_mal + loss_dis
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, mal_pred = mal_out.max(1)
        _, dis_pred = dis_out.max(1)
        correct_mal += mal_pred.eq(mal_labels).sum().item()
        correct_dis += dis_pred.eq(dis_labels).sum().item()
        total += images.size(0)
        
        if (batch_idx + 1) % 10 == 0:
            logger.info(
                f"  Batch {batch_idx+1}/{len(dataloader)} - "
                f"Loss: {loss.item():.4f} - "
                f"Mal Acc: {correct_mal/total:.3f} - "
                f"Dis Acc: {correct_dis/total:.3f}"
            )
    
    return running_loss / len(dataloader), correct_mal / total, correct_dis / total


def evaluate(model, dataloader, criterion_mal, criterion_dis, device):
    """Evaluate the model."""
    model.eval()
    running_loss = 0.0
    all_mal_preds = []
    all_mal_labels = []
    all_dis_preds = []
    all_dis_labels = []
    
    with torch.no_grad():
        for images, mal_labels, dis_labels in dataloader:
            images = images.to(device)
            mal_labels = mal_labels.to(device)
            dis_labels = dis_labels.to(device)
            
            mal_out, dis_out = model(images)
            
            loss_mal = criterion_mal(mal_out, mal_labels)
            loss_dis = criterion_dis(dis_out, dis_labels)
            loss = loss_mal + loss_dis
            running_loss += loss.item()
            
            _, mal_pred = mal_out.max(1)
            _, dis_pred = dis_out.max(1)
            
            all_mal_preds.extend(mal_pred.cpu().numpy())
            all_mal_labels.extend(mal_labels.cpu().numpy())
            all_dis_preds.extend(dis_pred.cpu().numpy())
            all_dis_labels.extend(dis_labels.cpu().numpy())
    
    avg_loss = running_loss / len(dataloader)
    mal_acc = np.mean(np.array(all_mal_preds) == np.array(all_mal_labels))
    dis_acc = np.mean(np.array(all_dis_preds) == np.array(all_dis_labels))
    
    return avg_loss, mal_acc, dis_acc, all_mal_preds, all_mal_labels, all_dis_preds, all_dis_labels


def main():
    parser = argparse.ArgumentParser(description="Train Dermatology CNN Model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--image-size", type=int, default=224, help="Input image size")
    parser.add_argument("--sample-size", type=int, default=None, help="Use a subset of data for quick testing")
    parser.add_argument("--output", type=str, default="./models/dermatology_model.pth", help="Output model path")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    args = parser.parse_args()
    
    # Paths
    data_dir = PROJECT_ROOT / "data" / "DERM12345"
    train_csv = data_dir / "derm12345_metadata_train.csv"
    test_csv = data_dir / "derm12345_metadata_test.csv"
    
    # Image directories
    train_image_dirs = [
        str(data_dir / "derm12345_train_part_1"),
        str(data_dir / "derm12345_train_part_2"),
    ]
    test_image_dirs = [
        str(data_dir / "derm12345_test"),
    ]
    
    # Discover classes from training data
    train_df = pd.read_csv(train_csv)
    malignancy_classes = sorted(train_df["malignancy"].dropna().unique().tolist())
    disease_classes = sorted(train_df["sub_class"].dropna().unique().tolist())
    
    logger.info(f"Malignancy classes ({len(malignancy_classes)}): {malignancy_classes}")
    logger.info(f"Disease classes ({len(disease_classes)}): {disease_classes}")
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Datasets
    train_dataset = DERM12345Dataset(
        metadata_csv=str(train_csv),
        image_dirs=train_image_dirs,
        malignancy_classes=malignancy_classes,
        disease_classes=disease_classes,
        transform=train_transform,
        sample_size=args.sample_size,
    )
    
    test_dataset = DERM12345Dataset(
        metadata_csv=str(test_csv),
        image_dirs=test_image_dirs,
        malignancy_classes=malignancy_classes,
        disease_classes=disease_classes,
        transform=test_transform,
        sample_size=args.sample_size // 5 if args.sample_size else None,
    )
    
    logger.info(f"Train: {len(train_dataset)} samples, Test: {len(test_dataset)} samples")
    
    if len(train_dataset) == 0:
        logger.error("No training samples found! Check data paths and image directories.")
        sys.exit(1)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    logger.info(f"Using device: {device}")
    
    # Model
    model = DermatologyNet(
        num_malignancy_classes=len(malignancy_classes),
        num_disease_classes=len(disease_classes),
        pretrained=True,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {total_params:,} total params, {trainable_params:,} trainable")
    
    # Loss & Optimizer
    criterion_mal = nn.CrossEntropyLoss()
    criterion_dis = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # Training loop
    best_acc = 0.0
    history = {"train_loss": [], "test_loss": [], "train_mal_acc": [], "test_mal_acc": []}
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting training: {args.epochs} epochs, batch_size={args.batch_size}")
    logger.info(f"{'='*60}\n")
    
    for epoch in range(1, args.epochs + 1):
        logger.info(f"Epoch {epoch}/{args.epochs}")
        
        # Train
        train_loss, train_mal_acc, train_dis_acc = train_one_epoch(
            model, train_loader, criterion_mal, criterion_dis, optimizer, device
        )
        
        # Evaluate
        test_loss, test_mal_acc, test_dis_acc, _, _, _, _ = evaluate(
            model, test_loader, criterion_mal, criterion_dis, device
        )
        
        scheduler.step()
        
        logger.info(
            f"Epoch {epoch}: "
            f"Train Loss={train_loss:.4f}, Mal Acc={train_mal_acc:.3f}, Dis Acc={train_dis_acc:.3f} | "
            f"Test Loss={test_loss:.4f}, Mal Acc={test_mal_acc:.3f}, Dis Acc={test_dis_acc:.3f}"
        )
        
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["train_mal_acc"].append(train_mal_acc)
        history["test_mal_acc"].append(test_mal_acc)
        
        # Save best model
        if test_mal_acc > best_acc:
            best_acc = test_mal_acc
            
            # Ensure output directory exists
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                "model_state_dict": model.state_dict(),
                "malignancy_classes": malignancy_classes,
                "disease_classes": disease_classes,
                "epoch": epoch,
                "best_mal_acc": best_acc,
                "args": vars(args),
            }, str(output_path))
            
            logger.info(f"  ✓ Best model saved to {output_path} (Mal Acc: {best_acc:.3f})")
    
    # Final evaluation
    logger.info(f"\n{'='*60}")
    logger.info("Final Evaluation on Test Set")
    logger.info(f"{'='*60}\n")
    
    _, _, _, mal_preds, mal_labels, dis_preds, dis_labels = evaluate(
        model, test_loader, criterion_mal, criterion_dis, device
    )
    
    logger.info("Malignancy Classification Report:")
    unique_mal_labels = sorted(list(set(mal_labels)))
    logger.info("\n" + classification_report(
        mal_labels, mal_preds,
        labels=unique_mal_labels,
        target_names=[malignancy_classes[i] for i in unique_mal_labels],
        zero_division=0
    ))
    
    logger.info("Disease Type Classification Report:")
    unique_dis_labels = sorted(list(set(dis_labels)))
    logger.info("\n" + classification_report(
        dis_labels, dis_preds,
        labels=unique_dis_labels,
        target_names=[disease_classes[i] for i in unique_dis_labels],
        zero_division=0
    ))
    
    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["test_loss"], label="Test Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Test Loss")
    axes[0].legend()
    axes[1].plot(history["train_mal_acc"], label="Train Malignancy Acc")
    axes[1].plot(history["test_mal_acc"], label="Test Malignancy Acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Malignancy Classification Accuracy")
    axes[1].legend()
    
    plt.tight_layout()
    plot_path = Path(args.output).parent / "dermatology_training_curves.png"
    plt.savefig(str(plot_path), dpi=150)
    logger.info(f"Training curves saved to {plot_path}")
    
    logger.info(f"\nTraining complete! Best malignancy accuracy: {best_acc:.3f}")
    logger.info(f"Model saved to: {args.output}")


if __name__ == "__main__":
    main()
