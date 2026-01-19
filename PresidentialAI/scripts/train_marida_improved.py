#!/usr/bin/env python3
"""
IMPROVED Training Script - Better hyperparameters, loss functions, and strategies
"""

import os
import sys
import random
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import rasterio
from tqdm import tqdm

# Add parent to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_improved import ImprovedUNet, FocalLoss

# Seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# IMPROVED Configuration
DATA_DIR = PROJECT_ROOT / "data" / "marida"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 8
EPOCHS = 50  # More epochs but reasonable
LEARNING_RATE = 5e-4  # Lower, more stable LR
WEIGHT_DECAY = 1e-4
IN_CHANNELS = 11
NUM_CLASSES = 2
PATIENCE = 10  # Early stopping
GRAD_CLIP = 1.0  # Gradient clipping

# Device
if torch.cuda.is_available():
    DEVICE = "cuda"
    print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = "mps"
    print("Using Apple Silicon MPS")
else:
    DEVICE = "cpu"
    print("Using CPU")

# MARIDA normalization stats
MARIDA_MEAN = np.array([0.05, 0.06, 0.06, 0.05, 0.08, 0.10, 0.11, 0.10, 0.12, 0.13, 0.09], dtype=np.float32)
MARIDA_STD = np.array([0.03, 0.03, 0.03, 0.03, 0.04, 0.05, 0.05, 0.05, 0.05, 0.06, 0.05], dtype=np.float32)

# MARIDA debris classes
DEBRIS_CLASSES = {1, 2, 3, 4}


class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        # Apply softmax to get probabilities
        probs = F.softmax(inputs, dim=1)
        # Get debris class probabilities
        debris_probs = probs[:, 1]
        
        # Flatten
        debris_probs = debris_probs.contiguous().view(-1)
        targets = targets.float().contiguous().view(-1)
        
        # Calculate dice
        intersection = (debris_probs * targets).sum()
        dice = (2. * intersection + self.smooth) / (debris_probs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice


class CombinedLoss(nn.Module):
    """Combine Focal Loss and Dice Loss for better performance"""
    def __init__(self, focal_weight=0.8, dice_weight=0.2):
        super().__init__()
        self.focal_loss = FocalLoss(alpha=0.75, gamma=2.0)  # More stable gamma
        self.dice_loss = DiceLoss()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
    
    def forward(self, inputs, targets):
        focal = self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        # Check for NaN and handle
        if torch.isnan(focal):
            focal = torch.tensor(0.0, device=inputs.device)
        if torch.isnan(dice):
            dice = torch.tensor(0.0, device=inputs.device)
        combined = self.focal_weight * focal + self.dice_weight * dice
        # Clamp to prevent NaN
        return torch.clamp(combined, min=0.0, max=100.0)


class MaridaDataset(Dataset):
    """Improved MARIDA dataset with better augmentation"""
    
    def __init__(self, data_dir, max_samples=1500, augment=False):
        self.data_dir = Path(data_dir)
        self.augment = augment
        self.samples = []
        
        # Find all image/label pairs
        patches_dir = self.data_dir / "patches"
        if not patches_dir.exists():
            raise FileNotFoundError(f"Patches directory not found: {patches_dir}")
        
        for folder in os.listdir(patches_dir):
            folder_path = patches_dir / folder
            if folder_path.is_dir():
                for file in os.listdir(folder_path):
                    if file.endswith('.tif') and '_cl' not in file and '_conf' not in file:
                        img_path = folder_path / file
                        label_path = folder_path / file.replace('.tif', '_cl.tif')
                        if label_path.exists():
                            self.samples.append((img_path, label_path))
        
        # Use more samples
        random.shuffle(self.samples)
        self.samples = self.samples[:max_samples]
        print(f"Loaded {len(self.samples)} MARIDA samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label_path = self.samples[idx]
        
        # Load image
        with rasterio.open(img_path) as src:
            image = src.read().astype(np.float32)
        
        # Load label
        with rasterio.open(label_path) as src:
            label = src.read(1)
        
        # Normalize each band using MARIDA statistics
        for i in range(min(len(MARIDA_MEAN), image.shape[0])):
            image[i] = (image[i] - MARIDA_MEAN[i]) / (MARIDA_STD[i] + 1e-8)
        
        # Clamp values to prevent extreme outliers
        image = np.clip(image, -10, 10)
        
        # Binary classification: debris-related vs not debris
        binary_label = np.isin(label.astype(np.int32), list(DEBRIS_CLASSES)).astype(np.int64)
        
        # Enhanced augmentation
        if self.augment:
            # Horizontal flip
            if random.random() > 0.5:
                image = np.flip(image, axis=2).copy()
                binary_label = np.flip(binary_label, axis=0).copy()
            
            # Vertical flip
            if random.random() > 0.5:
                image = np.flip(image, axis=1).copy()
                binary_label = np.flip(binary_label, axis=1).copy()
            
            # 90-degree rotation
            if random.random() > 0.5:
                k = random.randint(1, 3)
                image = np.rot90(image, k=k, axes=(1, 2)).copy()
                binary_label = np.rot90(binary_label, k=k).copy()
        
        return torch.from_numpy(image.copy()), torch.from_numpy(binary_label.copy())


def calculate_iou(pred, target):
    """Calculate Intersection over Union"""
    pred_binary = (pred > 0.5).float()
    target_binary = target.float()
    
    intersection = (pred_binary * target_binary).sum().item()
    union = pred_binary.sum().item() + target_binary.sum().item() - intersection
    
    if union == 0:
        return 1.0  # Perfect match if both are empty
    
    return intersection / (union + 1e-8)


def calculate_metrics(pred, target):
    """Calculate F1, precision, recall, IoU"""
    pred_binary = (pred > 0.5).float()
    target_binary = target.float()
    
    tp = (pred_binary * target_binary).sum().item()
    fp = (pred_binary * (1 - target_binary)).sum().item()
    fn = ((1 - pred_binary) * target_binary).sum().item()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = calculate_iou(pred, target)
    
    return f1, precision, recall, iou


def train():
    print("=" * 60)
    print("🚀 IMPROVED Training - Better Hyperparameters & Loss Functions")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Early stopping patience: {PATIENCE}")
    print()
    
    # Create datasets with MORE data
    print("Loading MARIDA dataset...")
    train_dataset = MaridaDataset(DATA_DIR, max_samples=1500, augment=True)  # More samples
    val_dataset = MaridaDataset(DATA_DIR, max_samples=300, augment=False)  # More validation
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print()
    
    # Create model
    print("Initializing ImprovedUNet...")
    model = ImprovedUNet(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
    model = model.to(DEVICE)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    print()
    
    # Optimizer with higher LR
    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    # Learning rate scheduler - reduce on plateau
    scheduler = ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # IMPROVED: Use Focal Loss with better parameters for class imbalance
    # Higher alpha for debris class (which is rare)
    criterion = FocalLoss(alpha=0.75, gamma=2.0)
    
    # Training loop
    print("Starting training...")
    print("=" * 60)
    
    best_f1 = 0.0
    best_iou = 0.0
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss = 0.0
        train_f1 = 0.0
        train_iou = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for images, masks in pbar:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Check for NaN/Inf and skip backward if needed
            if not (torch.isnan(loss) or torch.isinf(loss)):
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                
                optimizer.step()
            else:
                print(f"Warning: NaN/Inf loss detected, skipping batch")
                continue
            
            # Calculate metrics
            probs = F.softmax(outputs, dim=1)
            debris_probs = probs[:, 1]
            f1, _, _, iou = calculate_metrics(debris_probs, masks)
            
            train_loss += loss.item()
            train_f1 += f1
            train_iou += iou
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'f1': f'{f1:.4f}',
                'iou': f'{iou:.4f}'
            })
        
        train_loss /= len(train_loader)
        train_f1 /= len(train_loader)
        train_iou /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_f1 = 0.0
        val_precision = 0.0
        val_recall = 0.0
        val_iou = 0.0
        
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                probs = F.softmax(outputs, dim=1)
                debris_probs = probs[:, 1]
                f1, precision, recall, iou = calculate_metrics(debris_probs, masks)
                
                val_loss += loss.item()
                val_f1 += f1
                val_precision += precision
                val_recall += recall
                val_iou += iou
        
        val_loss /= len(val_loader)
        val_f1 /= len(val_loader)
        val_precision /= len(val_loader)
        val_recall /= len(val_loader)
        val_iou /= len(val_loader)
        
        # Update learning rate
        scheduler.step(val_f1)
        
        print(f"\nEpoch {epoch+1}/{EPOCHS}:")
        print(f"  Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}, Train IoU: {train_iou:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}, Val IoU: {val_iou:.4f}")
        print(f"  Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model (based on F1)
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_iou = val_iou
            best_epoch = epoch + 1
            patience_counter = 0
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'f1': val_f1,
                'iou': val_iou,
                'precision': val_precision,
                'recall': val_recall,
                'loss': val_loss,
            }
            model_path = OUTPUT_DIR / "improved_model.pth"
            torch.save(checkpoint, model_path)
            print(f"  ✅ Saved best model (F1: {val_f1:.4f}, IoU: {val_iou:.4f}) to {model_path}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {PATIENCE} epochs)")
                break
        
        print()
    
    print("=" * 60)
    print(f"Training complete! Best F1: {best_f1:.4f}, Best IoU: {best_iou:.4f} at epoch {best_epoch}")
    print(f"Model saved to: {OUTPUT_DIR / 'improved_model.pth'}")
    print("=" * 60)


if __name__ == "__main__":
    train()
