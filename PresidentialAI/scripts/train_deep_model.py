#!/usr/bin/env python3
"""
Deep Learning Model Training for Marine Debris Detection
=========================================================

This script trains a U-Net model for semantic segmentation of marine debris
in Sentinel-2 satellite imagery.

Requirements:
    pip install torch torchvision rasterio albumentations scikit-learn tqdm

Usage:
    python train_deep_model.py

Hardware:
    - GPU with 8GB+ VRAM recommended (RTX 3060 or better)
    - 16GB+ RAM
    - Training time: ~2-4 hours depending on GPU

Output:
    - Best model saved to: ../outputs/models/best_debris_unet.pth
    - Training logs printed to console
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

# Try to import optional libraries
try:
    import rasterio
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
    print("Warning: rasterio not installed. Run: pip install rasterio")

try:
    import albumentations as A
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    print("Warning: albumentations not installed. Run: pip install albumentations")

# =============================================================================
# CONFIGURATION
# =============================================================================
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data" / "marida" / "patches"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "models"

# Training hyperparameters
BATCH_SIZE = 8
EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
OVERSAMPLE_DEBRIS = 20  # Repeat debris patches this many times
NUM_WORKERS = 4

# Model configuration
IN_CHANNELS = 15  # 11 bands + 4 spectral indices
NUM_CLASSES = 2   # Background, Debris

# Determine device
if torch.cuda.is_available():
    DEVICE = "cuda"
    print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = "mps"
    print("Using Apple Silicon MPS")
else:
    DEVICE = "cpu"
    print("Using CPU (training will be slow)")

# =============================================================================
# SPECTRAL INDICES
# =============================================================================
def add_spectral_indices(image):
    """
    Add debris-relevant spectral indices as extra channels.
    
    Input: (11, H, W) array
    Output: (15, H, W) array with added FDI, NDWI, NDVI, PI
    """
    eps = 1e-8
    
    # Sentinel-2 band indices (0-indexed)
    blue = image[1]   # B2
    green = image[2]  # B3
    red = image[3]    # B4
    nir = image[7]    # B8
    swir1 = image[9]  # B11
    
    # Floating Debris Index - key for debris detection
    fdi = nir - (red + (swir1 - red) * 0.178)
    
    # Normalized Difference Water Index
    ndwi = (green - nir) / (green + nir + eps)
    
    # Normalized Difference Vegetation Index
    ndvi = (nir - red) / (nir + red + eps)
    
    # Plastic Index (experimental)
    pi = nir / (nir + red + eps)
    
    # Stack with original bands
    return np.concatenate([
        image, 
        fdi[None], 
        ndwi[None], 
        ndvi[None], 
        pi[None]
    ], axis=0)


def normalize_bands(image):
    """Normalize each band using percentile clipping."""
    normalized = np.zeros_like(image)
    for i in range(image.shape[0]):
        band = image[i]
        p2, p98 = np.percentile(band, (2, 98))
        normalized[i] = np.clip((band - p2) / (p98 - p2 + 1e-8), 0, 1)
    return normalized


# =============================================================================
# DATASET
# =============================================================================
class MaridaDataset(Dataset):
    """
    Dataset for MARIDA marine debris patches.
    
    Handles loading, normalization, spectral index computation, and oversampling.
    """
    
    def __init__(self, image_paths, transform=None, oversample_debris=10):
        self.samples = []
        self.transform = transform
        
        print(f"Loading dataset from {len(image_paths)} images...")
        
        debris_count = 0
        non_debris_count = 0
        
        for img_path in tqdm(image_paths, desc="Scanning dataset"):
            label_path = img_path.parent / f"{img_path.stem}_cl.tif"
            
            if not label_path.exists():
                continue
            
            # Check if patch has debris
            with rasterio.open(label_path) as src:
                labels = src.read(1)
            has_debris = np.any(labels == 1)
            
            # Add to samples (oversample debris patches)
            if has_debris:
                n_copies = oversample_debris
                debris_count += 1
            else:
                n_copies = 1
                non_debris_count += 1
            
            for _ in range(n_copies):
                self.samples.append((img_path, label_path))
        
        print(f"Dataset created:")
        print(f"  - Debris patches: {debris_count} (oversampled {oversample_debris}x)")
        print(f"  - Non-debris patches: {non_debris_count}")
        print(f"  - Total samples: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label_path = self.samples[idx]
        
        # Load image (11 bands)
        with rasterio.open(img_path) as src:
            image = src.read().astype(np.float32)
        
        # Load labels
        with rasterio.open(label_path) as src:
            labels = src.read(1)
        
        # Binary mask: 1 = debris, 0 = everything else
        mask = (labels == 1).astype(np.int64)
        
        # Normalize bands
        image = normalize_bands(image)
        
        # Add spectral indices
        image = add_spectral_indices(image)
        
        # Data augmentation
        if self.transform:
            # Albumentations expects HWC format
            image_hwc = image.transpose(1, 2, 0)
            
            augmented = self.transform(image=image_hwc, mask=mask)
            image = augmented['image'].transpose(2, 0, 1)
            mask = augmented['mask']
        
        return torch.from_numpy(image.copy()), torch.from_numpy(mask.copy())


# =============================================================================
# MODEL: U-Net
# =============================================================================
class ConvBlock(nn.Module):
    """Double convolution block with BatchNorm and ReLU."""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.conv(x)


class DebrisUNet(nn.Module):
    """
    U-Net architecture for marine debris segmentation.
    
    Adapted for 15-channel input (11 Sentinel-2 bands + 4 spectral indices).
    """
    
    def __init__(self, in_channels=15, num_classes=2):
        super().__init__()
        
        # Encoder (downsampling path)
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)
        
        # Bottleneck
        self.bottleneck = ConvBlock(512, 1024)
        
        # Decoder (upsampling path)
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(1024, 512)
        
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(128, 64)
        
        # Output layer
        self.out = nn.Conv2d(64, num_classes, kernel_size=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        
        return self.out(d1)


# =============================================================================
# LOSS FUNCTION: Focal Loss
# =============================================================================
class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    Reduces the loss contribution from easy examples (well-classified),
    focusing training on hard, misclassified examples.
    
    Reference: Lin et al., "Focal Loss for Dense Object Detection", 2017
    """
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


# =============================================================================
# TRAINING LOOP
# =============================================================================
def train():
    """Main training function."""
    
    if not RASTERIO_AVAILABLE:
        print("ERROR: rasterio is required. Install with: pip install rasterio")
        return
    
    if not DATA_DIR.exists():
        print(f"ERROR: Data directory not found: {DATA_DIR}")
        print("Please download the MARIDA dataset first.")
        return
    
    print("=" * 60)
    print("Marine Debris Detection - Deep Learning Training")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    
    # Find all image files (exclude label and confidence files)
    image_paths = []
    for folder in DATA_DIR.iterdir():
        if folder.is_dir():
            for tif in folder.glob("*.tif"):
                if "_cl" not in tif.name and "_conf" not in tif.name:
                    image_paths.append(tif)
    
    if len(image_paths) == 0:
        print("ERROR: No image files found in dataset directory")
        return
    
    print(f"Found {len(image_paths)} satellite images")
    
    # Split into train/validation sets
    train_paths, val_paths = train_test_split(
        image_paths, 
        test_size=0.2, 
        random_state=42
    )
    print(f"Train: {len(train_paths)}, Validation: {len(val_paths)}")
    
    # Data augmentation (training only)
    if ALBUMENTATIONS_AVAILABLE:
        train_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
        ])
    else:
        train_transform = None
        print("Warning: No augmentation (albumentations not installed)")
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = MaridaDataset(
        train_paths, 
        transform=train_transform, 
        oversample_debris=OVERSAMPLE_DEBRIS
    )
    val_dataset = MaridaDataset(
        val_paths, 
        transform=None, 
        oversample_debris=1
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=NUM_WORKERS, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=NUM_WORKERS, 
        pin_memory=True
    )
    
    # Create model
    print("\nInitializing model...")
    model = DebrisUNet(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
    model = model.to(DEVICE)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=LEARNING_RATE, 
        weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)
    
    # Loss function
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    best_f1 = 0
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")
    
    for epoch in range(EPOCHS):
        # ===================== Training =====================
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for images, masks in pbar:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # ===================== Validation =====================
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                outputs = model(images.to(DEVICE))
                preds = outputs.argmax(dim=1).cpu().numpy()
                
                all_preds.extend(preds.flatten())
                all_labels.extend(masks.numpy().flatten())
        
        # Calculate metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Only calculate metrics if there are positive samples
        if (all_labels == 1).sum() > 0:
            f1 = f1_score(all_labels, all_preds, pos_label=1, zero_division=0)
            precision = precision_score(all_labels, all_preds, pos_label=1, zero_division=0)
            recall = recall_score(all_labels, all_preds, pos_label=1, zero_division=0)
        else:
            f1, precision, recall = 0, 0, 0
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{EPOCHS}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        
        # Save best model
        if f1 > best_f1:
            best_f1 = f1
            save_path = OUTPUT_DIR / "best_debris_unet.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'f1': f1,
                'precision': precision,
                'recall': recall,
            }, save_path)
            print(f"  ★ New best model saved! (F1={f1:.4f})")
        
        scheduler.step()
    
    # Training complete
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best F1 Score: {best_f1:.4f}")
    print(f"Model saved to: {OUTPUT_DIR / 'best_debris_unet.pth'}")
    print()
    print("To use this model in the API, update backend/api.py to load")
    print("the trained weights instead of using anomaly detection.")


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    train()

