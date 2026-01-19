#!/usr/bin/env python3
"""
Improved Training Script - Better loss, more epochs, class weighting
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm

# Configuration
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 8  # Increased from 4
EPOCHS = 50  # Increased from 10
LEARNING_RATE = 5e-4  # Slightly higher
WEIGHT_DECAY = 1e-4
IN_CHANNELS = 11
NUM_CLASSES = 2

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


class SyntheticDataset(Dataset):
    """Generate synthetic satellite imagery for training."""
    
    def __init__(self, num_samples=500, image_size=256):  # More samples
        self.num_samples = num_samples
        self.image_size = image_size
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate synthetic 11-band Sentinel-2 image
        image = np.random.rand(11, self.image_size, self.image_size).astype(np.float32) * 0.2
        
        # Add more realistic structure
        for i in range(11):
            y, x = np.ogrid[:self.image_size, :self.image_size]
            center_y, center_x = self.image_size // 2, self.image_size // 2
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            image[i] += 0.1 * np.exp(-dist**2 / (2 * (self.image_size/4)**2))
            
            # Add some noise
            image[i] += np.random.randn(self.image_size, self.image_size).astype(np.float32) * 0.02
        
        # Generate synthetic debris mask (more realistic)
        mask = np.zeros((self.image_size, self.image_size), dtype=np.int64)
        
        # Add debris patches with more variety
        n_patches = np.random.randint(1, 8)  # More patches
        for _ in range(n_patches):
            cy = np.random.randint(30, self.image_size - 30)
            cx = np.random.randint(30, self.image_size - 30)
            size = np.random.randint(8, 40)  # Larger patches
            y, x = np.ogrid[:self.image_size, :self.image_size]
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            mask[dist < size] = 1
        
        # Ensure at least some debris in most images
        if idx % 3 != 0:  # 67% have debris
            if np.sum(mask) == 0:
                # Force at least one patch
                cy = np.random.randint(50, self.image_size - 50)
                cx = np.random.randint(50, self.image_size - 50)
                size = np.random.randint(10, 25)
                y, x = np.ogrid[:self.image_size, :self.image_size]
                dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                mask[dist < size] = 1
        
        return torch.from_numpy(image), torch.from_numpy(mask)


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class ImprovedUNet(nn.Module):
    """Improved U-Net with better architecture."""
    
    def __init__(self, in_channels=11, num_classes=2):
        super().__init__()
        
        # Encoder with more features
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(1024, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        self.out = nn.Conv2d(64, num_classes, 1)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout2d(0.1)
    
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        b = self.bottleneck(self.pool(e4))
        b = self.dropout(b)
        
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        
        return self.out(d1)


def calculate_class_weights(dataset, num_classes=2):
    """Calculate class weights for imbalanced data."""
    total_pixels = 0
    class_counts = np.zeros(num_classes)
    
    print("Calculating class weights...")
    for i in tqdm(range(min(100, len(dataset))), desc="Sampling"):
        _, mask = dataset[i]
        mask_np = mask.numpy()
        total_pixels += mask_np.size
        for c in range(num_classes):
            class_counts[c] += np.sum(mask_np == c)
    
    # Calculate weights (inverse frequency)
    class_weights = total_pixels / (num_classes * class_counts + 1e-8)
    class_weights = class_weights / class_weights.sum() * num_classes
    
    print(f"Class distribution: {class_counts}")
    print(f"Class weights: {class_weights}")
    
    return torch.FloatTensor(class_weights).to(DEVICE)


def train():
    print("=" * 60)
    print("🚀 IMPROVED Training - Marine Debris Detection")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print()
    
    # Create datasets with more samples
    print("Creating training data...")
    train_dataset = SyntheticDataset(num_samples=1000)  # More samples
    val_dataset = SyntheticDataset(num_samples=200)
    
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
    
    # Calculate class weights
    class_weights = calculate_class_weights(train_dataset)
    print()
    
    # Create model
    print("Initializing improved model...")
    model = ImprovedUNet(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
    model = model.to(DEVICE)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    print()
    
    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=LEARNING_RATE, 
        weight_decay=WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=LEARNING_RATE * 0.01
    )
    
    # Focal loss for class imbalance
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    
    # Also prepare weighted CE as backup
    weighted_ce = nn.CrossEntropyLoss(weight=class_weights)
    
    # Training loop
    print("Starting training...")
    print("=" * 60)
    
    best_loss = float('inf')
    best_f1 = 0.0
    patience = 10
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for images, masks in pbar:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            # Combine focal loss with weighted CE
            loss_focal = criterion(outputs, masks)
            loss_weighted = weighted_ce(outputs, masks)
            loss = 0.7 * loss_focal + 0.3 * loss_weighted
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)
                outputs = model(images)
                
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
                # Get predictions for F1
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds.flatten())
                all_labels.extend(masks.cpu().numpy().flatten())
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Calculate F1 score
        from sklearn.metrics import f1_score
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        if (all_labels == 1).sum() > 0:
            f1 = f1_score(all_labels, all_preds, pos_label=1, zero_division=0)
        else:
            f1 = 0.0
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{EPOCHS}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}, F1: {f1:.4f}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if f1 > best_f1 or (f1 == best_f1 and avg_val_loss < best_loss):
            if f1 > best_f1:
                best_f1 = f1
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
            patience_counter = 0
            
            save_path = OUTPUT_DIR / "improved_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'f1': f1,
                'train_loss': avg_train_loss,
            }, save_path)
            print(f"  ★ Best model saved! (F1={f1:.4f}, Val Loss={avg_val_loss:.4f})")
        else:
            patience_counter += 1
        
        scheduler.step()
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Training complete
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Best F1 Score: {best_f1:.4f}")
    print(f"Best validation loss: {best_loss:.4f}")
    print(f"Model saved to: {OUTPUT_DIR / 'improved_model.pth'}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
