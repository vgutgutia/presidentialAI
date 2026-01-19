#!/usr/bin/env python3
"""
Quick Training Script - Works with or without MARIDA dataset
Creates synthetic data if real dataset is not available
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
DATA_DIR = PROJECT_ROOT / "data" / "marida" / "patches"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 4
EPOCHS = 10  # Quick training
LEARNING_RATE = 1e-4
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
    
    def __init__(self, num_samples=100, image_size=256):
        self.num_samples = num_samples
        self.image_size = image_size
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate synthetic 11-band Sentinel-2 image
        image = np.random.rand(11, self.image_size, self.image_size).astype(np.float32) * 0.2
        
        # Add some structure (water-like regions)
        for i in range(11):
            # Create some spatial structure
            y, x = np.ogrid[:self.image_size, :self.image_size]
            center_y, center_x = self.image_size // 2, self.image_size // 2
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            image[i] += 0.1 * np.exp(-dist**2 / (2 * (self.image_size/4)**2))
        
        # Generate synthetic debris mask (small random patches)
        mask = np.zeros((self.image_size, self.image_size), dtype=np.int64)
        
        # Add some debris patches
        n_patches = np.random.randint(0, 5)
        for _ in range(n_patches):
            cy = np.random.randint(50, self.image_size - 50)
            cx = np.random.randint(50, self.image_size - 50)
            size = np.random.randint(10, 30)
            y, x = np.ogrid[:self.image_size, :self.image_size]
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            mask[dist < size] = 1
        
        return torch.from_numpy(image), torch.from_numpy(mask)


class SimpleUNet(nn.Module):
    """Simple U-Net for quick training."""
    
    def __init__(self, in_channels=11, num_classes=2):
        super().__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
        )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
        )
        
        # Decoder
        self.up3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
        )
        
        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
        )
        
        self.up1 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
        )
        
        self.out = nn.Conv2d(32, num_classes, 1)
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        
        b = self.bottleneck(self.pool(e3))
        
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        
        return self.out(d1)


def train():
    print("=" * 60)
    print("Quick Training - Marine Debris Detection")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print()
    
    # Create synthetic dataset
    print("Creating synthetic training data...")
    train_dataset = SyntheticDataset(num_samples=200)
    val_dataset = SyntheticDataset(num_samples=50)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing issues
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
    print("Initializing model...")
    model = SimpleUNet(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
    model = model.to(DEVICE)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    print()
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    print("Starting training...")
    print("=" * 60)
    
    best_loss = float('inf')
    
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
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{EPOCHS}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            save_path = OUTPUT_DIR / "quick_trained_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
            }, save_path)
            print(f"  ★ Best model saved! (Val Loss: {avg_val_loss:.4f})")
        print()
    
    print("=" * 60)
    print("Training Complete!")
    print(f"Best validation loss: {best_loss:.4f}")
    print(f"Model saved to: {OUTPUT_DIR / 'quick_trained_model.pth'}")
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
