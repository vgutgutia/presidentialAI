"""
SUPER FAST Marine Debris Detection Model
Trains in 5-10 minutes on CPU
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
from torch.optim import Adam
import rasterio
from tqdm import tqdm

# Seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# =============================================================================
# TINY MODEL - Only ~60K parameters
# =============================================================================

class TinyUNet(nn.Module):
    """Super lightweight model for fast training"""
    
    def __init__(self, in_channels=11, num_classes=2):
        super().__init__()
        
        # Encoder (tiny)
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # Decoder
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        
        self.out = nn.Conv2d(16, num_classes, 1)
        self.pool = nn.MaxPool2d(2)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        
        d2 = self.dec2(torch.cat([self.up2(e3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        
        return self.out(d1)


# =============================================================================
# SIMPLE DATASET
# =============================================================================

# MARIDA Class Mapping (from README)
# 1: Marine Debris, 2: Dense Sargassum, 3: Sparse Sargassum, 4: Natural Organic Material
# 5: Ship, 6: Clouds, 7: Marine Water, 8: Sediment-Laden Water, 9: Foam
# 10: Turbid Water, 11: Shallow Water, 12: Waves, 13: Cloud Shadows, 14: Wakes, 15: Mixed Water

# BUT from our data inspection, the actual values are different. 
# Class 5 appears frequently - need to check which is actually debris

class FastDataset(Dataset):
    """Fast loading dataset - loads all valid files"""
    
    # Normalization values for MARIDA dataset
    MEAN = np.array([0.05, 0.06, 0.06, 0.05, 0.08, 0.10, 0.11, 0.10, 0.12, 0.13, 0.09], dtype=np.float32)
    STD = np.array([0.03, 0.03, 0.03, 0.03, 0.04, 0.05, 0.05, 0.05, 0.05, 0.06, 0.05], dtype=np.float32)
    
    # MARIDA debris-related classes (treating all as "potential debris" for binary)
    # Marine Debris = 1, Dense Sargassum = 2, Sparse Sargassum = 3, Natural Organic = 4
    DEBRIS_CLASSES = {1, 2, 3, 4}
    
    def __init__(self, data_dir, max_samples=200, augment=False):
        self.data_dir = Path(data_dir)
        self.augment = augment
        self.samples = []
        
        # Find all image/label pairs
        patches_dir = self.data_dir / "patches"
        for folder in os.listdir(patches_dir):
            folder_path = patches_dir / folder
            if folder_path.is_dir():
                for file in os.listdir(folder_path):
                    if file.endswith('.tif') and '_cl' not in file and '_conf' not in file:
                        img_path = folder_path / file
                        label_path = folder_path / file.replace('.tif', '_cl.tif')
                        if label_path.exists():
                            self.samples.append((img_path, label_path))
        
        # Limit samples for speed
        random.shuffle(self.samples)
        self.samples = self.samples[:max_samples]
        print(f"Loaded {len(self.samples)} samples")
    
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
        
        # Normalize each band
        for i in range(min(len(self.MEAN), image.shape[0])):
            image[i] = (image[i] - self.MEAN[i]) / (self.STD[i] + 1e-8)
        
        # Clamp values to prevent extreme outliers
        image = np.clip(image, -10, 10)
        
        # Binary classification: debris-related vs not debris
        # Classes 1-4 are debris-related in MARIDA
        binary_label = np.isin(label.astype(np.int32), list(self.DEBRIS_CLASSES)).astype(np.int64)
        
        # Simple augmentation
        if self.augment and random.random() > 0.5:
            image = np.flip(image, axis=2).copy()
            binary_label = np.flip(binary_label, axis=0).copy()
        
        return {
            "image": torch.from_numpy(image.copy()),
            "mask": torch.from_numpy(binary_label.copy()),
        }


# =============================================================================
# TRAINING
# =============================================================================

def train():
    print("=" * 50)
    print("FAST TRAINING - ~5-10 minutes")
    print("=" * 50)
    
    # Config
    DATA_DIR = Path("data/marida")
    OUTPUT_DIR = Path("outputs/models")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    EPOCHS = 10
    BATCH_SIZE = 16
    LR = 0.0005  # Lower learning rate for stability
    MAX_TRAIN_SAMPLES = 200
    MAX_VAL_SAMPLES = 50
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Data
    print("\nLoading data...")
    train_data = FastDataset(DATA_DIR, max_samples=MAX_TRAIN_SAMPLES, augment=True)
    val_data = FastDataset(DATA_DIR, max_samples=MAX_VAL_SAMPLES, augment=False)
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Check data
    print("\nChecking data...")
    sample = train_data[0]
    print(f"Image shape: {sample['image'].shape}")
    print(f"Image range: [{sample['image'].min():.4f}, {sample['image'].max():.4f}]")
    print(f"Mask shape: {sample['mask'].shape}")
    print(f"Mask unique: {torch.unique(sample['mask'])}")
    print(f"Mask debris pixels: {(sample['mask'] == 1).sum().item()}")
    
    # Model
    model = TinyUNet(in_channels=11, num_classes=2).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {params:,}")
    
    # Training setup - use weighted loss for class imbalance
    # Most pixels are NOT debris, so weight debris class higher
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.1, 0.9]).to(device))
    optimizer = Adam(model.parameters(), lr=LR)
    
    best_loss = float('inf')
    
    print("\nStarting training...")
    print("-" * 50)
    
    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss = 0
        train_samples = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch in pbar:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            
            # Check for NaN in input
            if torch.isnan(images).any():
                print("WARNING: NaN in input images!")
                continue
            
            optimizer.zero_grad()
            outputs = model(images)
            
            # Check for NaN in output
            if torch.isnan(outputs).any():
                print("WARNING: NaN in model output!")
                continue
            
            loss = criterion(outputs, masks)
            
            # Check for NaN in loss
            if torch.isnan(loss):
                print("WARNING: NaN loss, skipping batch")
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            train_samples += images.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        train_loss = train_loss / max(train_samples, 1)
        
        # Validate
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        val_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                if not torch.isnan(loss):
                    val_loss += loss.item() * images.size(0)
                    val_samples += images.size(0)
                
                preds = outputs.argmax(dim=1)
                correct += (preds == masks).sum().item()
                total += masks.numel()
        
        val_loss = val_loss / max(val_samples, 1)
        accuracy = correct / max(total, 1) * 100
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Acc={accuracy:.1f}%")
        
        # Save best
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": {"in_channels": 11, "num_classes": 2, "base_filters": 16},
                "epoch": epoch,
                "val_loss": val_loss,
            }, OUTPUT_DIR / "best_model.pth")
            print(f"  -> Saved best model!")
    
    print("-" * 50)
    print(f"Done! Best val loss: {best_loss:.4f}")
    print(f"Model saved to: {OUTPUT_DIR / 'best_model.pth'}")


if __name__ == "__main__":
    os.chdir(Path(__file__).parent.parent)
    train()
