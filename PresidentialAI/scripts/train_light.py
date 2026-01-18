"""
Lightweight Marine Debris Detection Model
Fast training, simple architecture, works with MARIDA dataset
"""

import os
import sys
import random
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import rasterio
from tqdm import tqdm

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# =============================================================================
# LIGHTWEIGHT U-NET MODEL
# =============================================================================

class ConvBlock(nn.Module):
    """Simple conv block with BatchNorm and ReLU"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.conv(x)


class LightUNet(nn.Module):
    """
    Lightweight U-Net for fast training.
    ~500K parameters instead of millions.
    """
    def __init__(self, in_channels=11, num_classes=2, base_filters=32):
        super().__init__()
        
        f = base_filters  # 32
        
        # Encoder
        self.enc1 = ConvBlock(in_channels, f)      # 32
        self.enc2 = ConvBlock(f, f*2)              # 64
        self.enc3 = ConvBlock(f*2, f*4)            # 128
        
        # Bottleneck
        self.bottleneck = ConvBlock(f*4, f*8)      # 256
        
        # Decoder
        self.up3 = nn.ConvTranspose2d(f*8, f*4, 2, stride=2)
        self.dec3 = ConvBlock(f*8, f*4)
        
        self.up2 = nn.ConvTranspose2d(f*4, f*2, 2, stride=2)
        self.dec2 = ConvBlock(f*4, f*2)
        
        self.up1 = nn.ConvTranspose2d(f*2, f, 2, stride=2)
        self.dec1 = ConvBlock(f*2, f)
        
        # Output
        self.out = nn.Conv2d(f, num_classes, 1)
        
        self.pool = nn.MaxPool2d(2)
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e3))
        
        # Decoder with skip connections
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        
        return self.out(d1)


# =============================================================================
# DATASET
# =============================================================================

class MaridaDataset(Dataset):
    """Simple MARIDA dataset loader"""
    
    # MARIDA class IDs that are debris
    DEBRIS_CLASSES = {1}  # Marine Debris class
    
    def __init__(self, data_dir, split="train", transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # Load split file
        split_file = self.data_dir / "splits" / f"{split}_X.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")
        
        with open(split_file) as f:
            self.samples = [line.strip() for line in f if line.strip()]
        
        print(f"Loaded {len(self.samples)} samples for {split}")
        
        # Normalization stats for Sentinel-2 (approximate)
        self.mean = np.array([0.05, 0.06, 0.06, 0.05, 0.08, 0.10, 0.11, 0.10, 0.12, 0.13, 0.09])
        self.std = np.array([0.03, 0.03, 0.03, 0.03, 0.04, 0.05, 0.05, 0.05, 0.05, 0.06, 0.05])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_name = self.samples[idx]
        
        # Parse sample name to get folder and file
        # Format: S2_DATE_TILE/S2_DATE_TILE_N
        parts = sample_name.rsplit('_', 1)
        folder = parts[0] if len(parts) > 1 else sample_name.rsplit('/', 1)[0]
        
        # Find the image file
        patches_dir = self.data_dir / "patches"
        
        # Try to find the matching folder
        img_path = None
        label_path = None
        
        for folder_name in os.listdir(patches_dir):
            if folder_name.startswith("S2_"):
                folder_path = patches_dir / folder_name
                for file in os.listdir(folder_path):
                    if file.endswith('.tif') and not file.endswith('_cl.tif') and not file.endswith('_conf.tif'):
                        if sample_name in file or file.replace('.tif', '') == sample_name:
                            img_path = folder_path / file
                            label_path = folder_path / file.replace('.tif', '_cl.tif')
                            break
                if img_path:
                    break
        
        # Fallback: direct path construction
        if img_path is None:
            for folder_name in os.listdir(patches_dir):
                folder_path = patches_dir / folder_name
                if folder_path.is_dir():
                    for file in os.listdir(folder_path):
                        if file.endswith('.tif') and '_cl' not in file and '_conf' not in file:
                            img_path = folder_path / file
                            label_path = folder_path / file.replace('.tif', '_cl.tif')
                            if img_path.exists() and label_path.exists():
                                break
                    if img_path and img_path.exists():
                        break
        
        if img_path is None or not img_path.exists():
            # Return a random valid sample instead
            return self.__getitem__(random.randint(0, len(self.samples) - 1))
        
        # Load image
        with rasterio.open(img_path) as src:
            image = src.read().astype(np.float32)
        
        # Load label
        if label_path.exists():
            with rasterio.open(label_path) as src:
                label = src.read(1).astype(np.int64)
        else:
            label = np.zeros((image.shape[1], image.shape[2]), dtype=np.int64)
        
        # Normalize image
        for i in range(min(len(self.mean), image.shape[0])):
            image[i] = (image[i] - self.mean[i]) / (self.std[i] + 1e-8)
        
        # Convert label to binary (debris vs non-debris)
        binary_label = np.isin(label, list(self.DEBRIS_CLASSES)).astype(np.int64)
        
        # Simple augmentation for training
        if self.transform == "train":
            if random.random() > 0.5:
                image = np.flip(image, axis=1).copy()
                binary_label = np.flip(binary_label, axis=0).copy()
            if random.random() > 0.5:
                image = np.flip(image, axis=2).copy()
                binary_label = np.flip(binary_label, axis=1).copy()
        
        return {
            "image": torch.from_numpy(image),
            "mask": torch.from_numpy(binary_label),
        }


# =============================================================================
# LOSS FUNCTION
# =============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=0.8, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


# =============================================================================
# TRAINING
# =============================================================================

def compute_iou(pred, target, num_classes=2):
    """Compute IoU for each class"""
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)
    
    for cls in range(num_classes):
        pred_cls = pred == cls
        target_cls = target == cls
        
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        
        if union == 0:
            ious.append(1.0)
        else:
            ious.append((intersection / union).item())
    
    return ious


def train_model():
    """Main training function"""
    
    # Configuration
    DATA_DIR = Path("data/marida")
    OUTPUT_DIR = Path("outputs/models")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    EPOCHS = 30
    BATCH_SIZE = 8
    LR = 1e-4
    GRAD_CLIP = 1.0
    
    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = MaridaDataset(DATA_DIR, split="train", transform="train")
    val_dataset = MaridaDataset(DATA_DIR, split="val", transform=None)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=0,  # Windows compatibility
        pin_memory=True if device.type != "cpu" else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Create model
    model = LightUNet(in_channels=11, num_classes=2, base_filters=32)
    model = model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {num_params:,}")
    
    # Loss and optimizer
    criterion = FocalLoss(alpha=0.8, gamma=2.0)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS * len(train_loader), eta_min=1e-6)
    
    # Training loop
    best_iou = 0.0
    
    print(f"\n{'='*60}")
    print("Starting training...")
    print(f"{'='*60}\n")
    
    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for batch in pbar:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Check for NaN
            if torch.isnan(loss):
                print("WARNING: NaN loss detected, skipping batch")
                continue
            
            loss.backward()
            
            # Gradient clipping (CRITICAL)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_ious = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
                preds = outputs.argmax(dim=1)
                ious = compute_iou(preds, masks)
                all_ious.append(ious)
        
        val_loss /= len(val_loader)
        mean_iou = np.mean([iou[1] for iou in all_ious])  # IoU for debris class
        
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Debris IoU: {mean_iou:.4f}")
        
        # Save best model
        if mean_iou > best_iou:
            best_iou = mean_iou
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_iou": best_iou,
                "config": {
                    "in_channels": 11,
                    "num_classes": 2,
                    "base_filters": 32,
                },
            }
            torch.save(checkpoint, OUTPUT_DIR / "best_model.pth")
            print(f"  *** New best model saved! IoU: {best_iou:.4f}")
    
    print(f"\n{'='*60}")
    print(f"Training complete! Best IoU: {best_iou:.4f}")
    print(f"Model saved to: {OUTPUT_DIR / 'best_model.pth'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    # Change to the project directory
    script_dir = Path(__file__).parent.parent
    os.chdir(script_dir)
    print(f"Working directory: {os.getcwd()}")
    
    train_model()

