#!/usr/bin/env python3
"""Test the model inference directly to debug the issue."""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

# Add paths
PRESIDENTIAL_AI_PATH = Path(__file__).parent / "PresidentialAI"
sys.path.insert(0, str(PRESIDENTIAL_AI_PATH))

from scripts.train_improved import ImprovedUNet, SyntheticDataset

# Load model
MODEL_PATH = PRESIDENTIAL_AI_PATH / "outputs" / "models" / "improved_model.pth"
device = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Loading model from {MODEL_PATH}")
model = ImprovedUNet(in_channels=11, num_classes=2)
checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

print(f"✅ Model loaded on {device}")
print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
print(f"   F1: {checkpoint.get('f1', 'N/A')}")

# Test with synthetic data (what it was trained on)
print("\n" + "="*60)
print("Testing with SYNTHETIC data (training format):")
print("="*60)
dataset = SyntheticDataset(num_samples=1, image_size=256)
test_image, test_mask = dataset[0]

print(f"Input shape: {test_image.shape}, range=[{test_image.min():.3f}, {test_image.max():.3f}]")
print(f"Ground truth mask: {test_mask.sum()} debris pixels")

# Run inference
with torch.no_grad():
    image_tensor = test_image.unsqueeze(0).to(device)
    output = model(image_tensor)
    probs = F.softmax(output, dim=1)
    debris_probs = probs[0, 1].cpu().numpy()

print(f"\nInference Results:")
print(f"  Probability range: [{debris_probs.min():.4f}, {debris_probs.max():.4f}]")
print(f"  Mean probability: {debris_probs.mean():.4f}")
print(f"  Max probability: {debris_probs.max():.4f}")

# Test different thresholds
thresholds = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
print(f"\nThreshold Analysis:")
for thresh in thresholds:
    detected = (debris_probs >= thresh).sum()
    print(f"  Threshold {thresh:.2f}: {detected} pixels detected")

# Compare with ground truth
if test_mask.sum() > 0:
    intersection = ((debris_probs >= 0.3) & (test_mask == 1)).sum()
    union = ((debris_probs >= 0.3) | (test_mask == 1)).sum()
    iou = intersection / (union + 1e-8)
    print(f"\nIoU at 0.3 threshold: {iou:.4f}")

print("\n" + "="*60)
print("Testing with REAL image format (scaled):")
print("="*60)

# Simulate a real image (like what comes from GeoTIFF)
real_image = np.random.rand(11, 256, 256).astype(np.float32) * 0.1  # Lower values
print(f"Input shape: {real_image.shape}, range=[{real_image.min():.3f}, {real_image.max():.3f}]")

with torch.no_grad():
    image_tensor = torch.from_numpy(real_image).unsqueeze(0).to(device)
    output = model(image_tensor)
    probs = F.softmax(output, dim=1)
    debris_probs = probs[0, 1].cpu().numpy()

print(f"\nInference Results:")
print(f"  Probability range: [{debris_probs.min():.4f}, {debris_probs.max():.4f}]")
print(f"  Mean probability: {debris_probs.mean():.4f}")
print(f"  Max probability: {debris_probs.max():.4f}")

for thresh in [0.01, 0.05, 0.1, 0.2]:
    detected = (debris_probs >= thresh).sum()
    print(f"  Threshold {thresh:.2f}: {detected} pixels detected")
