#!/usr/bin/env python3
"""
Inference script for the improved model.
"""

import sys
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_improved import ImprovedUNet, SyntheticDataset

# Configuration
MODEL_PATH = Path(__file__).parent.parent / "outputs" / "models" / "improved_model.pth"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "predictions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IN_CHANNELS = 11
NUM_CLASSES = 2

# Device
if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print("=" * 60)
print("🌊 Marine Debris Detection - Improved Model Inference")
print("=" * 60)
print(f"Device: {DEVICE}")
print(f"Model: {MODEL_PATH}")

# Load model
if not MODEL_PATH.exists():
    print(f"\n[ERROR] Model not found: {MODEL_PATH}")
    sys.exit(1)

print("\nLoading improved model...")
model = ImprovedUNet(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(DEVICE)
model.eval()

print(f"✓ Model loaded (Epoch {checkpoint.get('epoch', 'N/A')}, F1: {checkpoint.get('f1', 0):.4f})")

# Create test images
print("\nGenerating test images...")
test_dataset = SyntheticDataset(num_samples=5, image_size=256)
test_image, test_mask = test_dataset[0]
test_image = test_image.unsqueeze(0).to(DEVICE)

print(f"Input shape: {test_image.shape}")

# Run inference
print("\nRunning inference...")
with torch.no_grad():
    output = model(test_image)
    probs = F.softmax(output, dim=1)
    debris_prob = probs[0, 1].cpu().numpy()
    pred_mask = output.argmax(dim=1)[0].cpu().numpy()

print(f"Output shape: {output.shape}")
print(f"Debris probability range: [{debris_prob.min():.4f}, {debris_prob.max():.4f}]")
print(f"Predicted debris pixels: {np.sum(pred_mask == 1)}")

# Test different thresholds
print("\n" + "=" * 60)
print("Detection at Different Thresholds:")
print("=" * 60)
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
for thresh in thresholds:
    count = np.sum(debris_prob > thresh)
    pct = 100 * count / debris_prob.size
    print(f"  Threshold {thresh:.1f}: {count:5d} pixels ({pct:5.2f}%)")

# Create visualization
print("\nCreating visualization...")
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

# Input
rgb_input = test_image[0, [3, 2, 1]].cpu().numpy()
rgb_input = np.transpose(rgb_input, (1, 2, 0))
rgb_input = (rgb_input - rgb_input.min()) / (rgb_input.max() - rgb_input.min() + 1e-8)
axes[0].imshow(rgb_input)
axes[0].set_title("Input Image (RGB)")
axes[0].axis('off')

# Ground truth
test_mask_np = test_mask.numpy()
axes[1].imshow(test_mask_np, cmap='gray')
axes[1].set_title("Ground Truth")
axes[1].axis('off')

# Predicted mask
axes[2].imshow(pred_mask, cmap='gray')
axes[2].set_title("Predicted Mask (argmax)")
axes[2].axis('off')

# Probability heatmap
im = axes[3].imshow(debris_prob, cmap='hot', vmin=0, vmax=1)
axes[3].set_title("Debris Probability")
axes[3].axis('off')
plt.colorbar(im, ax=axes[3], fraction=0.046)

plt.tight_layout()
output_path = OUTPUT_DIR / "improved_inference_result.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✓ Visualization saved: {output_path}")

# Save probability map
prob_path = OUTPUT_DIR / "improved_debris_probability.npy"
np.save(prob_path, debris_prob)
print(f"✓ Probability map saved: {prob_path}")

# Statistics
print("\n" + "=" * 60)
print("Inference Results:")
print("=" * 60)
print(f"Total pixels: {pred_mask.size}")
print(f"Predicted debris pixels: {np.sum(pred_mask == 1)} ({100*np.sum(pred_mask == 1)/pred_mask.size:.2f}%)")
print(f"Actual debris pixels: {np.sum(test_mask_np == 1)} ({100*np.sum(test_mask_np == 1)/test_mask_np.size:.2f}%)")
print(f"Max debris probability: {debris_prob.max():.4f}")
print(f"Mean debris probability: {debris_prob.mean():.4f}")

# Calculate IoU
if np.sum(test_mask_np == 1) > 0:
    intersection = np.sum((pred_mask == 1) & (test_mask_np == 1))
    union = np.sum((pred_mask == 1) | (test_mask_np == 1))
    iou = intersection / (union + 1e-8)
    print(f"IoU (Intersection over Union): {iou:.4f}")

# Compare with old model
print("\n" + "=" * 60)
print("Improvement Summary:")
print("=" * 60)
old_max_prob = 0.3235  # From previous model
new_max_prob = debris_prob.max()
improvement = ((new_max_prob - old_max_prob) / old_max_prob) * 100 if old_max_prob > 0 else 0
print(f"Max probability: {old_max_prob:.4f} → {new_max_prob:.4f} ({improvement:+.1f}%)")
print(f"Mean probability: 0.0356 → {debris_prob.mean():.4f} ({((debris_prob.mean() - 0.0356) / 0.0356 * 100):+.1f}%)")

print("\n" + "=" * 60)
print("Inference Complete! ✓")
print("=" * 60)
