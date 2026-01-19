#!/usr/bin/env python3
"""
Quick inference script for the trained model.
Works with the quick_trained_model.pth from train_quick.py
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

from scripts.train_quick import SimpleUNet, SyntheticDataset

# Configuration
MODEL_PATH = Path(__file__).parent.parent / "outputs" / "models" / "quick_trained_model.pth"
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
print("🌊 Marine Debris Detection - Inference")
print("=" * 60)
print(f"Device: {DEVICE}")
print(f"Model: {MODEL_PATH}")

# Load model
if not MODEL_PATH.exists():
    print(f"\n[ERROR] Model not found: {MODEL_PATH}")
    print("Please train a model first: python scripts/train_quick.py")
    sys.exit(1)

print("\nLoading model...")
model = SimpleUNet(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(DEVICE)
model.eval()
print("✓ Model loaded")

# Create a test image (synthetic satellite data)
print("\nGenerating test image...")
test_dataset = SyntheticDataset(num_samples=1, image_size=256)
test_image, test_mask = test_dataset[0]
test_image = test_image.unsqueeze(0).to(DEVICE)  # Add batch dimension

print(f"Input shape: {test_image.shape}")

# Run inference
print("\nRunning inference...")
with torch.no_grad():
    output = model(test_image)
    probs = F.softmax(output, dim=1)
    debris_prob = probs[0, 1].cpu().numpy()  # Get debris probability
    pred_mask = output.argmax(dim=1)[0].cpu().numpy()  # Get predictions

print(f"Output shape: {output.shape}")
print(f"Debris probability range: [{debris_prob.min():.4f}, {debris_prob.max():.4f}]")
print(f"Predicted debris pixels: {np.sum(pred_mask == 1)}")

# Create visualization
print("\nCreating visualization...")
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

# Input (show first 3 bands as RGB)
rgb_input = test_image[0, [3, 2, 1]].cpu().numpy()  # Red, Green, Blue bands
rgb_input = np.transpose(rgb_input, (1, 2, 0))
rgb_input = (rgb_input - rgb_input.min()) / (rgb_input.max() - rgb_input.min() + 1e-8)
axes[0].imshow(rgb_input)
axes[0].set_title("Input Image (RGB)")
axes[0].axis('off')

# Ground truth mask
axes[1].imshow(test_mask.numpy(), cmap='gray')
axes[1].set_title("Ground Truth")
axes[1].axis('off')

# Predicted mask
axes[2].imshow(pred_mask, cmap='gray')
axes[2].set_title("Predicted Mask")
axes[2].axis('off')

# Probability heatmap
im = axes[3].imshow(debris_prob, cmap='hot', vmin=0, vmax=1)
axes[3].set_title("Debris Probability")
axes[3].axis('off')
plt.colorbar(im, ax=axes[3], fraction=0.046)

plt.tight_layout()
output_path = OUTPUT_DIR / "inference_result.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✓ Visualization saved: {output_path}")

# Save probability map as numpy array
prob_path = OUTPUT_DIR / "debris_probability.npy"
np.save(prob_path, debris_prob)
print(f"✓ Probability map saved: {prob_path}")

# Print statistics
print("\n" + "=" * 60)
print("Inference Results:")
print("=" * 60)
print(f"Total pixels: {pred_mask.size}")
print(f"Predicted debris pixels: {np.sum(pred_mask == 1)} ({100*np.sum(pred_mask == 1)/pred_mask.size:.2f}%)")
test_mask_np = test_mask.numpy()
print(f"Actual debris pixels: {np.sum(test_mask_np == 1)} ({100*np.sum(test_mask_np == 1)/test_mask_np.size:.2f}%)")
print(f"Max debris probability: {debris_prob.max():.4f}")
print(f"Mean debris probability: {debris_prob.mean():.4f}")

# Calculate IoU if ground truth available
if np.sum(test_mask.numpy() == 1) > 0:
    intersection = np.sum((pred_mask == 1) & (test_mask.numpy() == 1))
    union = np.sum((pred_mask == 1) | (test_mask.numpy() == 1))
    iou = intersection / (union + 1e-8)
    print(f"IoU (Intersection over Union): {iou:.4f}")

print("\n" + "=" * 60)
print("Inference Complete! ✓")
print("=" * 60)
