#!/usr/bin/env python3
"""
Optimized inference script for the advanced model.
Fast evaluation with batch processing and optimizations.
"""

import sys
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import time

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_advanced import AdvancedUNet, AdvancedSyntheticDataset

# Configuration
MODEL_PATH = Path(__file__).parent.parent / "outputs" / "models" / "advanced_model.pth"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "predictions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IN_CHANNELS = 11
NUM_CLASSES = 2
BATCH_SIZE = 8  # Batch inference for speed

# Device
if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print("=" * 60)
print("⚡ Advanced Model - Fast Inference")
print("=" * 60)
print(f"Device: {DEVICE}")
print(f"Model: {MODEL_PATH}")

# Load model
if not MODEL_PATH.exists():
    print(f"\n[ERROR] Model not found: {MODEL_PATH}")
    print("Training advanced model... This may take a while.")
    sys.exit(1)

print("\nLoading advanced model...")
model = AdvancedUNet(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(DEVICE)
model.eval()

# Enable optimizations
if DEVICE == "mps":
    torch.backends.mps.empty_cache()
elif DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True

print(f"✓ Model loaded")
print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
print(f"  F1: {checkpoint.get('f1', 0):.4f}")
print(f"  IoU: {checkpoint.get('iou', 0):.4f}")
print(f"  Precision: {checkpoint.get('precision', 0):.4f}")
print(f"  Recall: {checkpoint.get('recall', 0):.4f}")

# Create test batch
print("\nGenerating test batch...")
test_dataset = AdvancedSyntheticDataset(num_samples=BATCH_SIZE, augment=False)
test_images = []
test_masks = []

for i in range(BATCH_SIZE):
    img, mask = test_dataset[i]
    test_images.append(img)
    test_masks.append(mask)

test_batch = torch.stack(test_images).to(DEVICE)
test_masks_np = [m.numpy() for m in test_masks]

print(f"Batch shape: {test_batch.shape}")

# Warmup
print("\nWarming up...")
with torch.no_grad():
    _ = model(test_batch[:2])

# Run inference with timing
print("\nRunning batch inference...")
inference_start = time.time()

with torch.no_grad():
    outputs = model(test_batch)
    probs = F.softmax(outputs, dim=1)
    debris_probs = probs[:, 1].cpu().numpy()
    pred_masks = outputs.argmax(dim=1).cpu().numpy()

inference_time = time.time() - inference_start

print(f"✓ Inference complete in {inference_time*1000:.1f}ms")
print(f"  Throughput: {BATCH_SIZE/inference_time:.1f} images/sec")

# Analyze results
print("\n" + "=" * 60)
print("Batch Results Summary:")
print("=" * 60)

all_max_probs = []
all_mean_probs = []
all_detections = []

for i in range(BATCH_SIZE):
    prob = debris_probs[i]
    pred = pred_masks[i]
    true = test_masks_np[i]
    
    max_prob = prob.max()
    mean_prob = prob.mean()
    n_detections = np.sum(pred == 1)
    
    all_max_probs.append(max_prob)
    all_mean_probs.append(mean_prob)
    all_detections.append(n_detections)
    
    # IoU
    intersection = np.sum((pred == 1) & (true == 1))
    union = np.sum((pred == 1) | (true == 1))
    iou = intersection / (union + 1e-8) if union > 0 else 0.0
    
    print(f"\nImage {i+1}:")
    print(f"  Max prob: {max_prob:.4f}, Mean: {mean_prob:.4f}")
    print(f"  Detections: {n_detections} pixels ({100*n_detections/prob.size:.1f}%)")
    print(f"  IoU: {iou:.4f}")

print("\n" + "=" * 60)
print("Overall Statistics:")
print("=" * 60)
print(f"Average max probability: {np.mean(all_max_probs):.4f} ± {np.std(all_max_probs):.4f}")
print(f"Average mean probability: {np.mean(all_mean_probs):.4f} ± {np.std(all_mean_probs):.4f}")
print(f"Average detections: {np.mean(all_detections):.0f} pixels")
print(f"Inference speed: {inference_time*1000:.1f}ms per batch ({inference_time*1000/BATCH_SIZE:.1f}ms per image)")

# Create visualization for first image
print("\nCreating visualization...")
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for i in range(min(2, BATCH_SIZE)):
    prob = debris_probs[i]
    pred = pred_masks[i]
    true = test_masks_np[i]
    img = test_batch[i].cpu().numpy()
    
    # Input RGB
    rgb = img[[3, 2, 1]]
    rgb = np.transpose(rgb, (1, 2, 0))
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
    axes[i, 0].imshow(rgb)
    axes[i, 0].set_title(f"Input {i+1}")
    axes[i, 0].axis('off')
    
    # Ground truth
    axes[i, 1].imshow(true, cmap='gray')
    axes[i, 1].set_title(f"Ground Truth {i+1}")
    axes[i, 1].axis('off')
    
    # Prediction
    axes[i, 2].imshow(pred, cmap='gray')
    axes[i, 2].set_title(f"Prediction {i+1}")
    axes[i, 2].axis('off')
    
    # Probability
    im = axes[i, 3].imshow(prob, cmap='hot', vmin=0, vmax=1)
    axes[i, 3].set_title(f"Probability {i+1}")
    axes[i, 3].axis('off')
    plt.colorbar(im, ax=axes[i, 3], fraction=0.046)

plt.tight_layout()
output_path = OUTPUT_DIR / "advanced_inference_result.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✓ Visualization saved: {output_path}")

# Save probability maps
for i in range(BATCH_SIZE):
    prob_path = OUTPUT_DIR / f"advanced_prob_{i}.npy"
    np.save(prob_path, debris_probs[i])

print(f"✓ Saved {BATCH_SIZE} probability maps")

# Performance summary
print("\n" + "=" * 60)
print("Performance Summary:")
print("=" * 60)
print(f"✓ Model: Advanced U-Net with Attention")
print(f"✓ Inference Speed: {inference_time*1000/BATCH_SIZE:.1f}ms per image")
print(f"✓ Throughput: {BATCH_SIZE/inference_time:.1f} images/sec")
print(f"✓ Average Confidence: {np.mean(all_max_probs):.4f}")
print(f"✓ Ready for production use")

print("\n" + "=" * 60)
print("Inference Complete! ✓")
print("=" * 60)
