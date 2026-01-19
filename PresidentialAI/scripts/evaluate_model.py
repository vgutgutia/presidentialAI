#!/usr/bin/env python3
"""
Comprehensive Model Evaluation Script
Evaluates model performance and generates detailed report
"""

import sys
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import time
import json

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import models
ADVANCED_AVAILABLE = False
IMPROVED_AVAILABLE = False

try:
    from scripts.train_advanced import AdvancedUNet, AdvancedSyntheticDataset
    ADVANCED_AVAILABLE = True
except:
    pass

try:
    from scripts.train_improved import ImprovedUNet, SyntheticDataset
    IMPROVED_AVAILABLE = True
except:
    pass

if not ADVANCED_AVAILABLE and not IMPROVED_AVAILABLE:
    print("Could not import model classes")
    sys.exit(1)

# Configuration
MODEL_DIR = Path(__file__).parent.parent / "outputs" / "models"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "predictions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IN_CHANNELS = 11
NUM_CLASSES = 2
TEST_SAMPLES = 200
BATCH_SIZE = 16

# Device
if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print("=" * 70)
print("📊 COMPREHENSIVE MODEL EVALUATION REPORT")
print("=" * 70)
print(f"Device: {DEVICE}")
print()

# Find available models
available_models = list(MODEL_DIR.glob("*.pth"))
if not available_models:
    print("❌ No trained models found!")
    print(f"   Looking in: {MODEL_DIR}")
    sys.exit(1)

print(f"Found {len(available_models)} model(s):")
for i, model_path in enumerate(available_models, 1):
    size = model_path.stat().st_size / (1024 * 1024)  # MB
    print(f"  {i}. {model_path.name} ({size:.1f} MB)")

# Use the best trained model (prefer improved over advanced if advanced is early)
if "improved_model.pth" in [m.name for m in available_models]:
    model_path = MODEL_DIR / "improved_model.pth"
    # Check if advanced model is well-trained
    try:
        adv_path = MODEL_DIR / "advanced_model.pth"
        if adv_path.exists():
            adv_checkpoint = torch.load(adv_path, map_location="cpu", weights_only=False)
            adv_epoch = adv_checkpoint.get('epoch', 0)
            # Only use advanced if it's trained for at least 10 epochs
            if isinstance(adv_epoch, int) and adv_epoch >= 10:
                model_path = adv_path
    except:
        pass
elif "advanced_model.pth" in [m.name for m in available_models]:
    model_path = MODEL_DIR / "advanced_model.pth"
else:
    model_path = available_models[0]

print(f"\n📦 Loading model: {model_path.name}")

# Determine model type
if "advanced" in model_path.name.lower() and ADVANCED_AVAILABLE:
    MODEL_CLASS = AdvancedUNet
    MODEL_NAME = "Advanced"
    DatasetClass = AdvancedSyntheticDataset
elif "improved" in model_path.name.lower() and IMPROVED_AVAILABLE:
    MODEL_CLASS = ImprovedUNet
    MODEL_NAME = "Improved"
    DatasetClass = SyntheticDataset
elif ADVANCED_AVAILABLE:
    MODEL_CLASS = AdvancedUNet
    MODEL_NAME = "Advanced"
    DatasetClass = AdvancedSyntheticDataset
else:
    MODEL_CLASS = ImprovedUNet
    MODEL_NAME = "Improved"
    DatasetClass = SyntheticDataset

print(f"Model Type: {MODEL_NAME}")
print()

# Load model
model = MODEL_CLASS(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
try:
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()
    
    # Get checkpoint info
    epoch = checkpoint.get('epoch', 'N/A')
    val_loss = checkpoint.get('val_loss', 'N/A')
    f1 = checkpoint.get('f1', 'N/A')
    iou = checkpoint.get('iou', 'N/A')
    
    print(f"✓ Model loaded successfully")
    print(f"  Model type: {MODEL_NAME}")
    print(f"  Training epoch: {epoch}")
    print(f"  Validation loss: {val_loss:.4f}" if isinstance(val_loss, float) else f"  Validation loss: {val_loss}")
    print(f"  F1 Score: {f1:.4f}" if isinstance(f1, float) else f"  F1 Score: {f1}")
    print(f"  IoU: {iou:.4f}" if isinstance(iou, float) else f"  IoU: {iou}")
    print()
except Exception as e:
    print(f"❌ Error loading model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Create test dataset
print(f"📊 Creating test dataset ({TEST_SAMPLES} samples)...")
if MODEL_NAME == "Advanced":
    test_dataset = DatasetClass(num_samples=TEST_SAMPLES, augment=False)
else:
    test_dataset = DatasetClass(num_samples=TEST_SAMPLES)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# Evaluation
print("🔍 Running evaluation...")
print()

all_preds = []
all_labels = []
all_probs = []
inference_times = []

model.eval()
with torch.no_grad():
    for batch_idx, (images, masks) in enumerate(test_loader):
        images = images.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)
        
        # Time inference
        start_time = time.time()
        outputs = model(images)
        inference_time = time.time() - start_time
        inference_times.append(inference_time)
        
        # Get predictions
        probs = F.softmax(outputs, dim=1)
        debris_probs = probs[:, 1].cpu().numpy()
        preds = outputs.argmax(dim=1).cpu().numpy()
        
        all_preds.extend(preds.flatten())
        all_labels.extend(masks.cpu().numpy().flatten())
        all_probs.extend(debris_probs.flatten())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)

# Calculate metrics
print("=" * 70)
print("📈 PERFORMANCE METRICS")
print("=" * 70)

# Overall metrics
f1 = f1_score(all_labels, all_preds, pos_label=1, zero_division=0)
precision = precision_score(all_labels, all_preds, pos_label=1, zero_division=0)
recall = recall_score(all_labels, all_preds, pos_label=1, zero_division=0)

# IoU
intersection = np.sum((all_preds == 1) & (all_labels == 1))
union = np.sum((all_preds == 1) | (all_labels == 1))
iou = intersection / (union + 1e-8)

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

# Accuracy
accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)

# Probability statistics
debris_mask = all_labels == 1
if debris_mask.sum() > 0:
    debris_probs_mean = all_probs[debris_mask].mean()
    debris_probs_max = all_probs[debris_mask].max()
    debris_probs_std = all_probs[debris_mask].std()
else:
    debris_probs_mean = debris_probs_max = debris_probs_std = 0.0

# Inference speed
avg_inference_time = np.mean(inference_times)
total_pixels = len(all_labels)
pixels_per_second = total_pixels / sum(inference_times)

print(f"\n🎯 Classification Metrics:")
print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
print(f"  F1 Score:  {f1:.4f} ({f1*100:.2f}%)")
print(f"  IoU:       {iou:.4f} ({iou*100:.2f}%)")

print(f"\n📊 Confusion Matrix:")
print(f"  True Negatives:  {tn:>10,}")
print(f"  False Positives: {fp:>10,}")
print(f"  False Negatives: {fn:>10,}")
print(f"  True Positives:  {tp:>10,}")

print(f"\n📈 Probability Statistics:")
print(f"  Mean (debris pixels): {debris_probs_mean:.4f}")
print(f"  Max (debris pixels):  {debris_probs_max:.4f}")
print(f"  Std (debris pixels): {debris_probs_std:.4f}")

print(f"\n⚡ Performance:")
print(f"  Avg inference time: {avg_inference_time*1000:.2f}ms per batch")
print(f"  Throughput: {pixels_per_second/1e6:.2f}M pixels/sec")
print(f"  Images/sec: {TEST_SAMPLES/sum(inference_times):.1f}")

# Threshold analysis
print(f"\n🎚️  Threshold Analysis:")
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
print(f"  {'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'IoU':<12}")
print(f"  {'-'*60}")
for thresh in thresholds:
    thresh_preds = (all_probs >= thresh).astype(int)
    if thresh_preds.sum() > 0:
        thresh_f1 = f1_score(all_labels, thresh_preds, pos_label=1, zero_division=0)
        thresh_prec = precision_score(all_labels, thresh_preds, pos_label=1, zero_division=0)
        thresh_rec = recall_score(all_labels, thresh_preds, pos_label=1, zero_division=0)
        thresh_intersection = np.sum((thresh_preds == 1) & (all_labels == 1))
        thresh_union = np.sum((thresh_preds == 1) | (all_labels == 1))
        thresh_iou = thresh_intersection / (thresh_union + 1e-8)
        print(f"  {thresh:<12.2f} {thresh_prec:<12.4f} {thresh_rec:<12.4f} {thresh_f1:<12.4f} {thresh_iou:<12.4f}")
    else:
        print(f"  {thresh:<12.2f} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}")

# Class distribution
print(f"\n📋 Class Distribution:")
print(f"  Background pixels: {np.sum(all_labels == 0):,} ({np.sum(all_labels == 0)/len(all_labels)*100:.2f}%)")
print(f"  Debris pixels:      {np.sum(all_labels == 1):,} ({np.sum(all_labels == 1)/len(all_labels)*100:.2f}%)")

# Save report
report = {
    "model": model_path.name,
    "model_type": MODEL_NAME,
    "device": DEVICE,
    "test_samples": TEST_SAMPLES,
    "metrics": {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "iou": float(iou),
    },
    "confusion_matrix": {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    },
    "probability_stats": {
        "mean": float(debris_probs_mean),
        "max": float(debris_probs_max),
        "std": float(debris_probs_std),
    },
    "performance": {
        "avg_inference_time_ms": float(avg_inference_time * 1000),
        "throughput_mpixels_per_sec": float(pixels_per_second / 1e6),
        "images_per_sec": float(TEST_SAMPLES / sum(inference_times)),
    },
    "checkpoint_info": {
        "epoch": epoch if isinstance(epoch, int) else str(epoch),
        "val_loss": float(val_loss) if isinstance(val_loss, float) else str(val_loss),
        "checkpoint_f1": float(f1) if isinstance(f1, float) else str(f1),
        "checkpoint_iou": float(iou) if isinstance(iou, float) else str(iou),
    }
}

report_path = OUTPUT_DIR / "evaluation_report.json"
with open(report_path, 'w') as f:
    json.dump(report, f, indent=2)

print(f"\n💾 Report saved to: {report_path}")

print("\n" + "=" * 70)
print("✅ EVALUATION COMPLETE")
print("=" * 70)
