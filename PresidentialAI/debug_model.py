#!/usr/bin/env python3
"""Debug script to check what the model is actually predicting."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
import rasterio

from src.models.segformer import load_model
from src.data.dataset import MaridaDataset

def debug_model(model_path="outputs/models/best_model.pth", data_dir="data/marida"):
    print("=" * 60)
    print("DEBUGGING MODEL PREDICTIONS")
    print("=" * 60)
    
    # Device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Device: {device}")
    
    # Load dataset WITHOUT normalization first to see raw values
    print("\n1. Loading dataset...")
    dataset = MaridaDataset(
        root_dir=data_dir,
        split="train",
        transform=None,
        normalization=None,  # No normalization for debugging
        target_channels=11,
    )
    print(f"   Total samples: {len(dataset)}")
    
    # Find a sample WITH debris
    print("\n2. Finding samples with debris...")
    debris_samples = []
    for i in range(min(100, len(dataset))):
        sample = dataset[i]
        mask = sample["mask"].numpy()
        debris_pixels = np.sum(mask == 1)
        if debris_pixels > 0:
            debris_samples.append((i, debris_pixels, sample["id"]))
    
    print(f"   Found {len(debris_samples)} samples with debris in first 100")
    
    if len(debris_samples) == 0:
        print("\n   WARNING: No debris found in first 100 samples!")
        print("   Checking mask values...")
        sample = dataset[0]
        mask = sample["mask"].numpy()
        print(f"   Unique mask values: {np.unique(mask)}")
        return
    
    # Pick a sample with debris
    idx, debris_count, sample_id = debris_samples[0]
    print(f"\n3. Using sample {idx} ('{sample_id}') with {debris_count} debris pixels")
    
    sample = dataset[idx]
    image = sample["image"]
    mask = sample["mask"].numpy()
    
    print(f"   Image shape: {image.shape}")
    print(f"   Image range: [{image.min():.4f}, {image.max():.4f}]")
    print(f"   Mask unique values: {np.unique(mask)}")
    print(f"   Debris pixels (mask==1): {np.sum(mask == 1)}")
    print(f"   Background pixels (mask==0): {np.sum(mask == 0)}")
    
    # Load model
    print("\n4. Loading model...")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    
    # Get model config from checkpoint or detect from weights
    if "config" in checkpoint:
        model_config = checkpoint["config"].get("model", {})
    else:
        model_config = {"backbone": "mit_b2", "num_classes": 2}
    
    # Detect input channels
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    for key in state_dict:
        if "patch_embed1.proj.weight" in key:
            model_config["in_channels"] = state_dict[key].shape[1]
            break
    
    print(f"   Model config: {model_config}")
    
    model = load_model(model_path, model_config, device=device)
    model.eval()
    
    # Run prediction
    print("\n5. Running prediction...")
    
    # Normalize image (simple min-max for debugging)
    image_norm = image.float()
    if image_norm.max() > 1:
        image_norm = image_norm / 10000.0  # Sentinel-2 scaling
    
    image_batch = image_norm.unsqueeze(0).to(device)
    print(f"   Input shape: {image_batch.shape}")
    print(f"   Input range: [{image_batch.min():.4f}, {image_batch.max():.4f}]")
    
    with torch.no_grad():
        output = model(image_batch)
        
        if isinstance(output, dict):
            output = output.get("logits", list(output.values())[0])
        
        print(f"   Output shape: {output.shape}")
        print(f"   Output range: [{output.min():.4f}, {output.max():.4f}]")
        
        # Check logits for each class
        print(f"\n   Class 0 (background) logits: mean={output[0,0].mean():.4f}, std={output[0,0].std():.4f}")
        print(f"   Class 1 (debris) logits: mean={output[0,1].mean():.4f}, std={output[0,1].std():.4f}")
        
        # Get predictions
        probs = torch.softmax(output, dim=1)
        preds = output.argmax(dim=1).cpu().numpy()[0]
        
        print(f"\n   Prediction unique values: {np.unique(preds)}")
        print(f"   Predicted debris pixels: {np.sum(preds == 1)}")
        print(f"   Predicted background pixels: {np.sum(preds == 0)}")
        
        # Debris probability stats
        debris_prob = probs[0, 1].cpu().numpy()
        print(f"\n   Debris probability range: [{debris_prob.min():.4f}, {debris_prob.max():.4f}]")
        print(f"   Debris probability mean: {debris_prob.mean():.4f}")
        print(f"   Pixels with debris_prob > 0.5: {np.sum(debris_prob > 0.5)}")
        print(f"   Pixels with debris_prob > 0.3: {np.sum(debris_prob > 0.3)}")
        print(f"   Pixels with debris_prob > 0.1: {np.sum(debris_prob > 0.1)}")
    
    print("\n" + "=" * 60)
    print("DIAGNOSIS:")
    print("=" * 60)
    
    if output[0,1].mean() < output[0,0].mean() - 5:
        print("- Model heavily biased toward background")
        print("- Class 1 logits much lower than class 0")
        print("- Need stronger class weighting or different approach")
    
    if debris_prob.max() < 0.5:
        print(f"- Max debris probability is only {debris_prob.max():.4f}")
        print("- Model never confident about debris")
    
    if np.sum(preds == 1) == 0:
        print("- Model predicts ZERO debris pixels")
        print("- The class imbalance is too severe")

if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else "outputs/models/best_model.pth"
    debug_model(model_path)
