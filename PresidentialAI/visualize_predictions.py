#!/usr/bin/env python3
"""Visualize model predictions to understand what it's detecting."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
import matplotlib.pyplot as plt

from src.models.segformer import load_model
from src.data.dataset import MaridaDataset

def visualize_predictions(model_path="outputs/models/best_model.pth", data_dir="data/marida", num_samples=5):
    print("Loading model and data...")
    
    # Device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    # Load dataset (test split, no oversampling)
    dataset = MaridaDataset(
        root_dir=data_dir,
        split="test",
        transform=None,
        normalization=None,
        target_channels=11,
        oversample_debris=False,
    )
    
    # Find samples WITH debris
    debris_indices = []
    for i in range(len(dataset)):
        sample = dataset[i]
        if sample["mask"].sum() > 0:
            debris_indices.append(i)
        if len(debris_indices) >= num_samples:
            break
    
    print(f"Found {len(debris_indices)} samples with debris")
    
    if len(debris_indices) == 0:
        print("No debris samples found in test set!")
        return
    
    # Load model
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    
    model_config = {"backbone": "mit_b2", "num_classes": 2}
    for key in state_dict:
        if "patch_embed1.proj.weight" in key:
            model_config["in_channels"] = state_dict[key].shape[1]
            break
    
    model = load_model(model_path, model_config, device=device)
    model.eval()
    
    # Create visualization
    fig, axes = plt.subplots(len(debris_indices), 4, figsize=(16, 4*len(debris_indices)))
    if len(debris_indices) == 1:
        axes = axes.reshape(1, -1)
    
    for row, idx in enumerate(debris_indices):
        sample = dataset[idx]
        image = sample["image"]
        mask = sample["mask"].numpy()
        
        # Normalize for model
        image_norm = image.float()
        if image_norm.max() > 1:
            image_norm = image_norm / 10000.0
        
        # Run prediction
        with torch.no_grad():
            output = model(image_norm.unsqueeze(0).to(device))
            if isinstance(output, dict):
                output = output.get("logits", list(output.values())[0])
            
            probs = torch.softmax(output, dim=1)
            debris_prob = probs[0, 1].cpu().numpy()
        
        # Create RGB visualization (bands 3,2,1 = R,G,B for Sentinel-2)
        rgb = image.numpy()[[2, 1, 0], :, :]  # B4, B3, B2
        rgb = np.transpose(rgb, (1, 2, 0))
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
        rgb = np.clip(rgb * 3, 0, 1)  # Brighten
        
        # Plot
        axes[row, 0].imshow(rgb)
        axes[row, 0].set_title(f"RGB Image\n{sample['id']}")
        axes[row, 0].axis('off')
        
        axes[row, 1].imshow(mask, cmap='Reds', vmin=0, vmax=1)
        axes[row, 1].set_title(f"Ground Truth\n({mask.sum()} debris pixels)")
        axes[row, 1].axis('off')
        
        im = axes[row, 2].imshow(debris_prob, cmap='hot', vmin=0, vmax=1)
        axes[row, 2].set_title(f"Debris Probability\n(max: {debris_prob.max():.3f})")
        axes[row, 2].axis('off')
        plt.colorbar(im, ax=axes[row, 2], fraction=0.046)
        
        # Threshold comparison
        pred_03 = (debris_prob > 0.3).astype(np.uint8)
        overlay = np.zeros((*mask.shape, 3))
        overlay[mask == 1] = [0, 1, 0]  # Green = ground truth
        overlay[pred_03 == 1] = [1, 0, 0]  # Red = prediction
        overlay[(mask == 1) & (pred_03 == 1)] = [1, 1, 0]  # Yellow = correct
        
        axes[row, 3].imshow(overlay)
        tp = ((mask == 1) & (pred_03 == 1)).sum()
        fp = ((mask == 0) & (pred_03 == 1)).sum()
        fn = ((mask == 1) & (pred_03 == 0)).sum()
        axes[row, 3].set_title(f"Overlay (thresh=0.3)\nGreen=GT, Red=Pred, Yellow=TP\nTP:{tp}, FP:{fp}, FN:{fn}")
        axes[row, 3].axis('off')
    
    plt.tight_layout()
    
    # Save
    output_path = "outputs/prediction_visualization.png"
    Path("outputs").mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to: {output_path}")
    plt.close()
    
    # Print summary
    print("\nSummary:")
    print("- Green pixels = actual debris (ground truth)")
    print("- Red pixels = model predictions")  
    print("- Yellow pixels = correct detections (true positives)")
    print("\nIf red and green don't overlap, the model is detecting the wrong features.")

if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else "outputs/models/best_model.pth"
    visualize_predictions(model_path)
