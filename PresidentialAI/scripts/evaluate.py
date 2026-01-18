#!/usr/bin/env python3
"""
Evaluate a trained marine debris detection model.

Usage:
    python scripts/evaluate.py --model outputs/models/best_model.pth
    python scripts/evaluate.py --model outputs/models/best_model.pth --data-dir data/marida
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from tqdm import tqdm
from rich.console import Console
from rich.table import Table

from src.utils.device import get_device
from src.utils.config import load_config, get_default_config
from src.models.segformer import load_model
from src.data.dataset import MaridaDataset
from src.training.metrics import MetricTracker, ConfusionMatrix


console = Console()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate marine debris detection model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--model", "-m", type=str, required=True,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/marida",
        help="Path to MARIDA dataset",
    )
    parser.add_argument(
        "--split", type=str, default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate",
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Batch size",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device (auto, mps, cuda, cpu)",
    )
    parser.add_argument(
        "--output", type=str,
        help="Path to save evaluation results",
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Generate visualization plots",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.3,
        help="Probability threshold for debris detection (default: 0.3)",
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("🌊 MARINE DEBRIS DETECTION - EVALUATION")
    print("=" * 60)
    
    # Device
    device = get_device(args.device)
    print(f"\nUsing device: {device}")
    
    # Check paths
    model_path = Path(args.model)
    data_dir = Path(args.data_dir)
    
    if not model_path.exists():
        print(f"\n[ERROR] Model not found: {model_path}")
        sys.exit(1)
    
    if not data_dir.exists():
        print(f"\n[ERROR] Data directory not found: {data_dir}")
        sys.exit(1)
    
    # Load configuration
    if Path(args.config).exists():
        config = load_config(args.config)
    else:
        config = get_default_config()
    
    # Data configuration
    data_config = config.get("data", {})
    bands = data_config.get("bands", ["B2", "B3", "B4", "B8", "B11", "B12"])
    normalization = data_config.get("normalization", {
        "mean": [0.0582, 0.0556, 0.0480, 0.1011, 0.1257, 0.0902],
        "std": [0.0276, 0.0267, 0.0308, 0.0522, 0.0560, 0.0479],
    })
    
    # Model configuration - detect in_channels from checkpoint
    model_config = config.get("model", {})
    
    # Load checkpoint to detect input channels
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    
    # Detect input channels from first conv layer
    for key in state_dict:
        if "patch_embed1.proj.weight" in key:
            model_config["in_channels"] = state_dict[key].shape[1]
            break
    else:
        model_config["in_channels"] = 11  # Default to 11 for MARIDA
    
    print(f"\nModel: {model_path}")
    print(f"Dataset: {data_dir}")
    print(f"Split: {args.split}")
    
    # Load model
    print("\nLoading model...")
    model = load_model(str(model_path), model_config, device=device)
    model.eval()
    
    # Load dataset
    print("Loading dataset...")
    dataset = MaridaDataset(
        root_dir=str(data_dir),
        split=args.split,
        normalization=normalization,
        transform=None,
        target_channels=model_config["in_channels"],
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    print(f"Samples: {len(dataset)}")
    
    # Evaluation
    print("\nRunning evaluation...")
    
    # Get threshold from args
    threshold = args.threshold
    print(f"Using threshold: {threshold}")
    
    metric_tracker = MetricTracker(num_classes=model_config.get("num_classes", 2))
    confusion_matrix = ConfusionMatrix(num_classes=model_config.get("num_classes", 2))
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Use probability threshold instead of argmax
            probs = torch.softmax(outputs, dim=1)
            debris_prob = probs[:, 1]  # Probability of debris class
            predictions = (debris_prob > threshold).long()
            
            # Update metrics
            metric_tracker.update(outputs, masks)
            confusion_matrix.update(
                predictions.cpu().numpy(),
                masks.cpu().numpy()
            )
            
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(masks.cpu().numpy())
    
    # Compute final metrics
    metrics = metric_tracker.compute()
    cm = confusion_matrix.compute()
    
    # Display results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    # Main metrics table
    table = Table(title="Segmentation Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    key_metrics = [
        ("IoU (Debris)", metrics.get("iou_debris", metrics.get("iou_class_1", 0))),
        ("IoU (Background)", metrics.get("iou_class_0", 0)),
        ("IoU (Mean)", metrics.get("iou_mean", 0)),
        ("Dice/F1 (Debris)", metrics.get("dice_debris", metrics.get("dice_class_1", 0))),
        ("Precision (Debris)", metrics.get("precision_debris", metrics.get("precision_class_1", 0))),
        ("Recall (Debris)", metrics.get("recall_debris", metrics.get("recall_class_1", 0))),
    ]
    
    for name, value in key_metrics:
        table.add_row(name, f"{value:.4f}")
    
    console.print(table)
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    print("                 Predicted")
    print("              Background  Debris")
    print(f"Actual Background  {cm[0, 0]:8d}  {cm[0, 1]:8d}")
    print(f"       Debris      {cm[1, 0]:8d}  {cm[1, 1]:8d}")
    
    # Normalized
    cm_norm = confusion_matrix.get_normalized()
    print("\nNormalized Confusion Matrix:")
    print(f"Actual Background  {cm_norm[0, 0]:8.2%}  {cm_norm[0, 1]:8.2%}")
    print(f"       Debris      {cm_norm[1, 0]:8.2%}  {cm_norm[1, 1]:8.2%}")
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        import json
        results = {
            "model": str(model_path),
            "dataset": str(data_dir),
            "split": args.split,
            "n_samples": len(dataset),
            "metrics": {k: float(v) for k, v in metrics.items()},
            "confusion_matrix": cm.tolist(),
        }
        
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
    
    # Visualization
    if args.visualize:
        print("\nGenerating visualizations...")
        
        from src.utils.visualization import plot_confusion_matrix
        
        output_dir = Path(args.output).parent if args.output else Path("outputs/evaluation")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Confusion matrix plot
        plot_confusion_matrix(
            cm,
            class_names=["Background", "Marine Debris"],
            output_path=str(output_dir / "confusion_matrix.png"),
        )
        print(f"  - {output_dir}/confusion_matrix.png")
    
    print("\n" + "=" * 60)
    print("✅ EVALUATION COMPLETE")
    print("=" * 60)
    
    # Return metrics for programmatic use
    return metrics


if __name__ == "__main__":
    main()