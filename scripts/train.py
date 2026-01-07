#!/usr/bin/env python3
"""
Train the marine debris detection model.

Optimized for Apple Silicon (M4 Max) with automatic MPS detection.

Usage:
    python scripts/train.py
    python scripts/train.py --epochs 50 --batch-size 8
    python scripts/train.py --config config.yaml --checkpoint outputs/models/checkpoint.pth
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import random

from src.utils.device import get_device, print_device_info
from src.utils.config import load_config, get_default_config, merge_configs
from src.models.segformer import create_model
from src.data.dataset import create_dataloaders, MaridaDataset
from src.training.trainer import Trainer, create_augmentations


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def main():
    parser = argparse.ArgumentParser(
        description="Train marine debris detection model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Config
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to configuration file",
    )
    
    # Training parameters (override config)
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--device", type=str, help="Device (auto, mps, cuda, cpu)")
    
    # Data
    parser.add_argument(
        "--data-dir", type=str, default="data/marida",
        help="Path to MARIDA dataset",
    )
    parser.add_argument("--num-workers", type=int, default=4, help="Data loader workers")
    
    # Checkpointing
    parser.add_argument(
        "--checkpoint", type=str,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs",
        help="Output directory for models and logs",
    )
    
    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--debug", action="store_true", help="Debug mode (small dataset)")
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "=" * 60)
    print("🌊 MARINE DEBRIS DETECTION - TRAINING")
    print("=" * 60)
    
    # Print device info
    print_device_info()
    
    # Load configuration
    if Path(args.config).exists():
        config = load_config(args.config)
        print(f"Loaded config from: {args.config}")
    else:
        config = get_default_config()
        print("Using default configuration")
    
    # Override with command line arguments
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.lr:
        config["training"]["learning_rate"] = args.lr
    if args.device:
        config["device"] = args.device
    
    # Set seed
    seed = args.seed or config.get("project", {}).get("seed", 42)
    set_seed(seed)
    print(f"Random seed: {seed}")
    
    # Get device
    device = get_device(config.get("device", "auto"))
    print(f"Using device: {device}")
    
    # Check for data
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"\n[ERROR] Data directory not found: {data_dir}")
        print("\nPlease download the MARIDA dataset first:")
        print("  python scripts/download_marida.py")
        print("\nOr create sample data for testing:")
        print("  python scripts/download_marida.py --sample-only")
        sys.exit(1)
    
    # Create output directories
    output_dir = Path(args.output_dir)
    models_dir = output_dir / "models"
    logs_dir = output_dir / "logs"
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Data configuration
    data_config = config.get("data", {})
    bands = data_config.get("bands", ["B2", "B3", "B4", "B8", "B11", "B12"])
    normalization = data_config.get("normalization", {
        "mean": [0.0582, 0.0556, 0.0480, 0.1011, 0.1257, 0.0902],
        "std": [0.0276, 0.0267, 0.0308, 0.0522, 0.0560, 0.0479],
    })
    
    # Create augmentations
    aug_config = config.get("training", {}).get("augmentation", {})
    transform_train = create_augmentations(aug_config)
    transform_val = None  # No augmentation for validation
    
    print("\n" + "-" * 60)
    print("LOADING DATA")
    print("-" * 60)
    
    # Create dataloaders
    batch_size = config["training"].get("batch_size", 8)
    num_workers = args.num_workers
    
    # On macOS, use fewer workers to avoid issues
    if sys.platform == "darwin":
        num_workers = min(num_workers, 4)
    
    print(f"Batch size: {batch_size}")
    print(f"Num workers: {num_workers}")
    
    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            data_dir=str(data_dir),
            batch_size=batch_size,
            num_workers=num_workers,
            bands=bands,
            normalization=normalization,
            transform_train=transform_train,
            transform_val=transform_val,
        )
        
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print(f"Test samples: {len(test_loader.dataset)}")
        
    except Exception as e:
        print(f"\n[ERROR] Failed to load data: {e}")
        print("\nMake sure the MARIDA dataset is properly downloaded and extracted.")
        print("Expected structure:")
        print("  data/marida/patches/  (or scenes/)")
        print("  data/marida/masks/    (or labels/)")
        sys.exit(1)
    
    # Debug mode - use small subset
    if args.debug:
        print("\n[DEBUG MODE] Using small data subset")
        train_loader.dataset.samples = train_loader.dataset.samples[:20]
        val_loader.dataset.samples = val_loader.dataset.samples[:10]
    
    print("\n" + "-" * 60)
    print("CREATING MODEL")
    print("-" * 60)
    
    # Model configuration
    model_config = config.get("model", {})
    model_config["in_channels"] = len(bands)
    
    print(f"Backbone: {model_config.get('backbone', 'mit_b2')}")
    print(f"Input channels: {model_config['in_channels']}")
    print(f"Num classes: {model_config.get('num_classes', 2)}")
    
    # Create model
    model = create_model(model_config, device=device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("\n" + "-" * 60)
    print("TRAINING")
    print("-" * 60)
    
    # Training configuration
    train_config = config.get("training", {})
    train_config["num_classes"] = model_config.get("num_classes", 2)
    
    print(f"Epochs: {train_config.get('epochs', 100)}")
    print(f"Learning rate: {train_config.get('learning_rate', 1e-4)}")
    print(f"Optimizer: {train_config.get('optimizer', 'adamw')}")
    print(f"Scheduler: {train_config.get('scheduler', 'cosine')}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        config=train_config,
        device=device,
        output_dir=str(output_dir),
    )
    
    # Resume from checkpoint if specified
    if args.checkpoint:
        print(f"\nResuming from checkpoint: {args.checkpoint}")
        trainer.load_checkpoint(args.checkpoint)
    
    # Train
    print("\nStarting training...\n")
    
    try:
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=train_config.get("epochs", 100),
        )
        
        print("\n" + "=" * 60)
        print("✅ TRAINING COMPLETE")
        print("=" * 60)
        print(f"\nBest validation IoU: {trainer.best_metric:.4f}")
        print(f"\nSaved models:")
        print(f"  - {models_dir}/best_model.pth")
        print(f"  - {models_dir}/final_model.pth")
        print(f"\nTensorBoard logs: {logs_dir}/tensorboard")
        print("\nNext steps:")
        print("  1. Evaluate: python scripts/evaluate.py --model outputs/models/best_model.pth")
        print("  2. Predict:  python scripts/predict.py --model outputs/models/best_model.pth --input <image>")
        
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Training stopped by user")
        print("Checkpoint saved to: outputs/models/final_model.pth")
    
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
