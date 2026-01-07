#!/usr/bin/env python3
"""
Run marine debris detection inference on satellite imagery.

Usage:
    python scripts/predict.py --input image.tif --model outputs/models/best_model.pth
    python scripts/predict.py --input-dir data/raw/ --output outputs/predictions/
    python scripts/predict.py --demo
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

from src.utils.device import get_device, print_device_info
from src.utils.config import load_config, get_default_config
from src.inference.predictor import MarineDebrisPredictor, batch_predict
from src.data.download import create_sample_data


def run_demo():
    """Run demo prediction on synthetic data."""
    print("\n" + "=" * 60)
    print("🌊 DEMO MODE - Marine Debris Detection")
    print("=" * 60)
    
    print("\nCreating synthetic sample data...")
    sample_dir = create_sample_data("data/sample")
    
    print("\nNote: This demo uses a randomly initialized model.")
    print("For real predictions, train a model first or use pretrained weights.")
    print("\nSample input created at: data/sample/sample_scene.tif")
    print("\nTo run real inference:")
    print("  1. Train: python scripts/train.py")
    print("  2. Predict: python scripts/predict.py --input <your_image.tif> --model outputs/models/best_model.pth")


def main():
    parser = argparse.ArgumentParser(
        description="Run marine debris detection on satellite imagery",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Input/Output
    parser.add_argument(
        "--input", "-i", type=str,
        help="Input GeoTIFF image path",
    )
    parser.add_argument(
        "--input-dir", type=str,
        help="Input directory for batch processing",
    )
    parser.add_argument(
        "--output", "-o", type=str, default="outputs/predictions",
        help="Output directory",
    )
    
    # Model
    parser.add_argument(
        "--model", "-m", type=str, default="outputs/models/best_model.pth",
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to configuration file",
    )
    
    # Inference settings
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Confidence threshold for detection",
    )
    parser.add_argument(
        "--tile-size", type=int, default=512,
        help="Tile size for sliding window inference",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device (auto, mps, cuda, cpu)",
    )
    
    # Output options
    parser.add_argument(
        "--visualize", action="store_true",
        help="Generate visualization images",
    )
    parser.add_argument(
        "--no-geojson", action="store_true",
        help="Skip GeoJSON output",
    )
    
    # Demo mode
    parser.add_argument(
        "--demo", action="store_true",
        help="Run demo with synthetic data",
    )
    
    args = parser.parse_args()
    
    # Demo mode
    if args.demo:
        run_demo()
        return
    
    # Validate inputs
    if not args.input and not args.input_dir:
        parser.error("Either --input or --input-dir is required (or use --demo)")
    
    print("\n" + "=" * 60)
    print("🌊 MARINE DEBRIS DETECTION - INFERENCE")
    print("=" * 60)
    
    # Print device info
    device = get_device(args.device)
    print(f"\nUsing device: {device}")
    
    # Check model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"\n[ERROR] Model not found: {model_path}")
        print("\nPlease train a model first:")
        print("  python scripts/train.py")
        print("\nOr specify a different model path:")
        print("  python scripts/predict.py --model /path/to/model.pth")
        sys.exit(1)
    
    # Load configuration
    if Path(args.config).exists():
        config = load_config(args.config)
    else:
        config = get_default_config()
    
    # Override inference settings
    config["inference"]["tile_size"] = args.tile_size
    config["inference"]["batch_size"] = args.batch_size
    config["inference"]["confidence_threshold"] = args.threshold
    
    print(f"\nModel: {model_path}")
    print(f"Confidence threshold: {args.threshold}")
    print(f"Tile size: {args.tile_size}")
    
    # Initialize predictor
    print("\nLoading model...")
    try:
        predictor = MarineDebrisPredictor(
            model_path=str(model_path),
            config=config,
            device=device,
        )
    except Exception as e:
        print(f"\n[ERROR] Failed to load model: {e}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Run inference
    if args.input:
        # Single image
        input_path = Path(args.input)
        
        if not input_path.exists():
            print(f"\n[ERROR] Input file not found: {input_path}")
            sys.exit(1)
        
        print(f"\nProcessing: {input_path.name}")
        print("-" * 40)
        
        try:
            results = predictor.predict(
                str(input_path),
                str(output_dir),
            )
            
            # Print summary
            n_hotspots = len(results["hotspots"])
            
            print("\n" + "=" * 60)
            print("✅ PREDICTION COMPLETE")
            print("=" * 60)
            print(f"\nDetected hotspots: {n_hotspots}")
            
            if n_hotspots > 0:
                print("\nTop 5 hotspots:")
                for _, row in results["hotspots"].head(5).iterrows():
                    print(f"  #{int(row['rank'])}: ({row['centroid_lat']:.4f}, {row['centroid_lon']:.4f}) "
                          f"Area: {row['area_m2']/1000:.1f} km² Conf: {row['confidence']:.2f}")
            
            print(f"\nOutputs saved to: {output_dir}")
            print(f"  - {input_path.stem}_heatmap.tif")
            print(f"  - {input_path.stem}_hotspots.geojson")
            print(f"  - {input_path.stem}_hotspots.csv")
            print(f"  - {input_path.stem}_mask.tif")
            
            # Visualization
            if args.visualize:
                print("\nGenerating visualization...")
                from src.utils.visualization import visualize_prediction
                from src.data.preprocessing import load_sentinel2_scene
                
                image, _ = load_sentinel2_scene(str(input_path), normalize=False)
                
                vis_path = output_dir / f"{input_path.stem}_visualization.png"
                visualize_prediction(
                    image=image,
                    probability_map=results["probability_map"],
                    output_path=str(vis_path),
                )
                print(f"  - {vis_path.name}")
        
        except Exception as e:
            print(f"\n[ERROR] Prediction failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    elif args.input_dir:
        # Batch processing
        input_dir = Path(args.input_dir)
        
        if not input_dir.exists():
            print(f"\n[ERROR] Input directory not found: {input_dir}")
            sys.exit(1)
        
        print(f"\nBatch processing: {input_dir}")
        print("-" * 40)
        
        results = batch_predict(
            predictor=predictor,
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            pattern="*.tif",
        )
        
        # Summary
        successful = sum(1 for r in results if r["status"] == "success")
        failed = sum(1 for r in results if r["status"] == "error")
        
        print("\n" + "=" * 60)
        print("✅ BATCH PROCESSING COMPLETE")
        print("=" * 60)
        print(f"\nProcessed: {len(results)} files")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"\nOutputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
