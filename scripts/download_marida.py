#!/usr/bin/env python3
"""
Download the MARIDA dataset for marine debris detection training.

Usage:
    python scripts/download_marida.py
    python scripts/download_marida.py --output-dir data/marida
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.download import download_marida, create_sample_data


def main():
    parser = argparse.ArgumentParser(
        description="Download MARIDA dataset for marine debris detection"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/marida",
        help="Directory to save dataset (default: data/marida)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if data exists",
    )
    parser.add_argument(
        "--sample-only",
        action="store_true",
        help="Create sample synthetic data only (for testing)",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("MARIDA Dataset Downloader")
    print("Marine Debris Archive for Satellite Imagery")
    print("=" * 60)
    
    if args.sample_only:
        print("\nCreating sample synthetic data for testing...")
        create_sample_data("data/sample")
        print("\nSample data created! You can now test the pipeline.")
        print("For real training, run without --sample-only flag.")
        return
    
    print(f"\nOutput directory: {args.output_dir}")
    print("\nThis will download approximately 2GB of data.")
    print("The MARIDA dataset is hosted on Zenodo.")
    print("\nCitation:")
    print("  Kikaki et al. (2022). 'MARIDA: A benchmark for Marine Debris")
    print("  detection from Sentinel-2 remote sensing data'")
    print("  Scientific Data, Nature.")
    print()
    
    try:
        download_marida(args.output_dir, force_download=args.force)
        
        print("\n" + "=" * 60)
        print("Download complete!")
        print("=" * 60)
        print(f"\nDataset location: {args.output_dir}")
        print("\nNext steps:")
        print("  1. Verify the data: ls -la data/marida/")
        print("  2. Start training: python scripts/train.py")
        
    except Exception as e:
        print(f"\n[ERROR] Download failed: {e}")
        print("\nManual download instructions:")
        print("  1. Visit: https://zenodo.org/record/5151941")
        print("  2. Download: patches.zip, masks.zip, splits.zip")
        print(f"  3. Extract all files to: {args.output_dir}")
        print("\nAlternatively, clone from GitHub:")
        print("  https://github.com/marine-debris/marine-debris.github.io")
        sys.exit(1)


if __name__ == "__main__":
    main()
