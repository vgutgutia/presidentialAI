#!/usr/bin/env python3
"""
Download the MARIDA dataset for marine debris detection training.

The MARIDA dataset can be obtained from:
1. GitHub: https://github.com/marine-debris/marine-debris.github.io
2. Zenodo: https://zenodo.org/record/5151941

Usage:
    python scripts/download_marida.py
    python scripts/download_marida.py --output-dir data/marida
    python scripts/download_marida.py --verify
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_marida_structure(data_dir: Path) -> dict:
    """
    Check and report on MARIDA dataset structure.
    
    Returns dict with structure info and any issues found.
    """
    data_dir = Path(data_dir)
    
    result = {
        "valid": False,
        "structure": None,
        "issues": [],
        "patches_dir": None,
        "labels_source": None,
    }
    
    if not data_dir.exists():
        result["issues"].append(f"Directory does not exist: {data_dir}")
        return result
    
    # Check for patches directory
    patches_dir = data_dir / "patches"
    if not patches_dir.exists():
        result["issues"].append("'patches' directory not found")
    else:
        result["patches_dir"] = patches_dir
        
        # Count patch files - MARIDA has nested structure: patches/S2_DATE/files.tif
        tif_files = list(patches_dir.glob("**/*.tif"))
        result["num_patches"] = len(tif_files)
        
        if len(tif_files) == 0:
            result["issues"].append("No .tif files found in patches directory")
        else:
            # Check for label files (ending in _cl.tif)
            label_files = [f for f in tif_files if "_cl.tif" in f.name]
            image_files = [f for f in tif_files if "_cl.tif" not in f.name]
            result["num_images"] = len(image_files)
            result["num_labels"] = len(label_files)
            
            if len(label_files) > 0:
                result["labels_source"] = "cl_suffix"
    
    # Check for splits directory
    splits_dir = data_dir / "splits"
    if splits_dir.exists():
        result["has_splits"] = True
        for split in ["train", "val", "test"]:
            split_file = splits_dir / f"{split}.txt"
            if split_file.exists():
                with open(split_file) as f:
                    lines = [l.strip() for l in f.readlines() if l.strip()]
                    result[f"{split}_count"] = len(lines)
    else:
        result["has_splits"] = False
        result["issues"].append("'splits' directory not found - will create automatic splits")
    
    # Check for shapefiles directory (alternative label format)
    shapefiles_dir = data_dir / "shapefiles"
    if shapefiles_dir.exists():
        if result["labels_source"] is None:
            result["labels_source"] = "shapefiles"
        result["shapefiles_dir"] = shapefiles_dir
    
    # Check for labels_mapping.txt
    labels_file = data_dir / "labels_mapping.txt"
    if labels_file.exists():
        result["has_labels_mapping"] = True
    
    # Determine if valid
    has_images = result.get("num_images", 0) > 0
    has_labels = result.get("num_labels", 0) > 0 or result.get("labels_source") == "shapefiles"
    
    if has_images and has_labels:
        result["valid"] = True
        result["structure"] = "standard"
    elif has_images:
        result["valid"] = True
        result["structure"] = "images_only"
        result["issues"].append("No label files found - training will require labels")
    
    return result


def print_manual_instructions():
    """Print manual download instructions."""
    print("""
================================================================================
MANUAL DOWNLOAD INSTRUCTIONS
================================================================================

The MARIDA dataset must be downloaded manually from GitHub:

1. Go to: https://github.com/marine-debris/marine-debris.github.io

2. Download the repository:
   - Click "Code" -> "Download ZIP"
   - Or: git clone https://github.com/marine-debris/marine-debris.github.io.git

3. Copy the data folders to your project's data/marida/ directory:
   
   mkdir -p data/marida
   cp -r marine-debris.github.io/patches data/marida/
   cp -r marine-debris.github.io/shapefiles data/marida/
   cp -r marine-debris.github.io/splits data/marida/
   cp marine-debris.github.io/labels_mapping.txt data/marida/

4. Your data/marida/ directory should contain:
   
   data/marida/
   ├── labels_mapping.txt
   ├── patches/
   │   └── S2_DATE_ROI/
   │       ├── S2_DATE_ROI_X.tif      (image patches)
   │       └── S2_DATE_ROI_X_cl.tif   (label patches)
   ├── shapefiles/
   └── splits/
       ├── train.txt
       ├── val.txt
       └── test.txt

5. Verify with: python scripts/download_marida.py --verify

================================================================================
""")


def main():
    parser = argparse.ArgumentParser(
        description="Download/verify MARIDA dataset for marine debris detection"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/marida",
        help="Directory for dataset (default: data/marida)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify existing dataset structure",
    )
    parser.add_argument(
        "--sample-only",
        action="store_true",
        help="Create sample synthetic data only (for testing)",
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("MARIDA Dataset Setup")
    print("Marine Debris Archive for Satellite Imagery")
    print("=" * 70)
    
    output_dir = Path(args.output_dir)
    
    # Sample data mode
    if args.sample_only:
        print("\nCreating sample synthetic data for testing...")
        from src.data.download import create_sample_data
        sample_dir = create_sample_data("data/sample")
        print(f"\nSample data created at: {sample_dir}")
        print("You can now test the pipeline with synthetic data.")
        print("\nTo use real data, download MARIDA manually (see instructions below).")
        print_manual_instructions()
        return
    
    # Check if data already exists
    print(f"\nChecking dataset at: {output_dir}")
    
    if output_dir.exists():
        result = check_marida_structure(output_dir)
        
        print("\nDataset Analysis:")
        print("-" * 40)
        
        if result["valid"]:
            print("[OK] Valid MARIDA dataset found")
            print(f"  Structure: {result.get('structure', 'unknown')}")
            print(f"  Image patches: {result.get('num_images', 'unknown')}")
            print(f"  Label patches: {result.get('num_labels', 'unknown')}")
            print(f"  Labels source: {result.get('labels_source', 'none')}")
            
            if result.get("has_splits"):
                print(f"  Train samples: {result.get('train_count', 'unknown')}")
                print(f"  Val samples: {result.get('val_count', 'unknown')}")
                print(f"  Test samples: {result.get('test_count', 'unknown')}")
            
            if result["issues"]:
                print("\nWarnings:")
                for issue in result["issues"]:
                    print(f"  - {issue}")
            
            print("\n" + "=" * 70)
            print("Dataset is ready for training!")
            print("=" * 70)
            print("\nNext steps:")
            print("  python scripts/train.py")
            return
        else:
            print("[INCOMPLETE] Dataset found but has issues:")
            for issue in result["issues"]:
                print(f"  - {issue}")
    else:
        print(f"[NOT FOUND] No dataset at {output_dir}")
    
    # Show manual download instructions
    print_manual_instructions()
    
    # Create directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Citation info
    print("-" * 70)
    print("CITATION")
    print("-" * 70)
    print("""
If you use MARIDA in your research, please cite:

Kikaki, K., Kakogeorgiou, I., Mikeli, P., Raitsos, D.E., & Karantzalos, K. (2022).
MARIDA: A benchmark for Marine Debris detection from Sentinel-2 remote sensing data.
PLoS ONE 17(1): e0262247.
https://doi.org/10.1371/journal.pone.0262247
""")


if __name__ == "__main__":
    main()
