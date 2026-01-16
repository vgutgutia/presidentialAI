#!/usr/bin/env python3
"""Check MARIDA dataset labels to verify debris class."""

import sys
from pathlib import Path
import numpy as np

try:
    import rasterio
except ImportError:
    print("Installing rasterio...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "rasterio"])
    import rasterio

def check_labels(data_dir="data/marida"):
    data_dir = Path(data_dir)
    patches_dir = data_dir / "patches"
    
    if not patches_dir.exists():
        print(f"Error: {patches_dir} not found")
        return
    
    print("Scanning MARIDA labels...")
    print("-" * 50)
    
    all_classes = {}
    files_checked = 0
    debris_files = 0
    
    # Check label files
    for label_file in patches_dir.glob("**/*_cl.tif"):
        files_checked += 1
        
        with rasterio.open(label_file) as src:
            mask = src.read(1)
            unique_classes = np.unique(mask)
            
            for c in unique_classes:
                count = np.sum(mask == c)
                if c not in all_classes:
                    all_classes[c] = 0
                all_classes[c] += count
            
            # Check for potential debris classes (0, 1, or other low numbers)
            if 1 in unique_classes or 0 in unique_classes:
                if np.sum(mask == 1) > 0:
                    debris_files += 1
        
        if files_checked % 100 == 0:
            print(f"  Checked {files_checked} files...")
    
    print(f"\nTotal files checked: {files_checked}")
    print(f"Files with class 1 (debris?): {debris_files}")
    print("\nClass distribution:")
    print("-" * 50)
    
    total_pixels = sum(all_classes.values())
    for class_id in sorted(all_classes.keys()):
        count = all_classes[class_id]
        pct = 100 * count / total_pixels
        print(f"  Class {class_id:2d}: {count:12,d} pixels ({pct:6.3f}%)")
    
    print("-" * 50)
    print("\nMARIDA class mapping (from paper):")
    print("  1 = Marine Debris")
    print("  2 = Dense Sargassum")
    print("  3 = Sparse Sargassum")
    print("  4 = Natural Organic Material")
    print("  5 = Ship")
    print("  6 = Clouds")
    print("  7 = Marine Water")
    print("  8 = Sediment-Laden Water")
    print("  ... etc")
    
    print("\nIf class 1 has very few pixels, that's expected (debris is rare).")
    print("If class 1 has 0 pixels, there might be a labeling issue.")

if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/marida"
    check_labels(data_dir)
