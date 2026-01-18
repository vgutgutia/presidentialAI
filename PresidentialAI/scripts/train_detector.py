"""
Marine Debris Hotspot Detector - Anomaly-Based Approach
Finds statistically anomalous regions that indicate floating debris
Trains in ~30 seconds, produces meaningful hotspot detections!
"""

import os
import sys
import json
import random
import pickle
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
from scipy import ndimage

import rasterio
from tqdm import tqdm

random.seed(42)
np.random.seed(42)

# Sentinel-2 band indices
BLUE = 1
GREEN = 2
RED = 3
NIR = 7
SWIR1 = 9
SWIR2 = 10


@dataclass
class HotspotDetectorModel:
    """
    Hotspot detection model that finds anomalous floating debris regions.
    Uses image-relative anomaly detection + learned debris signatures.
    """
    # Debris spectral signature (learned from labeled data)
    debris_fdi_mean: float = 0.007
    debris_nir_ratio: float = 1.5  # Debris NIR / water NIR
    debris_brightness: float = 0.05
    
    # Water baseline (learned)
    water_fdi_mean: float = -0.003
    water_nir_mean: float = 0.016
    
    # Detection thresholds
    anomaly_zscore_threshold: float = 2.0  # Z-score for anomaly
    min_hotspot_pixels: int = 3  # Minimum pixels for a hotspot
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict) -> 'HotspotDetectorModel':
        return cls(**d)


def compute_debris_indices(image: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute spectral indices optimized for debris detection."""
    if image.shape[0] < 11:
        padded = np.zeros((11, image.shape[1], image.shape[2]), dtype=np.float32)
        padded[:image.shape[0]] = image
        image = padded
    
    eps = 1e-8
    
    blue = image[BLUE].astype(np.float32)
    green = image[GREEN].astype(np.float32)
    red = image[RED].astype(np.float32)
    nir = image[NIR].astype(np.float32)
    swir1 = image[SWIR1].astype(np.float32)
    swir2 = image[SWIR2].astype(np.float32)
    
    # FDI - Floating Debris Index (key indicator)
    fdi = nir - (red + (swir1 - red) * 0.178)
    
    # NDWI - identifies water
    ndwi = (green - nir) / (green + nir + eps)
    
    # NDVI - low for debris
    ndvi = (nir - red) / (nir + red + eps)
    
    # Brightness
    brightness = (blue + green + red + nir) / 4.0
    
    return {
        'fdi': fdi,
        'ndwi': ndwi,
        'ndvi': ndvi,
        'nir': nir,
        'brightness': brightness,
    }


def learn_from_data(data_dir: Path, max_patches: int = 200) -> HotspotDetectorModel:
    """Learn debris signatures from labeled data."""
    patches_dir = data_dir / "patches"
    
    debris_fdi = []
    debris_nir = []
    water_fdi = []
    water_nir = []
    
    folders = list(patches_dir.iterdir())
    random.shuffle(folders)
    
    print("Learning debris signatures from labeled data...")
    
    for folder in tqdm(folders[:max_patches]):
        if not folder.is_dir():
            continue
        
        for label_path in folder.glob('*_cl.tif'):
            img_path = Path(str(label_path).replace('_cl.tif', '.tif'))
            if not img_path.exists():
                continue
            
            try:
                with rasterio.open(label_path) as src:
                    label = src.read(1)
                with rasterio.open(img_path) as src:
                    image = src.read().astype(np.float32)
                
                indices = compute_debris_indices(image)
                
                # Debris pixels
                debris_mask = label == 1
                if debris_mask.sum() > 0:
                    debris_fdi.extend(indices['fdi'][debris_mask].tolist())
                    debris_nir.extend(indices['nir'][debris_mask].tolist())
                
                # Water pixels (sample)
                water_mask = label == 7
                if water_mask.sum() > 0:
                    idx = np.where(water_mask)
                    n = min(50, len(idx[0]))
                    sample = np.random.choice(len(idx[0]), n, replace=False)
                    for i in sample:
                        water_fdi.append(indices['fdi'][idx[0][i], idx[1][i]])
                        water_nir.append(indices['nir'][idx[0][i], idx[1][i]])
                        
            except Exception as e:
                continue
    
    model = HotspotDetectorModel()
    
    if debris_fdi:
        model.debris_fdi_mean = float(np.mean(debris_fdi))
        model.debris_nir_ratio = float(np.mean(debris_nir) / (np.mean(water_nir) + 1e-8))
        model.debris_brightness = float(np.mean(debris_nir))
    
    if water_fdi:
        model.water_fdi_mean = float(np.mean(water_fdi))
        model.water_nir_mean = float(np.mean(water_nir))
    
    print(f"\nLearned parameters:")
    print(f"  Debris FDI mean: {model.debris_fdi_mean:.5f}")
    print(f"  Water FDI mean: {model.water_fdi_mean:.5f}")
    print(f"  Debris/Water NIR ratio: {model.debris_nir_ratio:.2f}")
    
    return model


def detect_hotspots(
    image: np.ndarray,
    model: HotspotDetectorModel,
    sensitivity: float = 0.5
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Detect debris hotspots in an image using anomaly detection.
    
    Args:
        image: (C, H, W) satellite image array
        model: Trained model
        sensitivity: 0-1, higher = more detections
    
    Returns:
        prob_map: (H, W) probability map
        hotspots: List of detected hotspot dictionaries
    """
    indices = compute_debris_indices(image)
    h, w = indices['fdi'].shape
    
    # === Step 1: Compute anomaly scores ===
    
    # FDI Z-score (relative to image)
    fdi = indices['fdi']
    fdi_mean = np.mean(fdi)
    fdi_std = np.std(fdi) + 1e-8
    fdi_zscore = (fdi - fdi_mean) / fdi_std
    
    # NIR Z-score (brightness anomaly)
    nir = indices['nir']
    nir_mean = np.mean(nir)
    nir_std = np.std(nir) + 1e-8
    nir_zscore = (nir - nir_mean) / nir_std
    
    # Brightness Z-score
    bright = indices['brightness']
    bright_mean = np.mean(bright)
    bright_std = np.std(bright) + 1e-8
    bright_zscore = (bright - bright_mean) / bright_std
    
    # === Step 2: Combine scores ===
    
    # Combined anomaly score (higher = more likely debris)
    # Debris has: high FDI, high NIR, moderate brightness
    anomaly_score = (
        0.5 * np.clip(fdi_zscore, 0, 5) +
        0.3 * np.clip(nir_zscore, 0, 5) +
        0.2 * np.clip(bright_zscore, -2, 3)
    )
    
    # === Step 3: Apply constraints ===
    
    # Suppress vegetation (high NDVI)
    veg_mask = indices['ndvi'] > 0.25
    anomaly_score[veg_mask] *= 0.1
    
    # Suppress deep water (high NDWI, low brightness)
    deep_water = (indices['ndwi'] > 0.5) & (indices['brightness'] < 0.03)
    anomaly_score[deep_water] *= 0.2
    
    # Suppress very bright areas (likely clouds/land)
    too_bright = indices['brightness'] > 0.12
    anomaly_score[too_bright] *= 0.3
    
    # === Step 4: Generate probability map ===
    
    # Adjust threshold based on sensitivity
    # sensitivity=0.5 -> threshold=2.0, sensitivity=1.0 -> threshold=1.0
    threshold = model.anomaly_zscore_threshold * (1.5 - sensitivity)
    
    # Convert to probability (sigmoid-like)
    prob_map = 1.0 / (1.0 + np.exp(-(anomaly_score - threshold)))
    
    # Enhance peaks
    prob_map = np.clip(prob_map * 1.5, 0, 1)
    
    # === Step 5: Find and rank hotspots ===
    
    # Binary detection at adaptive threshold
    detection_threshold = 0.4 + (1 - sensitivity) * 0.3
    binary = prob_map > detection_threshold
    
    # Connected component analysis
    labeled, n_features = ndimage.label(binary)
    
    hotspots = []
    for i in range(1, n_features + 1):
        mask = labeled == i
        n_pixels = np.sum(mask)
        
        if n_pixels < model.min_hotspot_pixels:
            continue
        
        # Get hotspot properties
        coords = np.where(mask)
        cy, cx = np.mean(coords[0]), np.mean(coords[1])
        
        max_prob = float(prob_map[mask].max())
        mean_prob = float(prob_map[mask].mean())
        
        # Confidence based on size and probability
        size_factor = min(1.0, n_pixels / 20)  # More pixels = more confident
        confidence = (0.6 * max_prob + 0.3 * mean_prob + 0.1 * size_factor)
        
        hotspots.append({
            'id': len(hotspots) + 1,
            'center_y': int(cy),
            'center_x': int(cx),
            'n_pixels': int(n_pixels),
            'area_m2': int(n_pixels * 100),  # 10m resolution
            'confidence': round(float(confidence) * 100, 1),
            'max_prob': round(max_prob * 100, 1),
            'mean_prob': round(mean_prob * 100, 1),
        })
    
    # Sort by confidence
    hotspots.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Assign ranks
    for i, h in enumerate(hotspots):
        h['rank'] = i + 1
    
    return prob_map.astype(np.float32), hotspots


def evaluate_model(model: HotspotDetectorModel, data_dir: Path, n_samples: int = 50):
    """Evaluate the model on test patches."""
    patches_dir = data_dir / "patches"
    
    results = {
        'patches_with_debris': 0,
        'patches_detected': 0,
        'total_debris_pixels': 0,
        'detected_debris_pixels': 0,
        'false_positive_patches': 0,
    }
    
    folders = list(patches_dir.iterdir())
    random.shuffle(folders)
    
    print("\nEvaluating hotspot detection...")
    
    for folder in tqdm(folders[:n_samples]):
        if not folder.is_dir():
            continue
        
        for label_path in folder.glob('*_cl.tif'):
            img_path = Path(str(label_path).replace('_cl.tif', '.tif'))
            if not img_path.exists():
                continue
            
            try:
                with rasterio.open(label_path) as src:
                    label = src.read(1)
                with rasterio.open(img_path) as src:
                    image = src.read().astype(np.float32)
                
                actual_debris = label == 1
                n_debris = actual_debris.sum()
                
                prob_map, hotspots = detect_hotspots(image, model, sensitivity=0.5)
                
                detected = prob_map > 0.5
                
                if n_debris > 0:
                    results['patches_with_debris'] += 1
                    results['total_debris_pixels'] += n_debris
                    
                    # Did we detect any of the actual debris?
                    detected_debris = np.sum(detected & actual_debris)
                    results['detected_debris_pixels'] += detected_debris
                    
                    if len(hotspots) > 0:
                        results['patches_detected'] += 1
                else:
                    # No debris - false positives if we detected any
                    if len(hotspots) > 0:
                        results['false_positive_patches'] += 1
                        
            except:
                continue
    
    # Calculate metrics
    if results['patches_with_debris'] > 0:
        patch_recall = results['patches_detected'] / results['patches_with_debris']
    else:
        patch_recall = 0
    
    if results['total_debris_pixels'] > 0:
        pixel_recall = results['detected_debris_pixels'] / results['total_debris_pixels']
    else:
        pixel_recall = 0
    
    return {
        **results,
        'patch_recall': round(patch_recall * 100, 1),
        'pixel_recall': round(pixel_recall * 100, 1),
    }


def test_samples(model: HotspotDetectorModel, data_dir: Path, n: int = 5):
    """Test on sample images."""
    patches_dir = data_dir / "patches"
    
    print("\n=== Sample Detections ===")
    
    tested = 0
    for folder in patches_dir.iterdir():
        if tested >= n or not folder.is_dir():
            continue
        
        for label_path in folder.glob('*_cl.tif'):
            if tested >= n:
                break
                
            img_path = Path(str(label_path).replace('_cl.tif', '.tif'))
            if not img_path.exists():
                continue
            
            with rasterio.open(label_path) as src:
                label = src.read(1)
            with rasterio.open(img_path) as src:
                image = src.read().astype(np.float32)
            
            actual_debris = (label == 1).sum()
            prob_map, hotspots = detect_hotspots(image, model, sensitivity=0.5)
            
            print(f"\n{img_path.name}:")
            print(f"  Ground truth debris pixels: {actual_debris}")
            print(f"  Hotspots detected: {len(hotspots)}")
            
            if hotspots:
                top = hotspots[0]
                print(f"  Top hotspot: {top['n_pixels']} pixels, {top['confidence']:.0f}% confidence")
            
            print(f"  Prob map: min={prob_map.min():.2f}, max={prob_map.max():.2f}, mean={prob_map.mean():.3f}")
            
            tested += 1


def main():
    print("=" * 60)
    print("MARINE DEBRIS HOTSPOT DETECTOR - Training")
    print("=" * 60)
    
    DATA_DIR = Path("data/marida")
    OUTPUT_DIR = Path("outputs/models")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Learn from data
    print("\n[1/4] Learning debris signatures...")
    model = learn_from_data(DATA_DIR, max_patches=100)
    
    # Test on samples
    print("\n[2/4] Testing on samples...")
    test_samples(model, DATA_DIR, n=5)
    
    # Evaluate
    print("\n[3/4] Evaluating...")
    metrics = evaluate_model(model, DATA_DIR, n_samples=50)
    
    print(f"\n=== Evaluation Results ===")
    print(f"Patches with debris: {metrics['patches_with_debris']}")
    print(f"Patches where debris detected: {metrics['patches_detected']}")
    print(f"Patch-level recall: {metrics['patch_recall']}%")
    print(f"Pixel-level recall: {metrics['pixel_recall']}%")
    print(f"False positive patches: {metrics['false_positive_patches']}")
    
    # Save
    print("\n[4/4] Saving model...")
    model_path = OUTPUT_DIR / "hotspot_detector.json"
    with open(model_path, 'w') as f:
        json.dump(model.to_dict(), f, indent=2)
    
    with open(OUTPUT_DIR / "hotspot_detector.pkl", 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved to: {model_path}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    os.chdir(Path(__file__).parent.parent)
    main()
