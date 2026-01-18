"""
Fast Marine Debris Detection using Random Forest + Spectral Indices
Trains in 2-3 minutes, works reliably!
"""

import os
import sys
import random
import pickle
import numpy as np
from pathlib import Path
from collections import Counter

import rasterio
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import joblib

# Seed for reproducibility
random.seed(42)
np.random.seed(42)

# =============================================================================
# SENTINEL-2 BAND INDICES (MARIDA uses 11 bands)
# =============================================================================
# Band order in MARIDA:
# 0: B1  - Coastal aerosol
# 1: B2  - Blue
# 2: B3  - Green  
# 3: B4  - Red
# 4: B5  - Red Edge 1
# 5: B6  - Red Edge 2
# 6: B7  - Red Edge 3
# 7: B8  - NIR
# 8: B8A - Red Edge 4
# 9: B11 - SWIR 1
# 10: B12 - SWIR 2

BLUE = 1
GREEN = 2
RED = 3
RED_EDGE1 = 4
RED_EDGE2 = 5
RED_EDGE3 = 6
NIR = 7
RED_EDGE4 = 8
SWIR1 = 9
SWIR2 = 10


def safe_divide(a, b, eps=1e-8):
    """Safe division to avoid NaN"""
    return np.divide(a, b + eps, where=(np.abs(b) > eps), out=np.zeros_like(a, dtype=np.float32))


def compute_spectral_indices(image):
    """
    Compute spectral indices that highlight marine debris.
    
    Args:
        image: numpy array of shape (11, H, W) with Sentinel-2 bands
    
    Returns:
        features: numpy array of shape (N_features, H, W)
    """
    # Ensure float32
    image = image.astype(np.float32)
    
    # Extract bands
    blue = image[BLUE]
    green = image[GREEN]
    red = image[RED]
    re1 = image[RED_EDGE1]
    re2 = image[RED_EDGE2]
    re3 = image[RED_EDGE3]
    nir = image[NIR]
    re4 = image[RED_EDGE4]
    swir1 = image[SWIR1]
    swir2 = image[SWIR2]
    
    features = []
    
    # 1. NDWI - Normalized Difference Water Index (highlights water)
    ndwi = safe_divide(green - nir, green + nir)
    features.append(ndwi)
    
    # 2. NDVI - Normalized Difference Vegetation Index (catches Sargassum)
    ndvi = safe_divide(nir - red, nir + red)
    features.append(ndvi)
    
    # 3. FAI - Floating Algae Index (from MARIDA paper)
    # FAI = NIR - (Red + (SWIR1 - Red) * (NIR_wavelength - Red_wavelength) / (SWIR1_wavelength - Red_wavelength))
    # Simplified: FAI ≈ NIR - Red - 0.5 * (SWIR1 - Red)
    fai = nir - red - 0.5 * (swir1 - red)
    features.append(fai)
    
    # 4. FDI - Floating Debris Index
    # FDI = NIR - (RE1 + (SWIR1 - RE1) * (833 - 704) / (1614 - 704))
    # Simplified approximation
    fdi = nir - (re1 + (swir1 - re1) * 0.14)
    features.append(fdi)
    
    # 5. NDMI - Normalized Difference Moisture Index
    ndmi = safe_divide(nir - swir1, nir + swir1)
    features.append(ndmi)
    
    # 6. MNDWI - Modified NDWI (better for turbid water)
    mndwi = safe_divide(green - swir1, green + swir1)
    features.append(mndwi)
    
    # 7. Plastic Index (PI) - based on SWIR absorption
    pi = safe_divide(swir1, swir2)
    features.append(pi)
    
    # 8. Red Edge NDVI variants (good for floating material)
    re_ndvi1 = safe_divide(re3 - red, re3 + red)
    features.append(re_ndvi1)
    
    re_ndvi2 = safe_divide(nir - re1, nir + re1)
    features.append(re_ndvi2)
    
    # 9. Band ratios
    blue_green = safe_divide(blue, green)
    features.append(blue_green)
    
    green_red = safe_divide(green, red)
    features.append(green_red)
    
    nir_red = safe_divide(nir, red)
    features.append(nir_red)
    
    swir_nir = safe_divide(swir1, nir)
    features.append(swir_nir)
    
    # 10. Raw bands (normalized)
    for i in range(image.shape[0]):
        # Normalize each band to 0-1 range
        band = image[i]
        band_norm = (band - band.min()) / (band.max() - band.min() + 1e-8)
        features.append(band_norm)
    
    # Stack all features
    features = np.stack(features, axis=0)
    
    # Replace any NaN/Inf with 0
    features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
    
    return features


def extract_pixels_from_patch(image_path, label_path, max_pixels_per_class=1000):
    """
    Extract pixel samples from a patch.
    
    Returns:
        X: feature vectors (N, F)
        y: labels (N,)
    """
    # Load image and label
    with rasterio.open(image_path) as src:
        image = src.read().astype(np.float32)
    
    with rasterio.open(label_path) as src:
        label = src.read(1).astype(np.int32)
    
    # Compute features
    features = compute_spectral_indices(image)
    
    # Reshape to (H*W, F)
    n_features, h, w = features.shape
    X = features.reshape(n_features, -1).T  # (H*W, F)
    y = label.flatten()  # (H*W,)
    
    # MARIDA classes:
    # 1: Marine Debris, 2: Dense Sargassum, 3: Sparse Sargassum, 4: Natural Organic Material
    # 5: Ship, 6: Clouds, 7: Marine Water, 8: Sediment-Laden Water, 9: Foam
    # 10: Turbid Water, 11: Shallow Water, 12: Waves, 13: Cloud Shadows, 14: Wakes, 15: Mixed Water
    
    # Binary: debris-related (1-4) vs background
    debris_classes = {1, 2, 3, 4}
    y_binary = np.isin(y, list(debris_classes)).astype(np.int32)
    
    # Sample pixels to balance classes
    debris_idx = np.where(y_binary == 1)[0]
    background_idx = np.where(y_binary == 0)[0]
    
    # Take all debris pixels, sample background
    n_debris = len(debris_idx)
    n_background = min(len(background_idx), max(n_debris * 2, max_pixels_per_class))
    
    if n_debris > 0:
        sampled_debris = debris_idx
    else:
        sampled_debris = np.array([], dtype=np.int32)
    
    if n_background > 0:
        sampled_background = np.random.choice(background_idx, size=min(n_background, len(background_idx)), replace=False)
    else:
        sampled_background = np.array([], dtype=np.int32)
    
    # Combine
    if len(sampled_debris) > 0 or len(sampled_background) > 0:
        all_idx = np.concatenate([sampled_debris, sampled_background])
        return X[all_idx], y_binary[all_idx]
    else:
        return np.array([]).reshape(0, n_features), np.array([])


def load_training_data(data_dir, max_patches=300):
    """Load and prepare training data from MARIDA patches."""
    patches_dir = Path(data_dir) / "patches"
    
    all_X = []
    all_y = []
    
    # Find all patches
    patch_folders = [f for f in patches_dir.iterdir() if f.is_dir()]
    random.shuffle(patch_folders)
    patch_folders = patch_folders[:max_patches]
    
    print(f"Loading data from {len(patch_folders)} patches...")
    
    for folder in tqdm(patch_folders):
        # Find image and label files
        image_files = [f for f in folder.glob('*.tif') 
                       if '_cl' not in f.name and '_conf' not in f.name]
        
        for image_path in image_files:
            label_path = folder / image_path.name.replace('.tif', '_cl.tif')
            
            if not label_path.exists():
                continue
            
            try:
                X, y = extract_pixels_from_patch(image_path, label_path)
                if len(X) > 0:
                    all_X.append(X)
                    all_y.append(y)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
    
    # Combine all data
    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    
    return X, y


def train_model(X, y):
    """Train Random Forest classifier."""
    print(f"\nTraining data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution: {Counter(y)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Train Random Forest with class weighting
    print("\nTraining Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',  # Handle class imbalance
        n_jobs=-1,  # Use all cores
        random_state=42,
        verbose=1,
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    print("\nEvaluating...")
    y_pred = model.predict(X_test)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Background', 'Debris']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Feature importance
    print("\nTop 10 Feature Importances:")
    feature_names = [
        'NDWI', 'NDVI', 'FAI', 'FDI', 'NDMI', 'MNDWI', 'PI', 
        'RE_NDVI1', 'RE_NDVI2', 'Blue/Green', 'Green/Red', 'NIR/Red', 'SWIR/NIR',
        'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'
    ]
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    for i, idx in enumerate(indices):
        name = feature_names[idx] if idx < len(feature_names) else f'Feature_{idx}'
        print(f"  {i+1}. {name}: {importances[idx]:.4f}")
    
    return model


def main():
    print("=" * 60)
    print("MARINE DEBRIS DETECTION - Random Forest with Spectral Indices")
    print("=" * 60)
    
    # Paths
    DATA_DIR = Path("data/marida")
    OUTPUT_DIR = Path("outputs/models")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    X, y = load_training_data(DATA_DIR, max_patches=300)
    
    if len(X) == 0:
        print("ERROR: No training data found!")
        return
    
    # Train model
    model = train_model(X, y)
    
    # Save model
    model_path = OUTPUT_DIR / "debris_rf_model.joblib"
    joblib.dump(model, model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Also save as pickle for compatibility
    pickle_path = OUTPUT_DIR / "debris_rf_model.pkl"
    with open(pickle_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model also saved to: {pickle_path}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    os.chdir(Path(__file__).parent.parent)
    main()

