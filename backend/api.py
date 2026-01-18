"""
OceanGuard AI - Marine Debris Detection API
Uses spectral anomaly detection for hotspot identification
"""

import io
import base64
import json
import random
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, asdict

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

# Try to import ML libraries
try:
    import rasterio
    from rasterio.io import MemoryFile
    from scipy import ndimage
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("Warning: ML libraries not available")

# Paths
BACKEND_DIR = Path(__file__).parent
PROJECT_ROOT = BACKEND_DIR.parent
PRESIDENTIAL_AI_PATH = PROJECT_ROOT / "PresidentialAI"
MODEL_PATH = PRESIDENTIAL_AI_PATH / "outputs" / "models" / "hotspot_detector.json"

# =============================================================================
# MODEL DEFINITION (must match training)
# =============================================================================
BLUE = 1
GREEN = 2
RED = 3
NIR = 7
SWIR1 = 9
SWIR2 = 10


def to_python_types(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_python_types(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj


@dataclass
class HotspotDetectorModel:
    debris_fdi_mean: float = 0.007
    debris_nir_ratio: float = 1.5
    debris_brightness: float = 0.05
    water_fdi_mean: float = -0.003
    water_nir_mean: float = 0.016
    anomaly_zscore_threshold: float = 2.0
    min_hotspot_pixels: int = 3
    
    @classmethod
    def from_dict(cls, d: dict) -> 'HotspotDetectorModel':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def compute_debris_indices(image: np.ndarray) -> dict:
    """Compute spectral indices for debris detection."""
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
    
    fdi = nir - (red + (swir1 - red) * 0.178)
    ndwi = (green - nir) / (green + nir + eps)
    ndvi = (nir - red) / (nir + red + eps)
    brightness = (blue + green + red + nir) / 4.0
    
    return {
        'fdi': fdi,
        'ndwi': ndwi,
        'ndvi': ndvi,
        'nir': nir,
        'brightness': brightness,
    }


def detect_hotspots(image: np.ndarray, model: HotspotDetectorModel, sensitivity: float = 0.5):
    """
    Detect debris hotspots using anomaly detection.
    Returns probability map and list of hotspots.
    """
    indices = compute_debris_indices(image)
    h, w = indices['fdi'].shape
    
    # Compute Z-scores (anomalies relative to image)
    fdi = indices['fdi']
    fdi_mean, fdi_std = np.mean(fdi), np.std(fdi) + 1e-8
    fdi_zscore = (fdi - fdi_mean) / fdi_std
    
    nir = indices['nir']
    nir_mean, nir_std = np.mean(nir), np.std(nir) + 1e-8
    nir_zscore = (nir - nir_mean) / nir_std
    
    bright = indices['brightness']
    bright_mean, bright_std = np.mean(bright), np.std(bright) + 1e-8
    bright_zscore = (bright - bright_mean) / bright_std
    
    # Debug logging
    print(f"  FDI: mean={fdi_mean:.4f}, std={fdi_std:.4f}, max_z={fdi_zscore.max():.2f}")
    print(f"  NIR: mean={nir_mean:.4f}, range=[{nir.min():.4f}, {nir.max():.4f}]")
    print(f"  Brightness: mean={bright_mean:.4f}, max={bright.max():.4f}")
    
    # Combined anomaly score - debris appears as bright anomalies in FDI and NIR
    anomaly_score = (
        0.5 * np.clip(fdi_zscore, 0, 5) +
        0.3 * np.clip(nir_zscore, 0, 5) +
        0.2 * np.clip(bright_zscore, -2, 3)
    )
    
    print(f"  Anomaly score before suppression: max={anomaly_score.max():.2f}")
    
    # Suppress non-debris areas (less aggressive now)
    veg_mask = indices['ndvi'] > 0.35  # Was 0.25
    anomaly_score[veg_mask] *= 0.2  # Was 0.1
    
    deep_water = (indices['ndwi'] > 0.6) & (indices['brightness'] < 0.02)  # More strict
    anomaly_score[deep_water] *= 0.3  # Was 0.2
    
    too_bright = indices['brightness'] > 0.2  # Was 0.12
    anomaly_score[too_bright] *= 0.5  # Was 0.3
    
    print(f"  Anomaly score after suppression: max={anomaly_score.max():.2f}")
    
    # Apply spatial smoothing to group nearby anomalies
    from scipy.ndimage import gaussian_filter
    anomaly_score = gaussian_filter(anomaly_score, sigma=1.5)
    
    # Convert to probability - balanced threshold
    # Higher sensitivity = lower threshold = more detections
    base_threshold = 1.8 - sensitivity * 1.0  # Range: 0.8 to 1.8
    prob_map = 1.0 / (1.0 + np.exp(-(anomaly_score - base_threshold) * 2))
    
    # Find hotspots via connected components - balanced threshold
    detection_threshold = 0.40 + (1 - sensitivity) * 0.25  # Range: 0.40 to 0.65
    binary = prob_map > detection_threshold
    
    print(f"  Detection threshold: {detection_threshold:.2f}, prob max: {prob_map.max():.2f}")
    
    # Apply morphological operations to clean up
    from scipy.ndimage import binary_opening, binary_closing
    binary = binary_opening(binary, structure=np.ones((2, 2)))
    binary = binary_closing(binary, structure=np.ones((3, 3)))
    
    labeled, n_features = ndimage.label(binary)
    
    # Minimum pixels based on sensitivity
    min_pixels = max(5, int(10 * (1 - sensitivity)))  # 5-10 pixels minimum
    
    hotspots = []
    for i in range(1, n_features + 1):
        mask = labeled == i
        n_pixels = np.sum(mask)
        
        if n_pixels < min_pixels:
            continue
        
        coords = np.where(mask)
        cy, cx = np.mean(coords[0]), np.mean(coords[1])
        
        max_prob = float(prob_map[mask].max())
        mean_prob = float(prob_map[mask].mean())
        
        # Confidence based on probability, size, and compactness
        size_factor = min(1.0, n_pixels / 50)  # Larger = more confident
        compactness = n_pixels / (len(np.unique(coords[0])) * len(np.unique(coords[1])) + 1)
        compact_factor = min(1.0, compactness * 2)
        
        confidence = (
            0.5 * max_prob +
            0.25 * mean_prob +
            0.15 * size_factor +
            0.1 * compact_factor
        )
        
        # Generate GPS coordinates
        base_lat, base_lon = 37.77, -122.42
        lat = base_lat + (cy - h/2) * 0.0001
        lon = base_lon + (cx - w/2) * 0.0001
        
        hotspots.append({
            'id': int(len(hotspots) + 1),
            'confidence': round(float(confidence) * 100, 1),
            'area_m2': int(n_pixels) * 100,
            'lat': round(float(lat), 4),
            'lon': round(float(lon), 4),
            'center_y': int(cy),
            'center_x': int(cx),
            'n_pixels': int(n_pixels),
        })
    
    # Sort by confidence and limit to top results
    hotspots.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Limit based on sensitivity (more sensitivity = more results)
    max_hotspots = int(5 + sensitivity * 15)  # 5-20 hotspots
    hotspots = hotspots[:max_hotspots]
    
    # Assign final ranks
    for i, h in enumerate(hotspots):
        h['rank'] = i + 1
    
    return prob_map.astype(np.float32), hotspots


# =============================================================================
# API SETUP
# =============================================================================
app = FastAPI(
    title="OceanGuard AI API",
    description="Marine Debris Hotspot Detection using Spectral Anomaly Analysis",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model
model: Optional[HotspotDetectorModel] = None


class PredictionResult(BaseModel):
    success: bool
    hotspots_count: int
    avg_confidence: float
    processing_time_ms: float
    heatmap_base64: Optional[str] = None
    preview_base64: Optional[str] = None
    hotspots: List[dict] = []
    message: str = ""


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: str


# =============================================================================
# IMAGE UTILITIES
# =============================================================================
def tif_to_preview_base64(tif_bytes):
    """Convert TIF to PNG for web display."""
    try:
        with MemoryFile(tif_bytes) as memfile:
            with memfile.open() as src:
                if src.count >= 3:
                    r = src.read(4)  # Red
                    g = src.read(3)  # Green
                    b = src.read(2)  # Blue
                else:
                    r = g = b = src.read(1)
                
                def normalize(band):
                    band = band.astype(np.float32)
                    p2, p98 = np.percentile(band, (2, 98))
                    band = np.clip((band - p2) / (p98 - p2 + 1e-8), 0, 1)
                    return (band * 255).astype(np.uint8)
                
                rgb = np.stack([normalize(r), normalize(g), normalize(b)], axis=-1)
                rgb = np.squeeze(rgb)
                
                img = Image.fromarray(rgb)
                buffer = io.BytesIO()
                img.save(buffer, format='PNG')
                return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Preview error: {e}")
        return None


def create_heatmap_overlay(prob_map: np.ndarray):
    """Create a heatmap visualization."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        
        # Custom colormap: transparent -> yellow -> orange -> red
        colors = [(0, 0, 0, 0), (1, 1, 0, 0.4), (1, 0.5, 0, 0.7), (1, 0, 0, 0.9)]
        cmap = mcolors.LinearSegmentedColormap.from_list('debris', colors)
        
        colored = cmap(prob_map)
        img = Image.fromarray((colored * 255).astype(np.uint8))
        
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Heatmap error: {e}")
        return None


# =============================================================================
# MODEL LOADING
# =============================================================================
@app.on_event("startup")
async def startup_event():
    load_model()


def load_model():
    """Load the trained hotspot detector model."""
    global model
    
    if not ML_AVAILABLE:
        print("ML libraries not available - running in demo mode")
        model = HotspotDetectorModel()
        return
    
    if MODEL_PATH.exists():
        try:
            with open(MODEL_PATH, 'r') as f:
                model_dict = json.load(f)
            model = HotspotDetectorModel.from_dict(model_dict)
            print(f"Model loaded from {MODEL_PATH}")
        except Exception as e:
            print(f"Error loading model: {e}")
            model = HotspotDetectorModel()
    else:
        print(f"Model not found at {MODEL_PATH}, using defaults")
        model = HotspotDetectorModel()


# =============================================================================
# API ENDPOINTS
# =============================================================================
@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        model_type="Hotspot Detector (Anomaly-Based)",
    )


@app.post("/predict", response_model=PredictionResult)
async def predict(
    file: UploadFile = File(...),
    sensitivity: float = Query(0.5, ge=0.0, le=1.0)
):
    """Predict marine debris hotspots from uploaded GeoTIFF."""
    start_time = datetime.now()
    
    if not file.filename.endswith(('.tif', '.tiff')):
        raise HTTPException(status_code=400, detail="File must be a GeoTIFF (.tif)")
    
    try:
        contents = await file.read()
        preview_base64 = tif_to_preview_base64(contents)
        
        # Load image
        with MemoryFile(contents) as memfile:
            with memfile.open() as src:
                image = src.read().astype(np.float32)
        
        print(f"Processing image: shape={image.shape}, range=[{image.min():.3f}, {image.max():.3f}]")
        
        # Run detection
        prob_map, hotspots = detect_hotspots(image, model, sensitivity=sensitivity)
        
        print(f"Detection complete: {len(hotspots)} hotspots, prob range=[{prob_map.min():.2f}, {prob_map.max():.2f}]")
        
        # Create heatmap
        heatmap_base64 = create_heatmap_overlay(prob_map)
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        avg_conf = float(np.mean([h['confidence'] for h in hotspots])) if hotspots else 0.0
        
        # Convert all numpy types to Python native types
        hotspots = to_python_types(hotspots)
        
        return PredictionResult(
            success=True,
            hotspots_count=len(hotspots),
            avg_confidence=round(avg_conf, 1),
            processing_time_ms=round(processing_time, 1),
            preview_base64=preview_base64,
            heatmap_base64=heatmap_base64,
            hotspots=hotspots,
            message=f"Detected {len(hotspots)} potential debris hotspots",
        )
        
    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict-sample")
async def predict_sample(
    sample_id: int = Query(1, ge=1, le=5),
    sensitivity: float = Query(0.5, ge=0.0, le=1.0)
):
    """Run prediction on a sample image from the dataset."""
    start_time = datetime.now()
    
    # Sample paths - including ones WITH debris
    sample_paths = {
        1: PRESIDENTIAL_AI_PATH / "data" / "marida" / "patches" / "S2_1-12-19_48MYU" / "S2_1-12-19_48MYU_2.tif",  # Has debris
        2: PRESIDENTIAL_AI_PATH / "data" / "marida" / "patches" / "S2_1-12-19_48MYU" / "S2_1-12-19_48MYU_3.tif",  # Has debris
        3: PRESIDENTIAL_AI_PATH / "data" / "marida" / "patches" / "S2_11-1-19_19QDA" / "S2_11-1-19_19QDA_0.tif",
        4: PRESIDENTIAL_AI_PATH / "data" / "marida" / "patches" / "S2_11-6-18_16PCC" / "S2_11-6-18_16PCC_0.tif",
        5: PRESIDENTIAL_AI_PATH / "data" / "marida" / "patches" / "S2_12-1-17_16PCC" / "S2_12-1-17_16PCC_0.tif",
    }
    
    sample_path = sample_paths.get(sample_id)
    
    if not sample_path or not sample_path.exists():
        # Find any valid sample
        patches_dir = PRESIDENTIAL_AI_PATH / "data" / "marida" / "patches"
        for folder in patches_dir.iterdir():
            if folder.is_dir():
                for tif in folder.glob("*.tif"):
                    if "_cl" not in tif.name and "_conf" not in tif.name:
                        sample_path = tif
                        break
                if sample_path and sample_path.exists():
                    break
    
    if not sample_path or not sample_path.exists():
        return PredictionResult(
            success=False,
            hotspots_count=0,
            avg_confidence=0,
            processing_time_ms=0,
            message="Sample file not found",
            hotspots=[],
        )
    
    try:
        with open(sample_path, 'rb') as f:
            contents = f.read()
        
        preview_base64 = tif_to_preview_base64(contents)
        
        with rasterio.open(sample_path) as src:
            image = src.read().astype(np.float32)
        
        print(f"Sample {sample_id}: {sample_path.name}, shape={image.shape}")
        
        prob_map, hotspots = detect_hotspots(image, model, sensitivity=sensitivity)
        heatmap_base64 = create_heatmap_overlay(prob_map)
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        avg_conf = float(np.mean([h['confidence'] for h in hotspots])) if hotspots else 0.0
        
        # Convert all numpy types to Python native types
        hotspots = to_python_types(hotspots)
        
        return PredictionResult(
            success=True,
            hotspots_count=len(hotspots),
            avg_confidence=round(avg_conf, 1),
            processing_time_ms=round(processing_time, 1),
            preview_base64=preview_base64,
            heatmap_base64=heatmap_base64,
            hotspots=hotspots,
            message=f"Detected {len(hotspots)} potential debris hotspots",
        )
        
    except Exception as e:
        print(f"Sample prediction error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Sample prediction failed: {str(e)}")


@app.get("/sample-preview/{sample_id}")
async def get_sample_preview(sample_id: int):
    """Get preview image for a sample."""
    sample_paths = {
        1: PRESIDENTIAL_AI_PATH / "data" / "marida" / "patches" / "S2_1-12-19_48MYU" / "S2_1-12-19_48MYU_2.tif",
        2: PRESIDENTIAL_AI_PATH / "data" / "marida" / "patches" / "S2_1-12-19_48MYU" / "S2_1-12-19_48MYU_3.tif",
        3: PRESIDENTIAL_AI_PATH / "data" / "marida" / "patches" / "S2_11-1-19_19QDA" / "S2_11-1-19_19QDA_0.tif",
        4: PRESIDENTIAL_AI_PATH / "data" / "marida" / "patches" / "S2_11-6-18_16PCC" / "S2_11-6-18_16PCC_0.tif",
        5: PRESIDENTIAL_AI_PATH / "data" / "marida" / "patches" / "S2_12-1-17_16PCC" / "S2_12-1-17_16PCC_0.tif",
    }
    
    sample_path = sample_paths.get(sample_id)
    
    if not sample_path or not sample_path.exists():
        return {"preview_base64": None, "error": "Sample not found"}
    
    try:
        with open(sample_path, 'rb') as f:
            contents = f.read()
        
        preview_base64 = tif_to_preview_base64(contents)
        return {"preview_base64": preview_base64, "name": sample_path.stem}
    except Exception as e:
        return {"preview_base64": None, "error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
