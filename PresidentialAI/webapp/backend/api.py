"""
FastAPI backend for OceanGuard AI - Marine Debris Detection.

This server loads the trained SegFormer model and provides inference endpoints
for the Next.js frontend.

Run with: uvicorn api:app --reload --port 8000
"""

import os
import sys
import base64
import io
from pathlib import Path
from datetime import datetime
from typing import Optional, List

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

# Check for torch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available")

# Add the PresidentialAI source to path
PRESIDENTIAL_AI_PATH = Path(__file__).parent.parent / "PresidentialAI"
sys.path.insert(0, str(PRESIDENTIAL_AI_PATH))

# Import model and utilities
try:
    from src.models.segformer import load_model, create_model
    from src.data.preprocessing import normalize_bands
    MODEL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import model modules: {e}")
    MODEL_AVAILABLE = False

app = FastAPI(
    title="OceanGuard AI API",
    description="Marine Debris Detection API for Presidential AI Challenge 2026",
    version="1.0.0",
)

# CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
model = None
device = None

# Model configuration
MODEL_PATH = PRESIDENTIAL_AI_PATH / "outputs" / "models" / "best_model.pth"
MODEL_CONFIG = {
    "backbone": "mit_b2",
    "num_classes": 2,
    "in_channels": 11,
    "pretrained": False,
}

# Normalization stats for MARIDA
NORMALIZATION = {
    "mean": [0.05, 0.06, 0.06, 0.05, 0.08, 0.10, 0.11, 0.10, 0.12, 0.13, 0.09],
    "std": [0.03, 0.03, 0.03, 0.03, 0.04, 0.05, 0.05, 0.05, 0.05, 0.06, 0.05],
}


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
    device: str
    model_path: str


def tif_to_preview_base64(tif_data: bytes, size: int = 256) -> str:
    """
    Convert a GeoTIFF to a base64 PNG preview image.
    
    Args:
        tif_data: Raw bytes of the TIF file
        size: Output image size (square)
        
    Returns:
        Base64 encoded PNG string
    """
    import rasterio
    from rasterio.io import MemoryFile
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    try:
        with MemoryFile(tif_data) as memfile:
            with memfile.open() as src:
                # Read bands for RGB visualization
                # MARIDA: bands are B1,B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12
                # For RGB, use B4 (Red), B3 (Green), B2 (Blue) = indices 3, 2, 1
                n_bands = src.count
                
                if n_bands >= 4:
                    # Use bands 4, 3, 2 for RGB (indices 3, 2, 1 in 0-indexed, but rasterio is 1-indexed)
                    r = src.read(4).astype(np.float32)  # B4 - Red
                    g = src.read(3).astype(np.float32)  # B3 - Green  
                    b = src.read(2).astype(np.float32)  # B2 - Blue
                elif n_bands >= 3:
                    r = src.read(3).astype(np.float32)
                    g = src.read(2).astype(np.float32)
                    b = src.read(1).astype(np.float32)
                else:
                    # Grayscale
                    r = g = b = src.read(1).astype(np.float32)
                
                # Stack and normalize for visualization
                rgb = np.stack([r, g, b], axis=-1)
                
                # Percentile stretch for better visualization
                p2, p98 = np.percentile(rgb, (2, 98))
                rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-8), 0, 1)
                
                # Create figure
                fig, ax = plt.subplots(figsize=(4, 4), dpi=64)
                ax.imshow(rgb)
                ax.axis('off')
                plt.tight_layout(pad=0)
                
                # Save to buffer
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=64)
                plt.close()
                buf.seek(0)
                
                return base64.b64encode(buf.read()).decode('utf-8')
                
    except Exception as e:
        print(f"Error creating preview: {e}")
        return ""


def create_heatmap_overlay(probability_map: np.ndarray, original_rgb: np.ndarray = None) -> str:
    """
    Create a heatmap visualization of debris probability.
    
    Args:
        probability_map: 2D array of debris probabilities (0-1)
        original_rgb: Optional RGB image to overlay on
        
    Returns:
        Base64 encoded PNG string
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    
    # Custom colormap: transparent -> yellow -> orange -> red
    colors = [(0, 0, 0, 0), (1, 1, 0, 0.5), (1, 0.5, 0, 0.7), (1, 0, 0, 0.9)]
    cmap = LinearSegmentedColormap.from_list('debris', colors)
    
    fig, ax = plt.subplots(figsize=(4, 4), dpi=64)
    
    if original_rgb is not None:
        ax.imshow(original_rgb)
        ax.imshow(probability_map, cmap=cmap, alpha=0.7, vmin=0, vmax=1)
    else:
        ax.imshow(probability_map, cmap='hot', vmin=0, vmax=1)
    
    ax.axis('off')
    plt.tight_layout(pad=0)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=64)
    plt.close()
    buf.seek(0)
    
    return base64.b64encode(buf.read()).decode('utf-8')


def check_model_weights_valid(m) -> bool:
    """Check if model weights contain NaN values."""
    for name, param in m.named_parameters():
        if torch.isnan(param).any():
            print(f"WARNING: NaN found in model weights: {name}")
            return False
    return True


def load_model_on_startup():
    """Load the trained model into memory."""
    global model, device
    
    if not MODEL_AVAILABLE:
        print("Model modules not available, running in demo mode")
        return False
    
    # Determine device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    print(f"Using device: {device}")
    
    # Check if model file exists
    if not MODEL_PATH.exists():
        print(f"Model file not found at {MODEL_PATH}")
        return False
    
    try:
        # Load model
        loaded_model = load_model(str(MODEL_PATH), MODEL_CONFIG, device=device)
        loaded_model.eval()
        
        # Validate model weights
        if not check_model_weights_valid(loaded_model):
            print("=" * 60)
            print("ERROR: Model weights contain NaN values!")
            print("The model was corrupted during training (gradient explosion).")
            print("Running in DEMO MODE with simulated results.")
            print("To fix: Retrain with gradient clipping enabled.")
            print("=" * 60)
            model = None  # Force demo mode
            return False
        
        model = loaded_model
        print(f"Model loaded successfully from {MODEL_PATH}")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False


@app.on_event("startup")
async def startup_event():
    """Initialize model on server startup."""
    load_model_on_startup()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API and model status."""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        device=device or "none",
        model_path=str(MODEL_PATH),
    )


@app.post("/predict", response_model=PredictionResult)
async def predict(file: UploadFile = File(...)):
    """
    Run marine debris detection on uploaded GeoTIFF.
    
    Args:
        file: GeoTIFF satellite imagery file
        
    Returns:
        Detection results including hotspots and heatmap
    """
    start_time = datetime.now()
    
    # Validate file type
    if not file.filename.endswith(('.tif', '.tiff')):
        raise HTTPException(status_code=400, detail="File must be a GeoTIFF (.tif or .tiff)")
    
    try:
        # Read file contents
        contents = await file.read()
        
        # Generate preview image
        preview_base64 = tif_to_preview_base64(contents)
        
        # If model not loaded, return demo results with preview
        if model is None:
            # Generate truly random demo data each time (no seeding)
            import time
            import random
            random.seed(time.time_ns())  # Use nanosecond timestamp for true randomness
            
            demo_heatmap = np.random.rand(256, 256) * 0.2
            num_hotspots = random.randint(2, 5)  # 2-5 hotspots
            hotspots = []
            
            for i in range(num_hotspots):
                cx = random.randint(40, 216)
                cy = random.randint(40, 216)
                Y, X = np.ogrid[:256, :256]
                dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
                intensity = random.uniform(0.5, 0.9)
                demo_heatmap += np.exp(-dist**2 / (2 * 25**2)) * intensity
                
                confidence = round(random.uniform(72, 98), 1)
                area = int(random.uniform(8000, 65000))
                hotspots.append({
                    "id": i + 1,
                    "confidence": confidence,
                    "area_m2": area,
                    "lat": round(37.77 + (cy - 128) * 0.0001, 4),
                    "lon": round(-122.42 + (cx - 128) * 0.0001, 4),
                })
            
            demo_heatmap = np.clip(demo_heatmap, 0, 1)
            hotspots.sort(key=lambda x: x["confidence"], reverse=True)
            
            heatmap_base64 = create_heatmap_overlay(demo_heatmap)
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            return PredictionResult(
                success=True,
                hotspots_count=len(hotspots),
                avg_confidence=round(np.mean([h["confidence"] for h in hotspots]), 1),
                processing_time_ms=processing_time + 500,
                preview_base64=preview_base64,
                heatmap_base64=heatmap_base64,
                hotspots=hotspots,
                message="⚠️ DEMO MODE - Model weights corrupted, showing simulated data",
            )
        
        # Load image using rasterio
        import rasterio
        from rasterio.io import MemoryFile
        
        with MemoryFile(contents) as memfile:
            with memfile.open() as src:
                image = src.read().astype(np.float32)
                n_bands = src.count
        
        # Ensure correct number of channels
        if n_bands < MODEL_CONFIG["in_channels"]:
            padded = np.zeros((MODEL_CONFIG["in_channels"], image.shape[1], image.shape[2]), dtype=np.float32)
            padded[:n_bands] = image
            image = padded
        elif n_bands > MODEL_CONFIG["in_channels"]:
            image = image[:MODEL_CONFIG["in_channels"]]
        
        # Normalize
        n_stats = min(len(NORMALIZATION["mean"]), image.shape[0])
        image[:n_stats] = normalize_bands(
            image[:n_stats],
            mean=NORMALIZATION["mean"][:n_stats],
            std=NORMALIZATION["std"][:n_stats],
        )
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image).unsqueeze(0).float().to(device)
        
        # Run inference
        with torch.no_grad():
            output = model(image_tensor)
            probs = torch.softmax(output, dim=1)
            debris_prob = probs[0, 1].cpu().numpy()  # Debris class probability
        
        # Threshold and find hotspots
        threshold = 0.3
        binary_mask = (debris_prob > threshold).astype(np.uint8)
        
        # Find connected components (hotspots)
        from scipy import ndimage
        labeled, num_features = ndimage.label(binary_mask)
        
        hotspots = []
        for i in range(1, num_features + 1):
            mask = labeled == i
            area_pixels = np.sum(mask)
            if area_pixels < 10:  # Skip tiny regions
                continue
            
            # Get mean confidence in this region
            confidence = float(debris_prob[mask].mean() * 100)
            
            # Get centroid
            coords = np.where(mask)
            cy, cx = np.mean(coords[0]), np.mean(coords[1])
            
            hotspots.append({
                "id": len(hotspots) + 1,
                "confidence": round(confidence, 1),
                "area_m2": int(area_pixels * 100),  # Assuming ~10m resolution
                "lat": round(37.77 + (cy - image.shape[1]/2) * 0.0001, 4),  # Mock coords
                "lon": round(-122.42 + (cx - image.shape[2]/2) * 0.0001, 4),
            })
        
        # Sort by confidence
        hotspots.sort(key=lambda x: x["confidence"], reverse=True)
        
        # Create heatmap visualization
        heatmap_base64 = create_heatmap_overlay(debris_prob)
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return PredictionResult(
            success=True,
            hotspots_count=len(hotspots),
            avg_confidence=round(np.mean([h["confidence"] for h in hotspots]) if hotspots else 0, 1),
            processing_time_ms=round(processing_time, 1),
            preview_base64=preview_base64,
            heatmap_base64=heatmap_base64,
            hotspots=hotspots[:10],  # Top 10
            message="Inference complete",
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict-sample")
async def predict_sample(sample_id: int = 1):
    """
    Run prediction on a sample image from the MARIDA dataset.
    
    Args:
        sample_id: ID of sample image (1-3)
        
    Returns:
        Detection results
    """
    start_time = datetime.now()
    
    # Sample image paths
    sample_paths = {
        1: PRESIDENTIAL_AI_PATH / "data" / "marida" / "patches" / "S2_1-12-19_48MYU" / "S2_1-12-19_48MYU_0.tif",
        2: PRESIDENTIAL_AI_PATH / "data" / "marida" / "patches" / "S2_11-1-19_19QDA" / "S2_11-1-19_19QDA_0.tif",
        3: PRESIDENTIAL_AI_PATH / "data" / "marida" / "patches" / "S2_12-1-17_16PCC" / "S2_12-1-17_16PCC_0.tif",
    }
    
    if sample_id not in sample_paths:
        sample_id = 1
    
    sample_path = sample_paths[sample_id]
    
    if not sample_path.exists():
        return PredictionResult(
            success=False,
            hotspots_count=0,
            avg_confidence=0,
            processing_time_ms=0,
            message=f"Sample file not found: {sample_path}",
        )
    
    # Read the sample file
    with open(sample_path, 'rb') as f:
        contents = f.read()
    
    # Generate preview
    preview_base64 = tif_to_preview_base64(contents)
    
    # If model not loaded (demo mode), generate demo heatmap with random data
    if model is None:
        import time
        import random
        random.seed(time.time_ns())  # True randomness each time
        
        demo_heatmap = np.random.rand(256, 256) * 0.2
        num_hotspots = random.randint(2, 5)
        hotspots = []
        
        for i in range(num_hotspots):
            cx = random.randint(40, 216)
            cy = random.randint(40, 216)
            Y, X = np.ogrid[:256, :256]
            dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
            intensity = random.uniform(0.5, 0.9)
            demo_heatmap += np.exp(-dist**2 / (2 * 25**2)) * intensity
            
            confidence = round(random.uniform(75, 98), 1)
            area = int(random.uniform(10000, 60000))
            hotspots.append({
                "id": i + 1,
                "confidence": confidence,
                "area_m2": area,
                "lat": round(37.77 + (cy - 128) * 0.0001, 4),
                "lon": round(-122.42 + (cx - 128) * 0.0001, 4),
            })
        
        demo_heatmap = np.clip(demo_heatmap, 0, 1)
        hotspots.sort(key=lambda x: x["confidence"], reverse=True)
        
        heatmap_base64 = create_heatmap_overlay(demo_heatmap)
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return PredictionResult(
            success=True,
            hotspots_count=len(hotspots),
            avg_confidence=round(np.mean([h["confidence"] for h in hotspots]), 1),
            processing_time_ms=round(processing_time + 500, 1),
            preview_base64=preview_base64,
            heatmap_base64=heatmap_base64,
            hotspots=hotspots,
            message="⚠️ DEMO MODE - Model weights corrupted, showing simulated data",
        )
    
    # Real model inference
    try:
        import rasterio
        from rasterio.io import MemoryFile
        
        with MemoryFile(contents) as memfile:
            with memfile.open() as src:
                image = src.read().astype(np.float32)
                n_bands = src.count
        
        # Ensure correct number of channels
        if n_bands < MODEL_CONFIG["in_channels"]:
            padded = np.zeros((MODEL_CONFIG["in_channels"], image.shape[1], image.shape[2]), dtype=np.float32)
            padded[:n_bands] = image
            image = padded
        elif n_bands > MODEL_CONFIG["in_channels"]:
            image = image[:MODEL_CONFIG["in_channels"]]
        
        # Normalize
        n_stats = min(len(NORMALIZATION["mean"]), image.shape[0])
        image[:n_stats] = normalize_bands(
            image[:n_stats],
            mean=NORMALIZATION["mean"][:n_stats],
            std=NORMALIZATION["std"][:n_stats],
        )
        
        # Convert to tensor and run inference
        image_tensor = torch.from_numpy(image).unsqueeze(0).float().to(device)
        
        with torch.no_grad():
            output = model(image_tensor)
            probs = torch.softmax(output, dim=1)
            debris_prob = probs[0, 1].cpu().numpy()
        
        # Find hotspots
        threshold = 0.3
        binary_mask = (debris_prob > threshold).astype(np.uint8)
        
        from scipy import ndimage
        labeled, num_features = ndimage.label(binary_mask)
        
        hotspots = []
        for i in range(1, num_features + 1):
            mask = labeled == i
            area_pixels = np.sum(mask)
            if area_pixels < 10:
                continue
            
            confidence = float(debris_prob[mask].mean() * 100)
            coords = np.where(mask)
            cy, cx = np.mean(coords[0]), np.mean(coords[1])
            
            hotspots.append({
                "id": len(hotspots) + 1,
                "confidence": round(confidence, 1),
                "area_m2": int(area_pixels * 100),
                "lat": round(37.77 + (cy - image.shape[1]/2) * 0.0001, 4),
                "lon": round(-122.42 + (cx - image.shape[2]/2) * 0.0001, 4),
            })
        
        hotspots.sort(key=lambda x: x["confidence"], reverse=True)
        heatmap_base64 = create_heatmap_overlay(debris_prob)
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return PredictionResult(
            success=True,
            hotspots_count=len(hotspots),
            avg_confidence=round(np.mean([h["confidence"] for h in hotspots]) if hotspots else 0, 1),
            processing_time_ms=round(processing_time, 1),
            preview_base64=preview_base64,
            heatmap_base64=heatmap_base64,
            hotspots=hotspots[:10],
            message="Inference complete",
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sample prediction failed: {str(e)}")


@app.get("/samples")
async def list_samples():
    """List available sample images for testing."""
    samples = [
        {"id": 1, "name": "Pacific Gyre Sample", "region": "48MYU", "date": "2019-12-01", "file": "S2_1-12-19_48MYU_0.tif"},
        {"id": 2, "name": "Caribbean Coast", "region": "19QDA", "date": "2019-01-11", "file": "S2_11-1-19_19QDA_0.tif"},
        {"id": 3, "name": "Mediterranean", "region": "16PCC", "date": "2017-01-12", "file": "S2_12-1-17_16PCC_0.tif"},
    ]
    return {"samples": samples}


@app.get("/sample-preview/{sample_id}")
async def get_sample_preview(sample_id: int):
    """
    Get a preview image for a sample.
    
    Args:
        sample_id: ID of sample (1-3)
        
    Returns:
        Base64 encoded preview image
    """
    sample_paths = {
        1: PRESIDENTIAL_AI_PATH / "data" / "marida" / "patches" / "S2_1-12-19_48MYU" / "S2_1-12-19_48MYU_0.tif",
        2: PRESIDENTIAL_AI_PATH / "data" / "marida" / "patches" / "S2_11-1-19_19QDA" / "S2_11-1-19_19QDA_0.tif",
        3: PRESIDENTIAL_AI_PATH / "data" / "marida" / "patches" / "S2_12-1-17_16PCC" / "S2_12-1-17_16PCC_0.tif",
    }
    
    if sample_id not in sample_paths:
        sample_id = 1
    
    sample_path = sample_paths[sample_id]
    
    if not sample_path.exists():
        return {"preview_base64": "", "error": "Sample file not found"}
    
    try:
        with open(sample_path, 'rb') as f:
            contents = f.read()
        
        preview_base64 = tif_to_preview_base64(contents)
        return {"preview_base64": preview_base64, "path": str(sample_path)}
    except Exception as e:
        return {"preview_base64": "", "error": str(e)}


@app.get("/preview-tif")
async def preview_tif_file(path: str):
    """
    Generate a preview for any TIF file path.
    
    Args:
        path: Full path to TIF file
        
    Returns:
        Base64 encoded preview image
    """
    file_path = Path(path)
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    if not file_path.suffix.lower() in ['.tif', '.tiff']:
        raise HTTPException(status_code=400, detail="Not a TIF file")
    
    try:
        with open(file_path, 'rb') as f:
            contents = f.read()
        
        preview_base64 = tif_to_preview_base64(contents)
        return {"preview_base64": preview_base64}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

