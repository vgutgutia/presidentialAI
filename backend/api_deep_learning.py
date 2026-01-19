"""
OceanGuard AI - Marine Debris Detection API
Uses trained deep learning model for debris detection
"""

import io
import base64
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

# Add PresidentialAI to path
BACKEND_DIR = Path(__file__).parent
PROJECT_ROOT = BACKEND_DIR.parent
PRESIDENTIAL_AI_PATH = PROJECT_ROOT / "PresidentialAI"
sys.path.insert(0, str(PRESIDENTIAL_AI_PATH))

# Try to import ML libraries
try:
    import rasterio
    from rasterio.io import MemoryFile
    from scipy import ndimage
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("Warning: ML libraries not available")

# Import model
try:
    from scripts.train_improved import ImprovedUNet
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    print("Warning: Model classes not available")

# Paths
MODEL_PATH = PRESIDENTIAL_AI_PATH / "outputs" / "models" / "improved_model.pth"
IN_CHANNELS = 11
NUM_CLASSES = 2

# Device
if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print(f"Using device: {DEVICE}")

# =============================================================================
# API SETUP
# =============================================================================
app = FastAPI(
    title="OceanGuard AI API",
    description="Marine Debris Detection using Deep Learning",
    version="4.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model
model: Optional[torch.nn.Module] = None


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
    device: str


# =============================================================================
# IMAGE UTILITIES
# =============================================================================
def tif_to_preview_base64(tif_bytes):
    """Convert TIF to PNG for web display."""
    if not ML_AVAILABLE:
        raise ValueError("rasterio is not installed")
    try:
        from rasterio.io import MemoryFile
        with MemoryFile(tif_bytes) as memfile:
            with memfile.open() as src:
                if src.count >= 3:
                    r = src.read(4) if src.count >= 4 else src.read(1)
                    g = src.read(3) if src.count >= 3 else src.read(1)
                    b = src.read(2) if src.count >= 2 else src.read(1)
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


def create_heatmap_overlay(prob_map: np.ndarray) -> str:
    """Create heatmap visualization from probability map."""
    try:
        # Normalize to 0-255
        prob_uint8 = (np.clip(prob_map, 0, 1) * 255).astype(np.uint8)
        
        # Create colormap (blue to red)
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        
        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(prob_uint8, cmap='hot', vmin=0, vmax=255)
        ax.axis('off')
        plt.tight_layout(pad=0)
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='PNG', bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close()
        
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Heatmap error: {e}")
        return None


def extract_hotspots(prob_map: np.ndarray, threshold: float = 0.5, min_area: int = 100) -> List[dict]:
    """Extract hotspot regions from probability map."""
    from scipy.ndimage import label, find_objects
    
    # Create binary mask
    binary = (prob_map >= threshold).astype(np.uint8)
    
    # Find connected components
    labeled, num_features = label(binary)
    
    hotspots = []
    h, w = prob_map.shape
    
    for i in range(1, num_features + 1):
        mask = (labeled == i)
        area = np.sum(mask)
        
        if area < min_area:
            continue
        
        # Get bounding box
        slices = find_objects(labeled == i)[0]
        y_min, y_max = slices[0].start, slices[0].stop
        x_min, x_max = slices[1].start, slices[1].stop
        
        # Center
        cy = (y_min + y_max) // 2
        cx = (x_min + x_max) // 2
        
        # Confidence from max probability in region
        confidence = float(prob_map[mask].max())
        
        # Generate GPS coordinates (approximate)
        base_lat, base_lon = 37.77, -122.42
        lat = base_lat + (cy - h/2) * 0.0001
        lon = base_lon + (cx - w/2) * 0.0001
        
        hotspots.append({
            'id': len(hotspots) + 1,
            'confidence': round(confidence * 100, 1),
            'area_m2': int(area * 100),
            'lat': round(float(lat), 4),
            'lon': round(float(lon), 4),
            'center_y': int(cy),
            'center_x': int(cx),
            'n_pixels': int(area),
            'rank': len(hotspots) + 1,
        })
    
    # Sort by confidence
    hotspots.sort(key=lambda x: x['confidence'], reverse=True)
    for i, h in enumerate(hotspots):
        h['rank'] = i + 1
    
    return hotspots


# =============================================================================
# MODEL LOADING
# =============================================================================
@app.on_event("startup")
async def startup_event():
    global model
    load_model()


def load_model():
    """Load the trained deep learning model."""
    global model
    
    if not MODEL_AVAILABLE:
        print("Warning: Model classes not available")
        return
    
    if not MODEL_PATH.exists():
        print(f"Model not found at {MODEL_PATH}")
        return
    
    try:
        model = ImprovedUNet(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
        checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(DEVICE)
        model.eval()
        print(f"✅ Model loaded from {MODEL_PATH}")
        print(f"   Device: {DEVICE}")
        print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"   F1: {checkpoint.get('f1', 'N/A')}")
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        model = None


# =============================================================================
# API ENDPOINTS
# =============================================================================
@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        model_type="Deep Learning (Improved UNet)" if model is not None else "Not loaded",
        device=DEVICE,
    )


@app.post("/predict", response_model=PredictionResult)
async def predict(
    file: UploadFile = File(...),
    sensitivity: float = Query(0.5, ge=0.0, le=1.0)
):
    """Predict marine debris hotspots from uploaded GeoTIFF using deep learning."""
    start_time = datetime.now()
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if not file.filename.endswith(('.tif', '.tiff')):
        raise HTTPException(status_code=400, detail="File must be a GeoTIFF (.tif)")
    
    try:
        if not ML_AVAILABLE:
            raise HTTPException(
                status_code=500,
                detail="ML libraries (rasterio) not available"
            )
        
        contents = await file.read()
        preview_base64 = tif_to_preview_base64(contents)
        
        # Load image
        from rasterio.io import MemoryFile
        with MemoryFile(contents) as memfile:
            with memfile.open() as src:
                image = src.read().astype(np.float32)
        
        print(f"Processing image: shape={image.shape}, range=[{image.min():.3f}, {image.max():.3f}]")
        
        # Normalize image (simple normalization for now)
        if image.max() > 1.0:
            image = image / 10000.0  # Sentinel-2 scaling
        
        # Ensure we have 11 channels (pad or select)
        if image.shape[0] < IN_CHANNELS:
            # Pad with zeros or duplicate last channel
            padding = np.zeros((IN_CHANNELS - image.shape[0], *image.shape[1:]), dtype=image.dtype)
            image = np.concatenate([image, padding], axis=0)
        elif image.shape[0] > IN_CHANNELS:
            image = image[:IN_CHANNELS]
        
        # Convert to tensor and add batch dimension
        image_tensor = torch.from_numpy(image).unsqueeze(0).to(DEVICE)
        
        # Run inference
        with torch.no_grad():
            output = model(image_tensor)
            probs = F.softmax(output, dim=1)
            debris_probs = probs[0, 1].cpu().numpy()  # Get debris class probabilities
        
        print(f"Inference complete: prob range=[{debris_probs.min():.2f}, {debris_probs.max():.2f}]")
        
        # Adjust threshold based on sensitivity (lower sensitivity = higher threshold)
        threshold = 0.5 + (0.5 - sensitivity) * 0.3  # Range: 0.2 to 0.8
        
        # Extract hotspots
        hotspots = extract_hotspots(debris_probs, threshold=threshold, min_area=50)
        
        print(f"Detection complete: {len(hotspots)} hotspots")
        
        # Create heatmap
        heatmap_base64 = create_heatmap_overlay(debris_probs)
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        avg_conf = float(np.mean([h['confidence'] for h in hotspots])) if hotspots else 0.0
        
        return PredictionResult(
            success=True,
            hotspots_count=len(hotspots),
            avg_confidence=round(avg_conf, 1),
            processing_time_ms=round(processing_time, 1),
            preview_base64=preview_base64,
            heatmap_base64=heatmap_base64,
            hotspots=hotspots,
            message=f"Detected {len(hotspots)} potential debris hotspots using deep learning model",
        )
        
    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
