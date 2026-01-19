# OceanGuard AI - Marine Debris Detection System

**Satellite-Based Trash Hotspot Detection Using Deep Learning**

*Presidential AI Challenge Submission*

---

## Overview

OceanGuard AI is an automated marine debris detection system that identifies floating trash, plastics, and waste in ocean waters using Sentinel-2 satellite imagery. The system employs a deep learning semantic segmentation model trained on the MARIDA (Marine Debris Archive) dataset to produce georeferenced probability heatmaps, ranked hotspot lists with GPS coordinates, and exportable data in multiple formats.

This solution addresses a critical environmental challenge: marine debris accumulates offshore before washing onto coastlines, damaging ecosystems, fisheries, and local economies. Traditional monitoring using ships, aircraft, or buoys is prohibitively expensive. This AI-driven approach enables scalable, low-cost detection using publicly available satellite data.

---

## Key Features

- **Deep Learning Detection**: Improved U-Net architecture trained on real MARIDA satellite imagery
- **High Accuracy**: Model achieves up to 80% confidence on debris detections
- **Fast Processing**: Sub-second inference time per image
- **Interactive Web Interface**: Modern Next.js frontend with real-time visualization
- **RESTful API**: FastAPI backend with comprehensive error handling
- **Georeferenced Outputs**: GPS coordinates, area calculations, and confidence scores for each detection
- **Multiple Export Formats**: JSON, CSV, and GeoJSON support
- **Apple Silicon Optimized**: Automatic MPS (Metal Performance Shaders) detection for M-series Macs

---

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.10+ | 3.11+ |
| Node.js | 18+ | 20+ |
| RAM | 16GB | 32GB+ |
| Storage | 20GB | 50GB+ |
| GPU | None (CPU works) | Apple M4 Max / NVIDIA GPU (8GB+ VRAM) |

---

## Installation

### Prerequisites

Ensure you have Python 3.10+ and Node.js 18+ installed on your system.

### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install Python dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install Node.js dependencies:
```bash
npm install
```

---

## Quick Start

### Starting the Backend API

From the `backend` directory:

```bash
python api.py
```

The API will start on `http://localhost:8000`. The API documentation is available at `http://localhost:8000/docs`.

### Starting the Frontend

From the `frontend` directory:

```bash
npm run dev
```

The web application will start on `http://localhost:3000`.

### Using the Application

1. Navigate to `http://localhost:3000/analyze` in your web browser
2. Upload a Sentinel-2 GeoTIFF image (11-band format recommended)
3. Adjust sensitivity settings if needed (default: 0.5)
4. View detection results including:
   - Probability heatmap visualization
   - Ranked hotspot list with confidence scores
   - GPS coordinates for each detection
   - Exportable data in multiple formats

---

## Project Structure

```
presidentialAI/
├── backend/                    # FastAPI backend server
│   ├── api.py                 # Main API endpoint definitions
│   ├── requirements.txt       # Python dependencies
│   └── backend.log           # Server logs
│
├── frontend/                  # Next.js frontend application
│   ├── src/
│   │   └── app/              # React components and pages
│   │       ├── analyze/      # Main analysis page
│   │       ├── dashboard/    # Dashboard view
│   │       └── api/          # API route handlers
│   ├── package.json          # Node.js dependencies
│   └── next.config.ts        # Next.js configuration
│
├── PresidentialAI/           # Machine learning model and training
│   ├── data/
│   │   └── marida/          # MARIDA dataset (user-provided)
│   │       ├── patches/     # Image and label patches
│   │       └── splits/      # Train/val/test splits
│   │
│   ├── outputs/
│   │   └── models/          # Trained model checkpoints
│   │       └── improved_model.pth
│   │
│   ├── scripts/             # Training and evaluation scripts
│   │   ├── train_marida.py  # Main training script
│   │   ├── train_improved.py # Model architecture definition
│   │   └── evaluate_model.py # Model evaluation
│   │
│   ├── src/                 # Source code modules
│   │   ├── data/           # Data loading and preprocessing
│   │   ├── models/         # Neural network architectures
│   │   ├── training/       # Training utilities
│   │   └── utils/          # Helper functions
│   │
│   └── config.yaml          # Configuration file
│
└── README.md                # This file
```

---

## Model Architecture

The system uses an Improved U-Net architecture for semantic segmentation:

- **Architecture**: U-Net with enhanced encoder-decoder structure
- **Input**: 11-band Sentinel-2 imagery (256x256 patches)
- **Output**: Binary segmentation (debris vs. non-debris)
- **Loss Function**: Focal Loss with alpha=0.75, gamma=2.0 for class imbalance handling
- **Training Data**: Real MARIDA dataset (1,000+ training samples, 200+ validation samples)
- **Training Epochs**: 50 epochs with early stopping
- **Optimizer**: AdamW with learning rate 5e-4
- **Data Augmentation**: Horizontal/vertical flips, rotations

### Model Performance

| Metric | Value |
|--------|-------|
| Max Detection Confidence | 80%+ |
| Average Confidence | 60-65% |
| Processing Time | <1 second per image |
| Model Size | 355 MB |
| Input Format | 11-band Sentinel-2 GeoTIFF |

---

## Training a Custom Model

### Dataset Preparation

The model is trained on the MARIDA (Marine Debris Archive) dataset. Ensure the dataset is located at `PresidentialAI/data/marida/` with the following structure:

```
data/marida/
├── patches/
│   └── S2_DATE_ROI/
│       ├── S2_DATE_ROI_0.tif      # Image patches
│       └── S2_DATE_ROI_0_cl.tif   # Label patches
└── splits/
    ├── train_X.txt
    ├── val_X.txt
    └── test_X.txt
```

### Training Process

1. Navigate to the PresidentialAI directory:
```bash
cd PresidentialAI
```

2. Run the training script:
```bash
python scripts/train_marida.py
```

Training parameters can be modified in the script:
- `BATCH_SIZE`: Batch size (default: 8)
- `EPOCHS`: Number of training epochs (default: 50)
- `LEARNING_RATE`: Learning rate (default: 5e-4)
- `max_samples`: Number of training samples (default: 1000)

3. Monitor training progress:
```bash
tail -f training_marida_final.log
```

4. The best model is automatically saved to `outputs/models/improved_model.pth` based on validation F1 score.

### Model Evaluation

Evaluate the trained model:
```bash
python scripts/evaluate_model.py
```

---

## API Documentation

### Endpoints

#### POST `/predict`

Upload a Sentinel-2 GeoTIFF image for debris detection.

**Request:**
- `file`: GeoTIFF image file (multipart/form-data)
- `sensitivity`: Float between 0.1 and 0.9 (optional, default: 0.5)

**Response:**
```json
{
  "success": true,
  "hotspots_count": 42,
  "avg_confidence": 61.3,
  "processing_time_ms": 600.5,
  "preview_base64": "data:image/png;base64,...",
  "heatmap_base64": "data:image/png;base64,...",
  "hotspots": [
    {
      "id": 1,
      "confidence": 80.3,
      "area_m2": 800,
      "lat": 37.7640,
      "lon": -122.4079,
      "center_y": 128,
      "center_x": 128,
      "n_pixels": 8,
      "rank": 1
    }
  ],
  "message": "Detected 42 potential debris hotspots using deep learning model"
}
```

#### GET `/health`

Check API health status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "mps"
}
```

### Example Usage

**Python:**
```python
import requests

with open('sentinel2_image.tif', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/predict',
        files={'file': f},
        params={'sensitivity': 0.5}
    )
    result = response.json()
    print(f"Detected {result['hotspots_count']} hotspots")
```

**cURL:**
```bash
curl -X POST "http://localhost:8000/predict?sensitivity=0.5" \
  -F "file=@sentinel2_image.tif"
```

---

## Configuration

### Backend Configuration

Model and processing parameters are configured in `backend/api.py`:

- `MODEL_PATH`: Path to trained model checkpoint
- `IN_CHANNELS`: Number of input channels (11 for Sentinel-2)
- `NUM_CLASSES`: Number of output classes (2 for binary segmentation)
- `DEVICE`: Computation device (auto-detected: mps/cuda/cpu)

### Threshold Configuration

The detection threshold is automatically adjusted based on:
- Maximum probability in the image
- Sensitivity parameter (0.1-0.9)
- Probability distribution statistics

For high-confidence detections (>70% max probability), the threshold ranges from 0.65 to 0.8. For lower probabilities, percentile-based thresholding is used.

---

## Output Formats

### Hotspot Data

Each detected hotspot includes:
- **ID**: Unique identifier
- **Confidence**: Detection confidence (0-100%)
- **Area**: Estimated area in square meters
- **Coordinates**: GPS latitude and longitude
- **Pixel Coordinates**: Center coordinates in image space
- **Rank**: Ranked by confidence (1 = highest)

### Export Formats

- **JSON**: Full API response with all metadata
- **CSV**: Tabular format for spreadsheet analysis
- **GeoJSON**: Geospatial format for GIS applications

---

## Performance Optimization

### Apple Silicon (M-series Macs)

The system automatically detects and uses MPS acceleration:
- Batch processing: 8-16 samples
- Memory efficient: Automatic gradient checkpointing
- Recommended: M4 Max or better for training

### NVIDIA GPU

For CUDA-enabled systems:
- Automatic GPU detection
- Batch size: 8-32 depending on VRAM
- Mixed precision training supported

### CPU-Only

The system works on CPU but will be significantly slower:
- Processing time: 3-5 seconds per image
- Training not recommended on CPU

---

## Troubleshooting

### Model Not Loading

If the model fails to load:
1. Verify `PresidentialAI/outputs/models/improved_model.pth` exists
2. Check file permissions
3. Ensure PyTorch is correctly installed

### Low Detection Confidence

If detections show low confidence:
1. Verify input image is properly normalized
2. Check that image contains 11 bands
3. Ensure image values are in reflectance range (0-1) or DN range (0-10000)

### Too Many Hotspots

If too many hotspots are detected:
1. Increase the sensitivity parameter (closer to 1.0)
2. The threshold will automatically increase
3. Minimum area filter will remove smaller detections

---

## Dataset Information

### MARIDA Dataset

The Marine Debris Archive (MARIDA) dataset:
- **Source**: Kikaki et al., 2022
- **Scenes**: 63 Sentinel-2 scenes
- **Patches**: 256x256 pixel patches
- **Classes**: 15 classes including debris, water, algae, ships
- **License**: See dataset repository for license information

### Citation

If you use the MARIDA dataset, please cite:

```
Kikaki, K., Kakogeorgiou, I., Mikeli, P., Raitsos, D. E., & Karantzalos, K. (2022).
MARIDA: A benchmark for Marine Debris detection from Sentinel-2 remote sensing data.
PLoS ONE, 17(1), e0262247.
```

---

## Contributing

Contributions are welcome. Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 for Python code
- Use TypeScript for frontend code
- Add tests for new features
- Update documentation as needed

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Acknowledgments

- **MARIDA Dataset**: Kikaki et al., 2022 - Marine Debris Archive
- **European Space Agency**: Sentinel-2 satellite imagery
- **Presidential AI Challenge**: Organizers and sponsors
- **PyTorch Team**: Deep learning framework
- **FastAPI**: Modern web framework for APIs
- **Next.js**: React framework for frontend

---

## Contact

For questions, issues, or contributions:
- Open an issue on GitHub
- Contact the development team

---

*Built for the Presidential AI Challenge 2026*
