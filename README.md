# Marine Debris Early Warning System

**Satellite-Based Detection of Offshore Marine Debris Using Deep Learning**

*Presidential AI Challenge Submission*

---

## Overview

This system automatically detects marine debris (plastics, waste) floating in ocean waters using free Sentinel-2 satellite imagery. It produces georeferenced probability heatmaps, ranked hotspot lists with GPS coordinates, and GIS-ready outputs for integration with environmental monitoring workflows.

The solution addresses a critical environmental challenge: marine debris accumulates offshore before washing onto coastlines, damaging ecosystems, fisheries, and local economies. Traditional monitoring using ships, aircraft, or buoys is prohibitively expensive. This AI-driven approach enables scalable, low-cost detection using publicly available satellite data.

---

## Key Features

- **Multispectral Analysis**: Processes 6 Sentinel-2 bands (Blue, Green, Red, NIR, SWIR1, SWIR2) for robust debris discrimination
- **Transformer Architecture**: SegFormer-based semantic segmentation adapted for satellite imagery
- **Georeferenced Outputs**: GeoTIFF heatmaps, GeoJSON polygons, and CSV reports preserving spatial metadata
- **Apple Silicon Optimized**: Automatic MPS (Metal Performance Shaders) detection for M-series Macs
- **Production Ready**: Comprehensive error handling, logging, and checkpoint management

---

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.10+ | 3.11 |
| RAM | 16GB | 32GB+ |
| Storage | 20GB | 50GB+ |
| GPU | None (CPU works) | Apple M4 Max / NVIDIA GPU |

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/marine-debris-detection.git
cd marine-debris-detection
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

---

## Dataset Setup

This project uses the MARIDA (Marine Debris Archive) dataset, which must be downloaded manually from GitHub.

### Download Instructions

1. Visit: https://github.com/marine-debris/marine-debris.github.io

2. Clone or download the repository:
   ```bash
   git clone https://github.com/marine-debris/marine-debris.github.io.git
   ```

3. Copy the data to your project:
   ```bash
   mkdir -p data/marida
   cp -r marine-debris.github.io/patches data/marida/
   cp -r marine-debris.github.io/shapefiles data/marida/
   cp -r marine-debris.github.io/splits data/marida/
   cp marine-debris.github.io/labels_mapping.txt data/marida/
   ```

4. Verify the dataset:
   ```bash
   python scripts/download_marida.py --verify
   ```

### Expected Directory Structure

```
data/marida/
├── labels_mapping.txt
├── patches/
│   └── S2_DATE_ROI/
│       ├── S2_DATE_ROI_0.tif      (image patches)
│       └── S2_DATE_ROI_0_cl.tif   (label patches)
├── shapefiles/
└── splits/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

### Alternative: Sample Data for Testing

To test the pipeline without downloading the full dataset:

```bash
python scripts/download_marida.py --sample-only
```

---

## Usage

### Training

```bash
# Basic training (auto-detects MPS on Apple Silicon)
python scripts/train.py

# Custom training configuration
python scripts/train.py --epochs 100 --batch-size 8 --lr 0.0001

# Resume from checkpoint
python scripts/train.py --checkpoint outputs/models/checkpoint.pth
```

Training progress is logged to TensorBoard:
```bash
tensorboard --logdir outputs/logs/tensorboard
```

### Inference

```bash
# Single image prediction
python scripts/predict.py \
    --input path/to/sentinel2_scene.tif \
    --model outputs/models/best_model.pth \
    --output outputs/predictions/

# Batch processing
python scripts/predict.py \
    --input-dir data/raw/scenes/ \
    --model outputs/models/best_model.pth \
    --output outputs/predictions/

# With visualization
python scripts/predict.py \
    --input path/to/scene.tif \
    --model outputs/models/best_model.pth \
    --visualize
```

### Evaluation

```bash
python scripts/evaluate.py \
    --model outputs/models/best_model.pth \
    --data-dir data/marida \
    --split test
```

---

## Project Structure

```
marine-debris-detection/
├── config.yaml              # Main configuration file
├── requirements.txt         # Python dependencies
├── setup.py                 # Package installation
│
├── src/                     # Source code
│   ├── data/                # Data loading and preprocessing
│   │   ├── dataset.py       # PyTorch datasets for MARIDA
│   │   ├── preprocessing.py # Image preprocessing utilities
│   │   └── download.py      # Dataset download helpers
│   │
│   ├── models/              # Neural network architectures
│   │   └── segformer.py     # SegFormer adapted for multispectral input
│   │
│   ├── training/            # Training utilities
│   │   ├── trainer.py       # Training loop
│   │   ├── losses.py        # Loss functions (Dice, Combined)
│   │   └── metrics.py       # Evaluation metrics (IoU, F1)
│   │
│   ├── inference/           # Prediction pipeline
│   │   └── predictor.py     # Inference engine with tiling
│   │
│   └── utils/               # Utilities
│       ├── device.py        # Device detection (MPS/CUDA/CPU)
│       ├── config.py        # Configuration management
│       ├── geo.py           # Geospatial utilities
│       └── visualization.py # Plotting functions
│
├── scripts/                 # Entry point scripts
│   ├── download_marida.py   # Dataset setup
│   ├── train.py             # Training script
│   ├── predict.py           # Inference script
│   └── evaluate.py          # Evaluation script
│
├── data/                    # Data directory
│   └── marida/              # MARIDA dataset (user-provided)
│
├── outputs/                 # Output directory
│   ├── models/              # Saved model weights
│   ├── predictions/         # Prediction outputs
│   └── logs/                # Training logs
│
├── tests/                   # Unit tests
└── notebooks/               # Jupyter notebooks
```

---

## Output Formats

### 1. Probability Heatmap (GeoTIFF)

Georeferenced probability map with values 0-1 indicating debris likelihood:
```
outputs/predictions/scene_name_heatmap.tif
```

### 2. Hotspot Polygons (GeoJSON)

Vector polygons of detected debris regions with attributes:
```json
{
  "type": "FeatureCollection",
  "features": [{
    "type": "Feature",
    "properties": {
      "confidence": 0.87,
      "area_m2": 45000,
      "centroid_lat": 37.7892,
      "centroid_lon": -122.4324,
      "rank": 1
    },
    "geometry": { "type": "Polygon", "coordinates": [...] }
  }]
}
```

### 3. Ranked Hotspot Report (CSV)

Tabular summary for prioritizing cleanup operations:
```csv
rank,latitude,longitude,area_m2,confidence,timestamp
1,37.7892,-122.4324,45000,0.87,2024-01-15T10:30:00Z
2,37.8123,-122.3987,32000,0.82,2024-01-15T10:30:00Z
```

---

## Configuration

All parameters are configured in `config.yaml`:

```yaml
# Model settings
model:
  backbone: "mit_b2"      # SegFormer variant
  num_classes: 2          # Binary classification
  in_channels: 6          # Sentinel-2 bands

# Training settings
training:
  epochs: 100
  batch_size: 8
  learning_rate: 0.0001
  loss:
    type: "combined"      # Cross-entropy + Dice
    ce_weight: 0.5
    dice_weight: 0.5

# Inference settings
inference:
  confidence_threshold: 0.5
  min_area_m2: 10000      # Minimum debris area
```

---

## Model Architecture

The system uses SegFormer (Xie et al., 2021), a transformer-based semantic segmentation architecture, adapted for 6-band multispectral input:

- **Backbone**: Mix Transformer (MiT-B2)
- **Input**: 6 Sentinel-2 bands at 10m resolution
- **Output**: 2-class segmentation (debris vs. non-debris)
- **Modification**: First projection layer adapted for N-band input

### Sentinel-2 Bands Used

| Band | Name | Wavelength | Resolution | Purpose |
|------|------|------------|------------|---------|
| B2 | Blue | 490nm | 10m | Water penetration |
| B3 | Green | 560nm | 10m | Debris detection |
| B4 | Red | 665nm | 10m | Debris detection |
| B8 | NIR | 842nm | 10m | Vegetation/debris separation |
| B11 | SWIR1 | 1610nm | 20m | Material discrimination |
| B12 | SWIR2 | 2190nm | 20m | Material discrimination |

---

## API Usage

```python
from src.inference import MarineDebrisPredictor

# Initialize predictor
predictor = MarineDebrisPredictor(
    model_path="outputs/models/best_model.pth",
    device="mps"  # or "cuda", "cpu"
)

# Run prediction
results = predictor.predict(
    image_path="path/to/scene.tif",
    output_dir="outputs/predictions/"
)

# Access results
heatmap = results["probability_map"]      # numpy array
hotspots = results["hotspots"]            # GeoDataFrame
metadata = results["metadata"]            # dict with CRS, transform
```

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html
```

---

## Performance Considerations

### Apple Silicon (M4 Max)

The code automatically detects and uses MPS acceleration:

```python
# Automatic device selection
device = "mps" if torch.backends.mps.is_available() else "cpu"
```

Recommended settings for M4 Max:
- Batch size: 8-16
- Number of workers: 4

### NVIDIA GPU

For CUDA-enabled systems, the code will automatically use GPU acceleration.

---

## Citation

If you use this work or the MARIDA dataset, please cite:

```bibtex
@article{kikaki2022marida,
  title={MARIDA: A benchmark for Marine Debris detection from Sentinel-2 remote sensing data},
  author={Kikaki, Katerina and Kakogeorgiou, Ioannis and Mikeli, Paraskevi and Raitsos, Dionysios E and Karantzalos, Konstantinos},
  journal={PLoS ONE},
  volume={17},
  number={1},
  pages={e0262247},
  year={2022},
  publisher={Public Library of Science}
}

@inproceedings{xie2021segformer,
  title={SegFormer: Simple and efficient design for semantic segmentation with transformers},
  author={Xie, Enze and Wang, Wenhai and Yu, Zhiding and Anandkumar, Anima and Alvarez, Jose M and Luo, Ping},
  booktitle={Advances in Neural Information Processing Systems},
  year={2021}
}
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- MARIDA dataset creators (Kikaki et al.)
- European Space Agency for Sentinel-2 data
- Hugging Face for transformer implementations

---

## Contact

For questions or issues, please open a GitHub issue or contact the development team.
