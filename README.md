# OceanGuard AI - Marine Debris Detection System

**Satellite-Based Trash Hotspot Detection Using AI**

*Presidential AI Challenge Submission*

---

## 🌊 What It Does

OceanGuard AI detects floating marine debris (plastic, trash, waste) in ocean waters using Sentinel-2 satellite imagery. Upload satellite images and get:

- 🎯 **Hotspot locations** with confidence scores
- 🗺️ **Heatmap visualization** of debris probability
- 📍 **GPS coordinates** for each detection
- 📊 **Exportable data** in multiple formats

---

## 🚀 Quick Start

### 1. Start the Backend API
```bash
cd backend
pip install -r requirements.txt
python api.py
# Runs on http://localhost:8000
```

### 2. Start the Frontend
```bash
cd frontend
npm install
npm run dev
# Runs on http://localhost:3000
```

### 3. Open the App
Navigate to **http://localhost:3000/analyze** and:
- Click a **sample image** to test immediately
- Or **upload** your own Sentinel-2 GeoTIFF

---

## 📁 Project Structure

```
PresidentialAI/
├── backend/                 # FastAPI backend
│   └── api.py              # Main API server
├── frontend/               # Next.js frontend
│   └── src/app/           # React components
├── PresidentialAI/        # ML model & data
│   ├── data/marida/       # MARIDA dataset
│   ├── outputs/models/    # Trained models
│   └── scripts/           # Training scripts
├── TRAINING_GUIDE.md      # ⭐ Guide for training better models
└── README.md              # This file
```

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [**TRAINING_GUIDE.md**](TRAINING_GUIDE.md) | Complete guide for training improved deep learning models |
| [PresidentialAI/README.md](PresidentialAI/README.md) | Detailed technical documentation |

---

## 🎓 Training a Better Model

The current model uses spectral anomaly detection (fast, but limited accuracy).

**To train a deep learning model with better accuracy:**

1. Read [TRAINING_GUIDE.md](TRAINING_GUIDE.md)
2. Requirements: GPU with 8GB+ VRAM (RTX 3060 or better)
3. Run the training script:
   ```bash
   cd PresidentialAI
   python scripts/train_deep_model.py
   ```

Expected improvement: F1 score from ~0.20 to 0.50-0.70

---

## 🔧 Current Model Performance

| Metric | Value |
|--------|-------|
| Detection Method | Spectral Anomaly (FDI, NDWI, NDVI) |
| Processing Time | ~300ms per image |
| Sensitivity Range | Adjustable 0.1-0.9 |
| Input Format | 11-band Sentinel-2 GeoTIFF |

---

## 📊 Dataset

Uses the **MARIDA** (Marine Debris Archive) dataset:
- 63 Sentinel-2 scenes
- 256×256 pixel patches
- 15 class labels (debris, water, algae, ships, etc.)

Location: `PresidentialAI/data/marida/patches/`

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📜 License

MIT License - see LICENSE file for details.

---

## 🙏 Acknowledgments

- **MARIDA Dataset**: Kikaki et al., 2022
- **Presidential AI Challenge** organizers
- **Sentinel-2** / ESA for satellite imagery

---

*Built for the Presidential AI Challenge 2026*

