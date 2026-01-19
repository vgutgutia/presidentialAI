# Quick Start Training Guide

## ✅ Model Code Fixed
The preprocessing module has been created and the model code is ready for training.

## 🚀 Quick Training (Synthetic Data)

I've created a quick training script that works without the MARIDA dataset:

```bash
cd PresidentialAI
python3 scripts/train_quick.py
```

This script:
- ✅ Generates synthetic training data automatically
- ✅ Works without downloading MARIDA dataset
- ✅ Trains a simple U-Net model
- ✅ Saves model to `outputs/models/quick_trained_model.pth`

## 📦 Install Dependencies First

Before training, install PyTorch and dependencies:

### Option 1: Using pip (Recommended)
```bash
cd PresidentialAI
pip3 install torch torchvision numpy tqdm
```

### Option 2: Using requirements file
```bash
cd PresidentialAI
pip3 install -r scripts/training_requirements.txt
```

### Option 3: For Apple Silicon (M1/M2/M3/M4)
```bash
# PyTorch with MPS support
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip3 install numpy tqdm
```

## 🎯 Full Training (With MARIDA Dataset)

For training with the real MARIDA dataset:

1. **Download MARIDA dataset:**
   ```bash
   cd PresidentialAI
   # Follow instructions from scripts/download_marida.py
   # Or manually download from: https://github.com/marine-debris/marine-debris.github.io
   ```

2. **Run full training:**
   ```bash
   python3 scripts/train_deep_model.py
   ```

## 📊 Training Scripts Available

1. **`train_quick.py`** - Quick training with synthetic data (no dataset needed)
2. **`train_deep_model.py`** - Full U-Net training with MARIDA dataset
3. **`train.py`** - SegFormer training (requires dataset)

## ⚡ Quick Start Command

```bash
cd PresidentialAI
pip3 install torch torchvision numpy tqdm
python3 scripts/train_quick.py
```

The training will:
- Create synthetic satellite imagery
- Train for 10 epochs (quick)
- Save best model to `outputs/models/quick_trained_model.pth`
- Show progress with progress bars

## 🔧 Troubleshooting

### Permission Errors
If you get permission errors installing packages:
```bash
# Use --user flag
pip3 install --user torch torchvision numpy tqdm

# Or use a virtual environment
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision numpy tqdm
```

### No GPU Available
The script automatically detects:
- CUDA (NVIDIA GPU)
- MPS (Apple Silicon)
- CPU (fallback)

Training will work on any device, just slower on CPU.

## 📝 Next Steps After Training

Once training completes:
1. Model saved to: `outputs/models/quick_trained_model.pth`
2. Use for inference: `python scripts/predict.py --model outputs/models/quick_trained_model.pth --input <image.tif>`
3. Or integrate into backend API

---

**Ready to train?** Run:
```bash
cd PresidentialAI && python3 scripts/train_quick.py
```
