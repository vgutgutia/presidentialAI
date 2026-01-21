# Railway Deployment Troubleshooting Guide

## Common Issues and Fixes

### Issue 1: Missing Dependencies

**Error:** `ModuleNotFoundError: No module named 'numpy'` or similar

**Fix:** The `requirements.txt` has been updated to include all necessary dependencies. Make sure Railway is using the correct requirements file.

**Solution:**
1. Verify `backend/requirements.txt` includes all dependencies
2. In Railway dashboard, check that the build is using the correct requirements file
3. If Railway auto-detects Python, it should use `requirements.txt` automatically

### Issue 2: Path Issues

**Error:** `FileNotFoundError` or path-related errors

**Fix:** The code uses relative paths. Make sure Railway is set to use `backend` as the root directory.

**Solution:**
1. In Railway dashboard → Settings → Service
2. Set **Root Directory** to `backend`
3. Or update the start command to `cd backend && python api.py`

### Issue 3: Port Configuration

**Error:** Service not starting or port binding issues

**Fix:** The code now uses the `PORT` environment variable that Railway provides.

**Solution:**
1. In Railway, the `PORT` environment variable is automatically set
2. The code reads: `port = int(os.environ.get("PORT", 8000))`
3. No manual configuration needed

### Issue 4: Build Fails on Rasterio

**Error:** `rasterio` installation fails (common on Railway)

**Fix:** Rasterio requires system libraries. Railway's Nixpacks should handle this, but if not:

**Solution:**
1. Create `backend/nixpacks.toml`:
```toml
[phases.setup]
nixPkgs = ['gdal', 'geos', 'proj']
```

2. Or use a different approach - make rasterio optional in the code (already done with try/except)

### Issue 5: Model File Not Found

**Error:** Model file doesn't exist at expected path

**Fix:** The model file might not be in the repository (too large for Git).

**Solution Options:**

**Option A: Include Model in Repo (if < 100MB)**
```bash
git add PresidentialAI/outputs/models/improved_model.pth
git commit -m "Add model file"
git push
```

**Option B: Use External Storage**
1. Upload model to cloud storage (AWS S3, Google Cloud, etc.)
2. Download on startup in `api.py`:
```python
import os
import requests
from pathlib import Path

MODEL_URL = os.environ.get("MODEL_URL", "")
if MODEL_URL and not MODEL_PATH.exists():
    print("Downloading model...")
    response = requests.get(MODEL_URL)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)
```

**Option C: Use Railway Volumes**
1. In Railway dashboard → Volumes
2. Create a volume and mount it
3. Store model file in the volume

### Issue 6: Python Version Mismatch

**Error:** Python version issues

**Fix:** Specify Python version explicitly.

**Solution:**
1. Create `backend/runtime.txt` with: `python-3.11`
2. Railway will use this version

### Issue 7: Memory Issues

**Error:** Out of memory errors

**Fix:** Railway free tier has limited memory. Optimize the code.

**Solution:**
1. Don't load the deep learning model if using spectral detection
2. Use lighter dependencies
3. Consider upgrading Railway plan if needed

## Step-by-Step Railway Setup

### 1. Create New Project
- Go to railway.app
- Click "New Project"
- Select "Deploy from GitHub repo"
- Choose your repository

### 2. Configure Service
- **Root Directory:** `backend`
- **Start Command:** `python api.py` (or leave default if root is set correctly)
- **Build Command:** Leave default (auto-detects Python)

### 3. Set Environment Variables
- `PORT=8000` (Railway sets this automatically, but you can set it)
- `PYTHONUNBUFFERED=1` (for better logging)

### 4. Deploy
- Railway will automatically:
  1. Detect Python
  2. Install dependencies from `requirements.txt`
  3. Run the start command
  4. Expose the service on a public URL

### 5. Check Logs
- Go to Railway dashboard → Deployments → View Logs
- Look for any errors during build or runtime

## Quick Fixes Checklist

- [ ] `requirements.txt` includes all dependencies (numpy, Pillow, rasterio, scipy)
- [ ] Root directory is set to `backend` in Railway
- [ ] Start command is `python api.py` (or `cd backend && python api.py`)
- [ ] `runtime.txt` specifies Python 3.11
- [ ] Model file is accessible (in repo or external storage)
- [ ] Environment variables are set correctly
- [ ] CORS allows your Vercel domain

## Testing Locally Before Deploying

Test that the backend works locally first:

```bash
cd backend
pip install -r requirements.txt
python api.py
```

Then test the health endpoint:
```bash
curl http://localhost:8000/health
```

If this works locally, it should work on Railway.

## Getting Help

If deployment still fails:
1. Check Railway logs for specific error messages
2. Share the error logs
3. Verify all files are committed to Git
4. Check that Railway can access your GitHub repo

## Alternative: Use Render Instead

If Railway continues to have issues, try Render (render.com):
1. Similar free tier
2. Often easier Python deployment
3. Better documentation for FastAPI

See `VERCEL_DEPLOYMENT.md` for Render instructions.
