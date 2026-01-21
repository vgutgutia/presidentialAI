# Vercel Deployment Guide

This guide will help you deploy OceanGuard AI to Vercel (frontend) and Railway (backend).

## Prerequisites

1. GitHub account
2. Vercel account (free) - Sign up at https://vercel.com
3. Railway account (free) - Sign up at https://railway.app (for backend)

---

## Step 1: Prepare Your Repository

### 1.1 Initialize Git (if not already done)

```bash
cd /Users/home/data_science_club/presidentialAI
git init
git add .
git commit -m "Initial commit for Vercel deployment"
```

### 1.2 Create .gitignore (if not exists)

Create or update `.gitignore`:

```
# Dependencies
node_modules/
__pycache__/
*.pyc
*.pyo

# Environment
.env
.env.local
.env*.local

# Logs
*.log
backend/backend.log
frontend/frontend.log

# Build outputs
.next/
dist/
build/

# OS
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/

# Model files (too large for git)
PresidentialAI/outputs/models/*.pth
PresidentialAI/outputs/models/*.json
!PresidentialAI/outputs/models/.gitkeep
```

### 1.3 Push to GitHub

```bash
# Create a new repository on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/presidentialAI.git
git branch -M main
git push -u origin main
```

---

## Step 2: Deploy Frontend to Vercel

### 2.1 Connect to Vercel

1. Go to https://vercel.com and sign in
2. Click "Add New Project"
3. Import your GitHub repository
4. Select the repository: `presidentialAI`

### 2.2 Configure Project Settings

**Root Directory:** Set to `frontend`

**Build Settings:**
- Framework Preset: Next.js
- Build Command: `npm run build` (or leave default)
- Output Directory: `.next` (default)
- Install Command: `npm install` (default)

**Environment Variables:**
Add these in Vercel dashboard → Settings → Environment Variables:

```
BACKEND_URL=https://your-backend-url.railway.app
```

(We'll get the backend URL in Step 3)

### 2.3 Deploy

Click "Deploy" and wait for the build to complete. Vercel will give you a URL like:
`https://presidential-ai.vercel.app`

---

## Step 3: Deploy Backend to Railway

### 3.1 Prepare Backend for Railway

Create `backend/railway.json`:

```json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "cd backend && python3 api.py",
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}
```

### 3.2 Create Railway Procfile

Create `backend/Procfile`:

```
web: cd backend && python3 api.py
```

### 3.3 Create Railway Requirements

Create `backend/railway-requirements.txt` (includes all dependencies):

```txt
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
python-multipart>=0.0.6
pydantic>=2.5.0
anthropic>=0.40.0
torch>=2.1.0
numpy>=1.24.0
rasterio>=1.3.9
scipy>=1.11.0
matplotlib>=3.8.0
Pillow>=10.0.0
```

### 3.4 Deploy to Railway

1. Go to https://railway.app and sign in
2. Click "New Project"
3. Select "Deploy from GitHub repo"
4. Select your repository
5. Railway will auto-detect Python

**Configure Service:**
- Root Directory: `backend`
- Start Command: `python3 api.py`
- Port: Railway will auto-assign (use `$PORT` environment variable)

**Environment Variables:**
Add in Railway dashboard:

```
PORT=8000
PYTHONUNBUFFERED=1
```

**Important:** Update `backend/api.py` to use Railway's port:

```python
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
```

### 3.5 Get Backend URL

After deployment, Railway will give you a URL like:
`https://presidential-ai-backend.railway.app`

Copy this URL and update the `BACKEND_URL` environment variable in Vercel.

---

## Step 4: Update Backend Code for Production

### 4.1 Update api.py for Railway

The backend needs to handle the PORT environment variable. Update the bottom of `backend/api.py`:

```python
if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
```

### 4.2 Update CORS Settings

In `backend/api.py`, update CORS to allow your Vercel domain:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://your-app.vercel.app",  # Add your Vercel URL
        "https://*.vercel.app",  # Allow all Vercel preview deployments
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## Step 5: Model Files

### Option A: Include Model in Repository (if < 100MB)

If your model file is small enough, commit it:

```bash
git add PresidentialAI/outputs/models/improved_model.pth
git commit -m "Add trained model"
git push
```

### Option B: Use External Storage (Recommended for large models)

1. Upload model to cloud storage (AWS S3, Google Cloud Storage, etc.)
2. Download model on Railway startup
3. Or use Railway's volume storage

Create `backend/download_model.py`:

```python
import os
import requests
from pathlib import Path

MODEL_URL = os.environ.get("MODEL_URL", "")
MODEL_PATH = Path(__file__).parent.parent / "PresidentialAI" / "outputs" / "models" / "improved_model.pth"

if MODEL_URL and not MODEL_PATH.exists():
    print(f"Downloading model from {MODEL_URL}...")
    response = requests.get(MODEL_URL)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)
    print("Model downloaded successfully")
```

---

## Step 6: Final Configuration

### 6.1 Update Vercel Environment Variables

Go to Vercel Dashboard → Your Project → Settings → Environment Variables:

```
BACKEND_URL=https://your-backend.railway.app
```

### 6.2 Redeploy

After updating environment variables, trigger a new deployment in Vercel.

---

## Step 7: Test Your Deployment

1. Visit your Vercel URL: `https://your-app.vercel.app`
2. Navigate to `/analyze`
3. Try uploading an image or using a sample
4. Check browser console for any errors
5. Check Railway logs for backend errors

---

## Troubleshooting

### Frontend Issues

**Build Fails:**
- Check Vercel build logs
- Ensure all dependencies are in `package.json`
- Check Node.js version (Vercel uses 18.x by default)

**API Routes Not Working:**
- Verify `BACKEND_URL` environment variable is set
- Check CORS settings in backend
- Check browser network tab for errors

### Backend Issues

**Backend Won't Start:**
- Check Railway logs
- Verify Python version (Railway uses 3.11 by default)
- Check that all dependencies are in `railway-requirements.txt`

**Model Not Loading:**
- Verify model file path is correct
- Check file permissions
- Ensure model file is accessible

**CORS Errors:**
- Update CORS origins in `backend/api.py`
- Include your Vercel domain

---

## Alternative: Deploy Backend to Render

If Railway doesn't work, try Render (https://render.com):

1. Create new Web Service
2. Connect GitHub repository
3. Settings:
   - Build Command: `pip install -r backend/requirements.txt`
   - Start Command: `cd backend && python3 api.py`
   - Environment: Python 3
4. Add environment variables
5. Deploy

---

## Cost

- **Vercel**: Free tier includes:
  - Unlimited personal projects
  - 100GB bandwidth/month
  - Automatic HTTPS
  
- **Railway**: Free tier includes:
  - $5 credit/month
  - 500 hours of usage
  - 100GB bandwidth

For production use, you may need to upgrade to paid plans.

---

## Next Steps

1. Set up custom domain (optional)
2. Configure monitoring and alerts
3. Set up CI/CD for automatic deployments
4. Add error tracking (Sentry, etc.)

---

## Quick Reference

**Frontend URL:** `https://your-app.vercel.app`
**Backend URL:** `https://your-backend.railway.app`
**Vercel Dashboard:** https://vercel.com/dashboard
**Railway Dashboard:** https://railway.app/dashboard
