# Quick Vercel Deployment Guide

## Prerequisites
- GitHub account
- Vercel account (free at vercel.com)
- Railway account (free at railway.app) - for backend

## Step 1: Push to GitHub

```bash
cd /Users/home/data_science_club/presidentialAI
git init
git add .
git commit -m "Ready for deployment"
git remote add origin https://github.com/YOUR_USERNAME/presidentialAI.git
git push -u origin main
```

## Step 2: Deploy Frontend to Vercel

1. Go to https://vercel.com
2. Click "Add New Project"
3. Import your GitHub repo
4. **Important Settings:**
   - Root Directory: `frontend`
   - Framework: Next.js (auto-detected)
   - Build Command: `npm run build`
   - Output Directory: `.next`
5. **Environment Variables:**
   - Add `BACKEND_URL` (we'll set this after backend deploys)
6. Click "Deploy"

Your frontend will be live at: `https://your-app.vercel.app`

## Step 3: Deploy Backend to Railway

1. Go to https://railway.app
2. Click "New Project" → "Deploy from GitHub"
3. Select your repo
4. **Settings:**
   - Root Directory: `backend`
   - Start Command: `python3 api.py`
5. **Environment Variables:**
   - `PORT=8000` (Railway auto-assigns, but set for clarity)
   - `PYTHONUNBUFFERED=1`
6. Wait for deployment

Railway will give you a URL like: `https://your-backend.railway.app`

## Step 4: Connect Frontend to Backend

1. Go back to Vercel dashboard
2. Settings → Environment Variables
3. Add/Update: `BACKEND_URL=https://your-backend.railway.app`
4. Redeploy (or it will auto-redeploy)

## Step 5: Test

Visit your Vercel URL and test the `/analyze` page!

## Troubleshooting

**CORS Errors:**
- Backend CORS is already configured for `*.vercel.app`
- If issues persist, check Railway logs

**Backend Not Starting:**
- Check Railway logs
- Verify Python dependencies are installed
- Model file must be accessible (include in repo or use external storage)

**Frontend Build Fails:**
- Check Vercel build logs
- Ensure all dependencies in `package.json`

## Notes

- **Model Files**: If `improved_model.pth` is > 100MB, you may need to:
  - Use Git LFS
  - Store on cloud storage and download on startup
  - Use Railway volumes

- **Free Tier Limits:**
  - Vercel: 100GB bandwidth/month
  - Railway: $5 credit/month (~500 hours)

## Next Steps

1. Set up custom domain (optional)
2. Configure monitoring
3. Set up automatic deployments on git push
