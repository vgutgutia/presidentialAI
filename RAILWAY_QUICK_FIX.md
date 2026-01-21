# Railway Deployment Quick Fix

## Most Common Issues

### 1. Missing Dependencies ✅ FIXED
Updated `requirements.txt` to include:
- numpy
- Pillow
- rasterio
- scipy

### 2. Railway Configuration

**In Railway Dashboard:**
1. Go to your service → Settings
2. Set **Root Directory** to: `backend`
3. **Start Command** should be: `python api.py` (or leave empty - Railway will use Procfile)
4. **Build Command**: Leave empty (auto-detects)

### 3. Environment Variables

Add these in Railway → Variables:
- `PORT=8000` (Railway sets this automatically, but you can set it)
- `PYTHONUNBUFFERED=1` (for better logs)

### 4. Check Logs

After deployment, check Railway logs for errors:
- Go to Deployments → Click latest deployment → View Logs
- Look for:
  - Import errors (missing packages)
  - Path errors (file not found)
  - Port binding errors

## Quick Test

If deployment succeeds, test the health endpoint:
```bash
curl https://your-app.railway.app/health
```

Should return:
```json
{"status":"healthy","model_loaded":true,"model_type":"..."}
```

## If Still Failing

Share the error message from Railway logs and I can help debug further!

Common error patterns:
- `ModuleNotFoundError` → Missing dependency in requirements.txt
- `FileNotFoundError` → Path issue, check root directory setting
- `Port already in use` → Railway handles this automatically
- `rasterio` build fails → May need system libraries (see RAILWAY_TROUBLESHOOTING.md)
