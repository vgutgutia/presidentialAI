# 🚀 Application Launch Information

## Services Status

### ✅ Backend API Server
- **Status**: Running
- **URL**: http://127.0.0.1:8000
- **Health Check**: http://127.0.0.1:8000/health
- **Log File**: `backend/backend.log`

### ✅ Frontend Web Application
- **Status**: Running
- **URL**: http://localhost:3000
- **Log File**: `frontend/frontend.log`

## Access the Application

🌐 **Open in your browser**: http://localhost:3000

## Available Pages

1. **Home** (`/`) - Landing page
2. **Dashboard** (`/dashboard`) - Statistics and overview
3. **Analyze** (`/analyze`) - Upload images and run detection
4. **About** (`/about`) - Project information

## API Endpoints

### Backend (FastAPI)
- `GET /health` - Health check
- `POST /predict` - Run debris detection on uploaded image
- `POST /predict-sample` - Run detection on sample image

### Frontend (Next.js API Routes)
- `GET /api/health` - Check backend connection
- `POST /api/predict` - Proxy to backend
- `POST /api/predict-sample` - Proxy to backend

## Stop Services

To stop the servers:

```bash
# Stop backend
pkill -f "python3 api.py"

# Stop frontend
pkill -f "next dev"
```

Or find and kill processes:
```bash
lsof -ti:8000 | xargs kill  # Backend
lsof -ti:3000 | xargs kill  # Frontend
```

## Troubleshooting

### Backend not responding?
- Check `backend/backend.log` for errors
- Verify port 8000 is not in use: `lsof -i:8000`
- Restart: `cd backend && python3 api.py`

### Frontend not loading?
- Check `frontend/frontend.log` for errors
- Verify port 3000 is not in use: `lsof -i:3000`
- Restart: `cd frontend && npm run dev`

### Connection issues?
- Ensure backend is running before frontend
- Check CORS settings in `backend/api.py`
- Verify `BACKEND_URL` in frontend API routes

## Model Information

The backend uses the **Improved Model** for debris detection:
- **Model**: `PresidentialAI/outputs/models/improved_model.pth`
- **F1 Score**: 29.47%
- **Recall**: 67.24%
- **Inference Speed**: 31.49ms per batch

---

**Last Updated**: Application launched successfully! 🎉
