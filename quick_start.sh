#!/bin/bash
# Quick start script for OceanGuard AI Backend

echo "🌊 OceanGuard AI - Quick Start"
echo "================================"
echo ""

# Check if we're in the right directory
if [ ! -f "backend/api.py" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 not found"
    exit 1
fi

echo "✓ Python found: $(python3 --version)"
echo ""

# Check if dependencies are installed
echo "Checking dependencies..."
python3 -c "import fastapi, uvicorn, rasterio, numpy, scipy, matplotlib" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Some dependencies are missing. Installing..."
    cd backend
    pip3 install -q -r requirements.txt
    cd ..
    echo "✓ Dependencies installed"
else
    echo "✓ All dependencies found"
fi

echo ""
echo "Starting backend API server..."
echo "Server will be available at: http://localhost:8000"
echo "API docs at: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

cd backend
python3 api.py
