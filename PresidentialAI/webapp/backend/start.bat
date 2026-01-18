@echo off
echo Starting OceanGuard AI Backend...
echo.
cd /d %~dp0
python -m uvicorn api:app --reload --host 0.0.0.0 --port 8000
pause

