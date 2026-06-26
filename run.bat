@echo off
echo ===================================================
echo   Starting ID-Rag FastAPI Server and UI
echo ===================================================
echo Installing dependencies from requirements.txt...
pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo [WARNING] Dependency installation failed. Trying to start server anyway...
)
echo Starting FastAPI application...
python main.py
pause
