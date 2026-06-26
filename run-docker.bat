@echo off
echo ===================================================
echo   Starting Librarian RAG Application in Docker
echo ===================================================

if not exist .env (
    echo [ERROR] .env file not found!
    echo Please copy .env.example to .env and configure your RAG_GOOGLE_API_KEY.
    echo.
    echo Creating .env from .env.example automatically...
    copy .env.example .env
    echo Please edit the .env file with your credentials before continuing.
    pause
    exit /b 1
)

echo Building and starting containers...
docker compose up --build -d

echo.
echo ===================================================
echo   Librarian container services started successfully!
echo   Headless API: http://localhost:8000/
echo   If you configure RAG_SERVE_UI=True in .env:
echo   Web Interface: http://localhost:8000/
echo ===================================================
echo To stop the application, run: docker compose down
echo.
pause
