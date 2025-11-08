@echo off
echo ========================================
echo ECL Risk Analysis Backend Startup Script
echo ========================================

:: Kill any existing Python processes
echo.
echo [1/4] Cleaning up existing processes...
taskkill /F /IM python.exe 2>nul
if %errorlevel% == 0 (
    echo [SUCCESS] Killed existing Python processes
) else (
    echo [INFO] No existing Python processes found
)

:: Set the correct working directory
echo.
echo [2/4] Setting working directory...
cd /d "%~dp0"
echo [SUCCESS] Working directory: %cd%

:: Check if .env file exists
echo.
echo [3/4] Checking environment configuration...
if exist .env (
    echo [SUCCESS] .env file found
) else (
    echo [ERROR] .env file not found!
    if exist .env.example (
        echo [INFO] Copying .env.example to .env...
        copy .env.example .env
        echo [SUCCESS] Created .env from template
    ) else (
        echo [ERROR] No .env.example found either!
        pause
        exit /b 1
    )
)

:: Activate virtual environment and start server
echo.
echo [4/4] Starting backend server...
if exist myvenv\Scripts\python.exe (
    echo [SUCCESS] Virtual environment found
    echo.
    echo Starting server on http://127.0.0.1:8000
    echo Press Ctrl+C to stop the server
    echo ========================================
    call myvenv\Scripts\activate
    python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
) else (
    echo [ERROR] Virtual environment not found at myvenv\Scripts\python.exe
    echo Please run: python -m venv myvenv
    echo Then run: myvenv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

pause
