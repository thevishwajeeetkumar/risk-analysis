# ECL Risk Analysis Backend Startup Script

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "ECL Risk Analysis Backend Startup Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Kill any existing Python processes
Write-Host "`n[1/4] Cleaning up existing processes..." -ForegroundColor Yellow
$pythonProcesses = Get-Process -Name python -ErrorAction SilentlyContinue
if ($pythonProcesses) {
    $pythonProcesses | Stop-Process -Force
    Write-Host "[SUCCESS] Killed existing Python processes" -ForegroundColor Green
} else {
    Write-Host "[INFO] No existing Python processes found" -ForegroundColor Gray
}

# Set the correct working directory
Write-Host "`n[2/4] Setting working directory..." -ForegroundColor Yellow
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir
Write-Host "[SUCCESS] Working directory: $PWD" -ForegroundColor Green

# Check if .env file exists
Write-Host "`n[3/4] Checking environment configuration..." -ForegroundColor Yellow
if (Test-Path ".env") {
    Write-Host "[SUCCESS] .env file found" -ForegroundColor Green
} else {
    Write-Host "[ERROR] .env file not found!" -ForegroundColor Red
    if (Test-Path ".env.example") {
        Write-Host "[INFO] Copying .env.example to .env..." -ForegroundColor Gray
        Copy-Item ".env.example" ".env"
        Write-Host "[SUCCESS] Created .env from template" -ForegroundColor Green
    } else {
        Write-Host "[ERROR] No .env.example found either!" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
}

# Check Python version
Write-Host "`n[4/5] Checking Python installation..." -ForegroundColor Yellow
$pythonPath = ".\myvenv\Scripts\python.exe"
if (Test-Path $pythonPath) {
    $pythonVersion = & $pythonPath --version 2>&1
    Write-Host "[SUCCESS] Found $pythonVersion" -ForegroundColor Green
} else {
    Write-Host "[ERROR] Virtual environment not found!" -ForegroundColor Red
    Write-Host "Please run: python -m venv myvenv" -ForegroundColor Yellow
    Write-Host "Then run: .\myvenv\Scripts\pip install -r requirements.txt" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Start the server
Write-Host "`n[5/5] Starting backend server..." -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Server will start on http://127.0.0.1:8000" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan

try {
    & $pythonPath -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
} catch {
    Write-Host "`n[ERROR] Server failed to start: $_" -ForegroundColor Red
    Read-Host "Press Enter to exit"
}
