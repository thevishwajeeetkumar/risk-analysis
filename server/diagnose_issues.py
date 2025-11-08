#!/usr/bin/env python3
"""
Diagnostic script to check all potential issues with the ECL backend
"""

import os
import sys
import subprocess
from pathlib import Path

print("=" * 60)
print("ECL Backend Diagnostic Tool")
print("=" * 60)

# Check Python version
print("\n[1] Python Version Check:")
print(f"Python: {sys.version}")
print(f"Executable: {sys.executable}")

# Check current directory
print("\n[2] Working Directory:")
print(f"Current: {os.getcwd()}")
script_dir = Path(__file__).parent
print(f"Script directory: {script_dir}")

# Check virtual environment
print("\n[3] Virtual Environment:")
venv_python = script_dir / "myvenv" / "Scripts" / "python.exe"
if venv_python.exists():
    print(f"[SUCCESS] Virtual environment found at: {venv_python}")
else:
    print(f"[ERROR] Virtual environment NOT found at: {venv_python}")

# Check .env file
print("\n[4] Environment Configuration:")
env_file = script_dir / ".env"
env_example = script_dir / ".env.example"

if env_file.exists():
    print(f"[SUCCESS] .env file found")
    # Check if it has required variables
    with open(env_file, 'r') as f:
        content = f.read()
        required_vars = ["DATABASE_URL", "JWT_SECRET_KEY", "OPENAI_API_KEY", "PINECONE_API_KEY"]
        for var in required_vars:
            if var in content:
                print(f"  [OK] {var} is set")
            else:
                print(f"  [MISSING] {var} is not set")
else:
    print(f"[ERROR] .env file NOT found")
    if env_example.exists():
        print(f"  [INFO] .env.example exists - copy it to .env")

# Check port 8000
print("\n[5] Port 8000 Status:")
try:
    result = subprocess.run(
        ["netstat", "-ano"], 
        capture_output=True, 
        text=True, 
        shell=True
    )
    if ":8000" in result.stdout:
        print("[WARNING] Port 8000 is in use")
        lines = [line for line in result.stdout.split('\n') if ':8000' in line]
        for line in lines[:3]:  # Show first 3 matches
            print(f"  {line.strip()}")
    else:
        print("[SUCCESS] Port 8000 is available")
except Exception as e:
    print(f"[ERROR] Could not check port status: {e}")

# Check package installation
print("\n[6] Required Packages:")
if venv_python.exists():
    try:
        # Use the venv pip to check packages
        venv_pip = script_dir / "myvenv" / "Scripts" / "pip.exe"
        result = subprocess.run(
            [str(venv_pip), "list"], 
            capture_output=True, 
            text=True
        )
        installed_packages = result.stdout.lower()
        
        required_packages = [
            "fastapi", "uvicorn", "pandas", "numpy", 
            "sqlalchemy", "asyncpg", "python-jose", 
            "passlib", "python-multipart", "slowapi",
            "langchain", "openai", "pinecone"
        ]
        
        for package in required_packages:
            if package in installed_packages:
                print(f"  [OK] {package}")
            else:
                print(f"  [MISSING] {package}")
    except Exception as e:
        print(f"[ERROR] Could not check packages: {e}")
else:
    print("[SKIP] Virtual environment not found")

# Check database connection
print("\n[7] Database Connection:")
try:
    # Try to import and check database URL format
    from dotenv import load_dotenv
    load_dotenv(env_file)
    
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        if "neon.tech" in db_url:
            print("[SUCCESS] Neon database URL detected")
        else:
            print("[INFO] Non-Neon database URL detected")
        
        # Mask password for display
        if "@" in db_url:
            parts = db_url.split("@")
            masked = parts[0].split(":")[-2] + ":****@" + parts[1]
            print(f"  URL format: {masked[:50]}...")
    else:
        print("[ERROR] DATABASE_URL not found in environment")
except Exception as e:
    print(f"[ERROR] Could not check database: {e}")

print("\n" + "=" * 60)
print("Diagnostic Summary:")
print("=" * 60)

# Provide action items
print("\nRecommended Actions:")
if not venv_python.exists():
    print("1. Create virtual environment: python -m venv myvenv")
    print("2. Install packages: .\\myvenv\\Scripts\\pip install -r requirements.txt")
elif not env_file.exists():
    print("1. Copy .env.example to .env")
    print("2. Fill in the required API keys and database URL")
else:
    print("Everything looks good! Try running:")
    print("  .\\start_backend.ps1")
    print("or")
    print("  .\\myvenv\\Scripts\\python -m uvicorn api.main:app --reload")
