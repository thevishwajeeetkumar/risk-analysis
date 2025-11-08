from __future__ import annotations

import os
import sys
from pathlib import Path

from fastapi import FastAPI

# Ensure the Interview package is importable when running on Vercel
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERVIEW_DIR = PROJECT_ROOT / "server"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(INTERVIEW_DIR) not in sys.path:
    sys.path.insert(0, str(INTERVIEW_DIR))

from server.api.main import app as fastapi_app  # noqa: E402


# Expose FastAPI app for Vercel's Python runtime
app: FastAPI = fastapi_app


@app.get("/vercel/health")
async def vercel_health() -> dict[str, str]:
    """
    Lightweight health endpoint for Vercel health checks.
    """
    return {"status": "ok", "environment": os.getenv("VERCEL", "local")}


