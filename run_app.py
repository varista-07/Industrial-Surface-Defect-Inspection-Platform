"""
run_app.py
Single-command launcher for Industrial Surface Defect Inspection Platform:
Runs the FastAPI application on http://localhost:8000 using Uvicorn.
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath("."))
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import uvicorn

if __name__ == "__main__":
    print("=" * 65)
    print("LAUNCHING INDUSTRIAL SURFACE DEFECT INSPECTION PLATFORM")
    print("Address: http://localhost:8000")
    print("Documentation API: http://localhost:8000/docs")
    print("=" * 65)
    
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        log_level="info",
        reload=False
    )
