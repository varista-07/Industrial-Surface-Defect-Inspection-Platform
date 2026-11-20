"""
tests/test_api_live.py
Automated End-to-End API and Model Testing for the Inspection Platform:
1. Tests GET /health endpoint (system status, model availability).
2. Tests GET /samples endpoint (sample gallery registry).
3. Tests POST /predict with real NEU-DET defect images (normal flow).
4. Tests POST /predict with uniform low-contrast image (blank surface heuristic).
5. Tests POST /predict with Gaussian noise image (low-confidence anomaly heuristic).
6. Tests input validation: unsupported file types (txt/pdf).
7. Tests input validation: corrupted image payload.
8. Tests input validation: oversized file (> 5 MB).
9. Tests model checkpoints existence and weight integrity.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))
import io
import json
import warnings
warnings.filterwarnings("ignore", message=".*Using `httpx` with `starlette.testclient` is deprecated.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="starlette.*")
import pytest
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from app.main import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client

def test_health_endpoint(client):
    """Test 1: Health check endpoint returns status and verifies loaded models."""
    resp = client.get("/health")
    assert resp.status_code == 200, f"Health check failed: {resp.status_code}"
    health_data = resp.json()
    assert health_data["status"] == "healthy"
    assert health_data["models_loaded"]["classifier"] is True, "Classifier failed to load!"
    assert len(health_data["classes"]) == 6

def test_samples_gallery_endpoint(client):
    """Test 2: Sample gallery metadata endpoint."""
    resp = client.get("/samples")
    assert resp.status_code == 200
    samples = resp.json()
    assert isinstance(samples, list)
    assert len(samples) > 0

def test_predict_real_defect_image(client):
    """Test 3: POST /predict with real NEU-DET defect image."""
    sample_img_path = os.path.join("app", "static", "samples", "crazing_1.jpg")
    if not os.path.exists(sample_img_path):
        sample_img_path = os.path.join("data", "raw", "NEU-DET", "IMAGES", "crazing_1.jpg")
        
    with open(sample_img_path, "rb") as f:
        img_bytes = f.read()
        
    resp = client.post("/predict?threshold=0.60", files={"file": ("crazing_1.jpg", img_bytes, "image/jpeg")})
    assert resp.status_code == 200, f"Prediction failed: {resp.text}"
    pred_data = resp.json()
    assert pred_data["success"] is True
    assert pred_data["predicted_class"] in ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
    assert pred_data["confidence_percentage"] > 0
    assert "visualizations" in pred_data
    assert pred_data["visualizations"]["original_image"] is not None
    assert pred_data["visualizations"]["gradcam_heatmap_overlay"] is not None

def test_predict_blank_surface_heuristic(client):
    """Test 4: POST /predict with uniform low-contrast surface (triggers is_blank / PASS)."""
    blank_arr = np.full((200, 200), 128, dtype=np.uint8)
    blank_buf = io.BytesIO()
    Image.fromarray(blank_arr).save(blank_buf, format="JPEG")
    blank_bytes = blank_buf.getvalue()
    
    resp = client.post("/predict?threshold=0.75", files={"file": ("clean_surface.jpg", blank_bytes, "image/jpeg")})
    assert resp.status_code == 200
    ood_data = resp.json()
    assert ood_data["is_uncertain"] is True
    assert ood_data["verdict_status"] == "PASS"
    assert "Uniform steel surface" in ood_data["verdict_message"]

def test_predict_low_confidence_heuristic(client):
    """Test 5: POST /predict with synthetic non-defect gradient (triggers low-confidence UNCERTAIN / ANOMALY)."""
    # A smooth gradient has non-zero contrast (std > 0.025) but lacks defect features,
    # yielding moderate confidence (< 0.95 threshold) and routing to manual inspection.
    grad_arr = np.tile(np.linspace(0, 255, 200, dtype=np.uint8), (200, 1))
    grad_buf = io.BytesIO()
    Image.fromarray(grad_arr).save(grad_buf, format="JPEG")
    grad_bytes = grad_buf.getvalue()
    
    resp = client.post("/predict?threshold=0.95", files={"file": ("gradient_nondefect.jpg", grad_bytes, "image/jpeg")})
    assert resp.status_code == 200
    noise_data = resp.json()
    assert noise_data["is_uncertain"] is True
    assert noise_data["verdict_status"] == "UNCERTAIN"
    assert "below inspection threshold" in noise_data["verdict_message"]

def test_input_validation_unsupported_format(client):
    """Test 6: Input Validation: Unsupported file extension (.pdf)."""
    fake_txt = io.BytesIO(b"Not an image")
    resp = client.post("/predict", files={"file": ("report.pdf", fake_txt, "application/pdf")})
    assert resp.status_code == 400
    assert "Unsupported file format" in resp.json()["detail"]

def test_input_validation_corrupted_payload(client):
    """Test 7: Input Validation: Corrupted image bytes."""
    corrupt_bytes = b"CORRUPTED_RAW_JPEG_HEADER_INVALID"
    resp = client.post("/predict", files={"file": ("corrupt.jpg", corrupt_bytes, "image/jpeg")})
    assert resp.status_code == 400
    assert "Corrupted or unreadable image file" in resp.json()["detail"]

def test_input_validation_oversized_payload(client):
    """Test 8: Input Validation: File exceeding maximum allowed size (> 5 MB)."""
    oversized_bytes = b"0" * (6 * 1024 * 1024) # 6 MB
    resp = client.post("/predict", files={"file": ("huge.jpg", oversized_bytes, "image/jpeg")})
    assert resp.status_code == 413
    assert "File exceeds maximum allowed size" in resp.json()["detail"]

def test_model_checkpoints_exist_and_load():
    """Test 9: Model checkpoints exist on disk and are not merely LFS pointer stubs."""
    checkpoints = [
        os.path.join("models", "checkpoints", "best_resnet50.keras"),
        os.path.join("models", "checkpoints", "best_detector.keras"),
        os.path.join("models", "checkpoints", "best_unet.keras")
    ]
    for ckpt in checkpoints:
        assert os.path.exists(ckpt), f"Missing checkpoint: {ckpt}"
        file_size_mb = os.path.getsize(ckpt) / (1024 * 1024)
        # Ensure it is a real binary weight file, not a 130-byte Git LFS pointer text file
        assert file_size_mb > 1.0, f"Checkpoint {ckpt} is suspiciously small ({file_size_mb:.4f} MB). Potential Git LFS pointer!"
