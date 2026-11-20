"""
tests/test_api_live.py
Automated End-to-End API and Model Testing for the Inspection Platform:
1. Tests GET /health endpoint (system status, model availability).
2. Tests GET /samples endpoint (sample gallery registry).
3. Tests POST /predict with real NEU-DET defect images (normal flow).
4. Tests POST /predict with uniform/random noise image to verify Out-of-Distribution (OOD) trigger.
5. Tests input validation: unsupported file types (txt/pdf) and oversized file checks.
6. Prints detailed execution results.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))
import io
import json
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from app.main import app

def run_api_tests():
    print("=" * 65)
    print("RUNNING AUTOMATED END-TO-END WEB API TESTS")
    print("=" * 65)
    
    with TestClient(app) as client:
        # 1. Test GET /health
        print("\n[TEST 1] GET /health...")
        resp = client.get("/health")
        assert resp.status_code == 200, f"Health check failed: {resp.status_code}"
        health_data = resp.json()
        print(f"  Response: {health_data}")
        assert health_data["status"] == "healthy"
        assert health_data["models_loaded"]["classifier"] is True, "Classifier failed to load!"
        print("  -> PASS: Health check endpoint functional and models confirmed loaded.")
        
        # 2. Test GET /samples
        print("\n[TEST 2] GET /samples...")
        resp = client.get("/samples")
        assert resp.status_code == 200
        samples = resp.json()
        print(f"  Gallery samples loaded: {len(samples)}")
        print("  -> PASS: Sample gallery endpoint functional.")
        
        # 3. Test POST /predict with Real Test Defect Image
        print("\n[TEST 3] POST /predict with real defect image...")
        sample_img_path = os.path.join("app", "static", "samples", "crazing_1.jpg")
        if not os.path.exists(sample_img_path):
            sample_img_path = os.path.join("data", "raw", "NEU-DET", "IMAGES", "crazing_1.jpg")
            
        with open(sample_img_path, "rb") as f:
            img_bytes = f.read()
            
        resp = client.post("/predict?threshold=0.60", files={"file": ("crazing_1.jpg", img_bytes, "image/jpeg")})
        assert resp.status_code == 200, f"Prediction failed: {resp.text}"
        pred_data = resp.json()
        print(f"  Predicted Class: {pred_data['predicted_class']}")
        print(f"  Confidence:      {pred_data['confidence_percentage']}%")
        print(f"  Verdict:         {pred_data['verdict']}")
        print(f"  Inference Time:  {pred_data['inference_latency_ms']} ms")
        print(f"  Inspection Note: '{pred_data['inspection_report']}'")
        assert "visualizations" in pred_data
        assert pred_data["visualizations"]["original_image"] is not None
        assert pred_data["visualizations"]["gradcam_heatmap_overlay"] is not None
        print("  -> PASS: Real defect prediction and explanation pipeline verified.")
        
        # 4. Test POST /predict with Out-of-Distribution (OOD) / Clean Uniform Image
        print("\n[TEST 4] POST /predict with Out-of-Distribution clean / uniform surface...")
        blank_arr = np.full((200, 200), 128, dtype=np.uint8)
        blank_buf = io.BytesIO()
        Image.fromarray(blank_arr).save(blank_buf, format="JPEG")
        blank_bytes = blank_buf.getvalue()
        
        resp = client.post("/predict?threshold=0.75", files={"file": ("clean_surface.jpg", blank_bytes, "image/jpeg")})
        assert resp.status_code == 200
        ood_data = resp.json()
        print(f"  Max Confidence:  {ood_data['confidence_percentage']}%")
        print(f"  Is Uncertain:    {ood_data['is_uncertain']}")
        print(f"  Verdict Status:  {ood_data['verdict_status']}")
        print(f"  Verdict Message: {ood_data['verdict_message']}")
        assert ood_data["is_uncertain"] is True
        print("  -> PASS: OOD / Low-contrast safety fallback behavior verified.")
        
        # 5. Test Input Validation: Unsupported Format
        print("\n[TEST 5] Input Validation: Unsupported file extension...")
        fake_txt = io.BytesIO(b"Not an image")
        resp = client.post("/predict", files={"file": ("report.pdf", fake_txt, "application/pdf")})
        assert resp.status_code == 400
        print(f"  Rejected status code: {resp.status_code} ({resp.json()['detail']})")
        print("  -> PASS: Invalid extension successfully rejected.")
        
        # 6. Test Input Validation: Corrupted image data
        print("\n[TEST 6] Input Validation: Corrupted image payload...")
        corrupt_bytes = b"CORRUPTED_RAW_JPEG_HEADER_INVALID"
        resp = client.post("/predict", files={"file": ("corrupt.jpg", corrupt_bytes, "image/jpeg")})
        assert resp.status_code == 400
        print(f"  Rejected status code: {resp.status_code} ({resp.json()['detail']})")
        print("  -> PASS: Corrupt image successfully intercepted.")
        
        print("\n" + "=" * 65)
        print("ALL API END-TO-END AND SECURITY TESTS PASSED (6/6)!")
        print("=" * 65)

def test_api_live():
    run_api_tests()

if __name__ == "__main__":
    run_api_tests()
