# Final Pre-Evaluation Checklist & Demonstration Guide

**Course**: CSAI3027P Deep Learning  
**Project**: Industrial Surface Defect Inspection Platform (Manufacturing Quality Assurance)  
**Evaluation Date**: 23 November 2026  
**Active Working Branch**: `final-audit-fixes`  
**Verified Platform URL**: `http://localhost:8000`  
**API Documentation URL**: `http://localhost:8000/docs`  

---

## 1. System Architecture & Startup Instructions

The platform operates as a unified, high-performance web service combining an asynchronous FastAPI backend with a responsive vanilla HTML5/CSS3/JavaScript single-page frontend.

### 1.1 Backend Startup Instructions
To launch the complete inspection platform with pre-warmed deep learning models:

```powershell
# Step 1: Open terminal in workspace root
cd "c:\Users\Acer\PBL PROJECT DL"

# Step 2: Activate the pinned Python virtual environment (Windows PowerShell)
.venv\Scripts\Activate.ps1

# (For Linux/macOS: source .venv/bin/activate)

# Step 3: Launch server with automatic model pre-loading
python run_app.py
```

*Expected Terminal Startup Output:*
```
=================================================================
LAUNCHING INDUSTRIAL SURFACE DEFECT INSPECTION PLATFORM
Address: http://localhost:8000
Documentation API: http://localhost:8000/docs
=================================================================
[SERVER] Initializing and loading inspection models into memory...
[SERVER] Successfully loaded ResNet50 classifier from models\checkpoints\best_resnet50.keras
[SERVER] Successfully loaded Defect Detector from models\checkpoints\best_detector.keras
[SERVER] Successfully loaded U-Net from models\checkpoints\best_unet.keras
[SERVER] Startup initialization complete. Ready for live requests.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

### 1.2 Frontend Access Instructions
The frontend is served directly by the FastAPI backend:
1. Open any web browser (Chrome, Edge, Firefox, Brave).
2. Navigate to: `http://localhost:8000` (or `http://127.0.0.1:8000`).
3. The dashboard loads immediately with:
   - Dynamic threshold slider (20% – 95%, default 70%).
   - Drag-and-drop surface image drop zone.
   - Quick-load Sample Gallery (6 representative NEU-DET defect categories).
   - Multi-modal Quad-View visual inspection matrix (Original, Bounding Box, Grad-CAM, Segmentation).
   - Synthetic natural-language inspection report generator.

---

## 2. Model Checkpoint Requirements & Integrity

All deep learning model checkpoints reside in `models/checkpoints/` and are verified full binary files (not Git LFS text pointer stubs):

| Model Name | Target Task | File Path | File Size | Memory Footprint | Architecture Summary |
|---|---|---|---|---|---|
| **Primary Classifier** | 6-Class Defect Categorization | `models/checkpoints/best_resnet50.keras` | 210.81 MB | ~95 MB RAM | ResNet50 Two-Stage Fine-Tuned (Frozen BatchNorm) |
| **Defect Detector** | Pascal VOC Bounding Box Localization | `models/checkpoints/best_detector.keras` | 219.99 MB | ~98 MB RAM | ResNet50 Multi-Task Shared Convolutional Head |
| **U-Net Segmentation** | Weak Pseudo-Mask Pixel Overlay | `models/checkpoints/best_unet.keras` | 22.15 MB | ~15 MB RAM | ResNet-Encoder U-Net with Skip Connections |
| **Benchmark Classifier** | Architectural Comparison | `models/checkpoints/best_inceptionv3.keras` | 90.33 MB | ~45 MB RAM | InceptionV3 Comparative Benchmark Model |

*Verification Command:*
```powershell
python -c "import os; [print(f'{f}: {os.path.getsize(f)/(1024*1024):.2f} MB') for f in ['models/checkpoints/best_resnet50.keras', 'models/checkpoints/best_detector.keras', 'models/checkpoints/best_unet.keras']]"
```

---

## 3. Live 6-Class Defect Verification Results

Tested against the running server (`http://127.0.0.1:8000/predict?threshold=0.70`) using real holdout samples from all six NEU-DET defect categories:

| Defect Class | Sample File | Predicted Class | Confidence | Match | Predicted Bounding Box `[x1, y1, x2, y2]` | Visualizations Rendered | Latency | Synthesized Inspection Note |
|---|---|---|---|---|---|---|---|---|
| **Crazing** | `crazing_1.jpg` | `crazing` | **100.00%** | **YES** | `[6, 7, 171, 186]` | Orig, BBox, Grad-CAM, UNet | 7.69 s | *Severe irregular clustered crazing defect identified near central region spanning approximately 73 percent surface area.* |
| **Inclusion** | `inclusion_1.jpg` | `inclusion` | **99.97%** | **YES** | `[106, 129, 133, 177]` | Orig, BBox, Grad-CAM, UNet | 2.33 s | *Minor isolated pinpoint inclusion defect identified near bottom region spanning approximately 3 percent surface area.* |
| **Patches** | `patches_1.jpg` | `patches` | **100.00%** | **YES** | `[51, 26, 102, 186]` | Orig, BBox, Grad-CAM, UNet | 1.99 s | *Moderate elongated patches defect identified near central region spanning approximately 20 percent surface area.* |
| **Pitted Surface** | `pitted_surface_1.jpg` | `pitted_surface` | **99.91%** | **YES** | `[37, 0, 198, 199]` | Orig, BBox, Grad-CAM, UNet | 2.05 s | *Severe irregular clustered pitted surface defect identified near central region spanning approximately 80 percent surface area.* |
| **Rolled-in Scale** | `rolled-in_scale_1.jpg` | `rolled-in_scale` | **99.98%** | **YES** | `[78, 85, 116, 149]` | Orig, BBox, Grad-CAM, UNet | 2.20 s | *Minor isolated pinpoint rolled-in scale defect identified near central region spanning approximately 6 percent surface area.* |
| **Scratches** | `scratches_1.jpg` | `scratches` | **99.07%** | **YES** | `[28, 8, 37, 161]` | Orig, BBox, Grad-CAM, UNet | 2.07 s | *Minor elongated scratches defect identified near central left spanning approximately 3 percent surface area.* |

*Overall Accuracy across 6 Classes*: **6 / 6 (100.0%) Correct**. Persisted in [outputs/metrics/six_classes_verification.json](file:///c:/Users/Acer/PBL%20PROJECT%20DL/outputs/metrics/six_classes_verification.json).

---

## 4. API Endpoints Verification Matrix

| Endpoint | HTTP Method | Expected Payload | Response Status | Verified Functionality |
|---|---|---|---|---|
| `/` | `GET` | None | `200 OK` | Serves complete single-page application (`index.html`) |
| `/health` | `GET` | None | `200 OK` | Returns JSON status: `healthy`, all 3 models confirmed loaded |
| `/samples` | `GET` | None | `200 OK` | Returns JSON array of 6 representative images with thumbnail URLs |
| `/static/*` | `GET` | Asset request | `200 OK` | Serves `style.css`, `app.js`, and sample JPEG images |
| `/predict` | `POST` | Multipart Form (`file`, `threshold`) | `200 OK` | Full inference: classification, quad-view base64 overlays, report |
| `/predict` | `POST` | Blank image ($\sigma < 0.025$) | `200 OK` | Low-contrast heuristic: returns `PASS: CLEAN / DEFECT-FREE SURFACE` |
| `/predict` | `POST` | Non-defect image ($\tau = 0.95$) | `200 OK` | Threshold triage: returns `UNCERTAIN / ANOMALY DETECTED` |
| `/predict` | `POST` | Unsupported extension (`.pdf`) | `400 Bad Request` | Security guard: rejects unsupported file formats |
| `/predict` | `POST` | Corrupt image byte stream | `400 Bad Request` | Defensive error handling: intercepts malformed headers |
| `/predict` | `POST` | File $> 5 \text{ MB}$ | `413 Payload Too Large` | DoS protection: rejects oversized payloads |

---

## 5. Automated Test Suite Execution Results

Executed command:
```powershell
pytest tests/ -v
```

*Actual Terminal Test Log:*
```
============================= test session starts =============================
platform win32 -- Python 3.12.8, pytest-9.2.0, pluggy-1.7.0 -- .venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: C:\Users\Acer\PBL PROJECT DL
configfile: pytest.ini
plugins: anyio-4.16.0
collected 15 items

tests/test_api_live.py::test_health_endpoint PASSED                      [  6%]
tests/test_api_live.py::test_samples_gallery_endpoint PASSED             [ 13%]
tests/test_api_live.py::test_predict_real_defect_image PASSED            [ 20%]
tests/test_api_live.py::test_predict_blank_surface_heuristic PASSED      [ 26%]
tests/test_api_live.py::test_predict_low_confidence_heuristic PASSED     [ 33%]
tests/test_api_live.py::test_input_validation_unsupported_format PASSED  [ 40%]
tests/test_api_live.py::test_input_validation_corrupted_payload PASSED   [ 46%]
tests/test_api_live.py::test_input_validation_oversized_payload PASSED   [ 53%]
tests/test_api_live.py::test_model_checkpoints_exist_and_load PASSED     [ 60%]
tests/test_fundamentals.py::test_tensors_and_broadcasting PASSED         [ 66%]
tests/test_fundamentals.py::test_mcculloch_pitts_logic_gates PASSED      [ 73%]
tests/test_fundamentals.py::test_activations_and_gradients PASSED        [ 80%]
tests/test_fundamentals.py::test_mlp_shapes_and_forward_backward PASSED  [ 86%]
tests/test_fundamentals.py::test_gradient_checking_strict PASSED         [ 93%]
tests/test_fundamentals.py::test_optimizers_update_weights PASSED        [100%]

============================= 15 passed in 34.50s =============================
```

*Warning Investigation*: All 4 initial warnings were investigated and completely resolved:
- **Keras Input Structure Warning**: Resolved in `models/gradcam.py` by supplying the exact input list structure `feature_extractor([x_pre], training=False)`.
- **Starlette TestClient Warning**: Filtered in `pytest.ini` (`StarletteDeprecationWarning`).
- **Result**: **0 failures, 0 warnings**.

---

## 6. Known Technical Limitations & Honest Disclosures

For evaluation defense, the following limitations are transparently documented:

1. **Multi-Box Detection Constraint**:
   - 73.17% of NEU-DET images contain $\ge 2$ defect instances (mean 2.33).
   - The deployed multi-task detector head predicts exactly one bounding box (the dominant defect by area). Secondary defects are not individually bounded.
   - Overall mAP@0.5 is **24.07%** (Mean IoU: **0.3292**), while compact clusters like `pitted_surface` achieve **86.67% AP@0.5**.
2. **Weak-Supervised Segmentation Pseudo-Masks**:
   - The U-Net was trained on synthetic pseudo-masks generated via Otsu thresholding inside bounding boxes.
   - Mean IoU is **0.1292** and Dice is **0.1950** evaluated against weak pseudo-labels, not human pixel ground truth.
3. **Vanilla Conv-VAE Texture Smoothing**:
   - In leakage-free experiments (`advanced/defect_vae.py`), VAE synthetic data augmentation resulted in a **-16.67% delta** (73.33% baseline $\to$ 56.67% augmented).
   - Vanilla MSE-based VAEs smooth high-frequency micro-textures, introducing label noise on fine steel scratch and crazing boundaries.
4. **OOD Heuristic Scope**:
   - The safety system relies on low-contrast surface detection ($\sigma < 0.025$) and confidence threshold triage ($\max(P) < \tau$). It is an operational decision heuristic, not a formal density-based epistemic OOD model.

---

## 7. Exact Step-by-Step Commands for Live Demonstration

Follow this exact sequence during the live PBL demonstration:

```powershell
# 1. Open Terminal and activate environment
cd "c:\Users\Acer\PBL PROJECT DL"
.venv\Scripts\Activate.ps1

# 2. Run all unit tests to prove test integrity before the examiner
pytest tests/ -v

# 3. Launch live web application
python run_app.py

# 4. Open browser at:
# http://localhost:8000

# 5. Live UI Demonstration Steps:
#   a. Click "Crazing" sample card -> click "ANALYZE SURFACE DEFECT" -> observe 100% confidence, quad-view matrix, and report.
#   b. Adjust slider to 90% -> observe instant threshold reactivity.
#   c. Click "Pitted Surface" sample card -> click Analyze -> observe 86.7% AP localization box and Grad-CAM heatmap.
#   d. Upload clean uniform image -> observe green "PASS: CLEAN / DEFECT-FREE SURFACE" verdict.
#   e. Upload ambiguous image with 95% threshold -> observe amber "UNCERTAIN / ANOMALY DETECTED" banner.

# 6. Execute programmatic 6-class live verification (if requested by examiner):
python test_six_classes.py
```
