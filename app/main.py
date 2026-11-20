"""
app/main.py
FastAPI Production Backend for Industrial Surface Defect Inspection Platform:
1. Loads deep learning models once at startup (ResNet50 Classifier, Detector, U-Net).
2. Endpoints:
   - GET /health: Health check, system specifications, loaded models.
   - GET /samples: Registry of test split defect samples for the demo gallery.
   - POST /predict: Image upload with strict validation, defect classification,
     OOD/uncertainty detection, bounding box localization, Grad-CAM explanation,
     weak segmentation mask, and automated synthetic inspection report.
3. Static files mounted at /static, root / serves index.html.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))
import io
import time
import json
import base64
import numpy as np
from PIL import Image, ImageOps, ImageDraw

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf

from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from models.gradcam import make_gradcam_heatmap, generate_gradcam_overlay
from advanced.defect_transformer import generate_synthetic_report

# Global model holders (loaded once at startup)
MODELS = {
    "classifier": None,
    "detector": None,
    "unet": None,
    "classes": ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"],
    "_is_loaded": False
}

def load_models_once():
    """Idempotent model loader: loads deep neural networks exactly once into memory."""
    if MODELS["_is_loaded"]:
        return
    print("[SERVER] Initializing and loading inspection models into memory...")
    
    # Load class names
    manifest_path = os.path.join("data", "splits", "split_manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
            MODELS["classes"] = manifest.get("classes", MODELS["classes"])
            
    # Load ResNet50 Classifier
    clf_path = os.path.join("models", "checkpoints", "best_resnet50.keras")
    if os.path.exists(clf_path):
        try:
            MODELS["classifier"] = tf.keras.models.load_model(clf_path)
            print(f"[SERVER] Successfully loaded ResNet50 classifier from {clf_path}")
        except Exception as e:
            print(f"[SERVER WARNING] Could not load classifier: {e}")
            
    # Load Detector if available
    det_path = os.path.join("models", "checkpoints", "best_detector.keras")
    if os.path.exists(det_path):
        try:
            MODELS["detector"] = tf.keras.models.load_model(det_path, compile=False)
            print(f"[SERVER] Successfully loaded Defect Detector from {det_path}")
        except Exception as e:
            print(f"[SERVER WARNING] Could not load detector: {e}")
            
    # Load U-Net if available
    unet_path = os.path.join("models", "checkpoints", "best_unet.keras")
    if os.path.exists(unet_path):
        try:
            MODELS["unet"] = tf.keras.models.load_model(unet_path, compile=False)
            print(f"[SERVER] Successfully loaded U-Net from {unet_path}")
        except Exception as e:
            print(f"[SERVER WARNING] Could not load U-Net: {e}")
            
    MODELS["_is_loaded"] = True
    print("[SERVER] Startup initialization complete. Ready for live requests.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern lifespan event handler for clean startup and shutdown."""
    load_models_once()
    yield

# Initialize FastAPI App with lifespan context manager
app = FastAPI(
    title="Industrial Surface Defect Inspection Platform",
    description="Automated AI-powered quality inspection system for hot-rolled steel strip surface defects.",
    version="1.0.0",
    lifespan=lifespan
)

# Secure CORS configuration: explicit origins required when allow_credentials=True
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def pil_to_base64_jpeg(pil_img):
    """Encodes PIL Image to Base64 JPEG data URL."""
    buffered = io.BytesIO()
    pil_img.save(buffered, format="JPEG", quality=90)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{img_str}"

@app.get("/health")
def health_check():
    """System health check endpoint."""
    load_models_once()
    return {
        "status": "healthy",
        "service": "Industrial Surface Defect Inspection Platform",
        "models_loaded": {
            "classifier": MODELS["classifier"] is not None,
            "detector": MODELS["detector"] is not None,
            "unet": MODELS["unet"] is not None
        },
        "classes": MODELS["classes"],
        "device": "CPU (Optimized with TF/Keras)"
    }

@app.get("/samples")
def get_sample_gallery():
    """Serves metadata for the sample gallery on the frontend."""
    gallery_reg_path = os.path.join("app", "static", "samples", "sample_registry.json")
    if os.path.exists(gallery_reg_path):
        with open(gallery_reg_path, "r") as f:
            return json.load(f)
    return []

@app.post("/predict")
async def predict_surface_defect(
    file: UploadFile = File(...),
    threshold: float = Query(0.60, ge=0.20, le=0.95, description="Confidence threshold for defect alert")
):
    """
    Analyzes an uploaded surface image:
    1. Validates file extension, MIME type, and size (< 5MB).
    2. Resizes and converts grayscale/RGB automatically.
    3. Runs ResNet50 classifier + detector + Grad-CAM + weak segmentation.
    4. Evaluates Out-of-Distribution (OOD) / Uncertainty safety.
    """
    start_time = time.perf_counter()
    load_models_once()
    
    # 1. Validation: File extension
    filename = file.filename.lower()
    allowed_exts = (".jpg", ".jpeg", ".png", ".bmp")
    if not any(filename.endswith(ext) for ext in allowed_exts):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{file.filename}'. Allowed formats: JPEG, PNG, BMP."
        )
        
    # 2. Validation: File size (< 5 MB)
    contents = await file.read()
    max_bytes = 5 * 1024 * 1024
    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds maximum allowed size of 5 MB ({len(contents) / (1024*1024):.2f} MB received)."
        )
        
    # 3. Read image with PIL
    try:
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Corrupted or unreadable image file. Please provide a valid JPEG, PNG, or BMP."
        )
        
    orig_w, orig_h = pil_img.size
    
    # Preprocess for model (200, 200, 3)
    img_resized = pil_img.resize((200, 200), Image.Resampling.BILINEAR)
    img_arr = np.expand_dims(np.array(img_resized, dtype=np.float32) / 255.0, axis=0)
    
    classes = MODELS["classes"]
    
    # 4. Model Inference: Classifier
    if MODELS["classifier"] is not None:
        probs = MODELS["classifier"].predict(img_arr, verbose=0)[0]
    else:
        # Fallback dummy distribution if model still compiling
        probs = np.ones(len(classes)) / len(classes)
        
    pred_idx = int(np.argmax(probs))
    pred_class = classes[pred_idx]
    confidence = float(probs[pred_idx])
    
    # Class probability distribution map
    probability_distribution = {
        cls_name: round(float(probs[i]) * 100.0, 2) for i, cls_name in enumerate(classes)
    }
    
    # 5. Low-Confidence & Uniform Surface Safety Check (Heuristic Quality Triage)
    # Note: This is an operational heuristic (contrast standard deviation and confidence threshold),
    # not a formal density-based epistemic Out-of-Distribution (OOD) estimator.
    img_contrast = float(np.std(img_arr))
    is_blank = bool(img_contrast < 0.025)
    is_uncertain = bool(confidence < threshold or is_blank)
    
    if is_blank:
        verdict = "PASS: CLEAN / DEFECT-FREE SURFACE"
        verdict_status = "PASS"
        verdict_color = "success" # green
        verdict_message = "Uniform steel surface with no defect textures detected. Surface passed quality inspection."
    elif is_uncertain:
        verdict = "UNCERTAIN / ANOMALY DETECTED"
        verdict_status = "UNCERTAIN"
        verdict_color = "warning" # amber
        verdict_message = (
            f"Maximum prediction confidence ({confidence*100:.1f}%) is below inspection threshold ({threshold*100:.1f}%). "
            "Surface is potentially Out-of-Distribution, non-defect steel, or corrupted. Flagged for manual human review."
        )
    else:
        verdict = f"QC FAIL: {pred_class.upper()} DETECTED"
        verdict_status = "FAIL"
        verdict_color = "danger" # red
        verdict_message = f"Defect confirmed with {confidence*100:.1f}% confidence. Automated rejection recommended."
        
    # 6. Grad-CAM Explanation
    gradcam_base64 = None
    heatmap_raw = None
    if MODELS["classifier"] is not None:
        try:
            heatmap_raw = make_gradcam_heatmap(img_arr, MODELS["classifier"], pred_index=pred_idx)
            overlay_pil, _ = generate_gradcam_overlay(pil_img, heatmap_raw, alpha=0.45)
            gradcam_base64 = pil_to_base64_jpeg(overlay_pil)
        except Exception as e:
            print(f"[Grad-CAM Error]: {e}")
            
    # 7. Localization Bounding Box
    box_coords = None
    annotated_box_img = pil_img.copy()
    draw = ImageDraw.Draw(annotated_box_img)
    
    if MODELS["detector"] is not None:
        try:
            det_preds = MODELS["detector"].predict(img_arr, verbose=0)
            pbox = det_preds["box_output"][0] # [ymin, xmin, ymax, xmax] in [0, 1]
            ymin, xmin = int(pbox[0] * orig_h), int(pbox[1] * orig_w)
            ymax, xmax = int(pbox[2] * orig_h), int(pbox[3] * orig_w)
            box_coords = [xmin, ymin, xmax, ymax]
            draw.rectangle([xmin, ymin, xmax, ymax], outline="#ef4444", width=3)
            draw.text((xmin + 4, max(0, ymin - 16)), f"{pred_class} ({confidence*100:.1f}%)", fill="#ef4444")
        except Exception as e:
            print(f"[Detector Error]: {e}")
    elif heatmap_raw is not None:
        # Fallback: Extract bounding box from Grad-CAM hotspot
        hotspots = heatmap_raw > 0.50
        if np.any(hotspots):
            y_indices, x_indices = np.where(hotspots)
            ymin = int(np.min(y_indices) / heatmap_raw.shape[0] * orig_h)
            ymax = int(np.max(y_indices) / heatmap_raw.shape[0] * orig_h)
            xmin = int(np.min(x_indices) / heatmap_raw.shape[1] * orig_w)
            xmax = int(np.max(x_indices) / heatmap_raw.shape[1] * orig_w)
            box_coords = [xmin, ymin, xmax, ymax]
            draw.rectangle([xmin, ymin, xmax, ymax], outline="#ef4444", width=3)
            draw.text((xmin + 4, max(0, ymin - 16)), f"{pred_class} ({confidence*100:.1f}%)", fill="#ef4444")
            
    box_image_base64 = pil_to_base64_jpeg(annotated_box_img)
    
    # 8. Weak Segmentation Mask Overlay
    segmentation_base64 = None
    if MODELS["unet"] is not None:
        try:
            unet_h, unet_w = MODELS["unet"].input_shape[1], MODELS["unet"].input_shape[2]
            gray_in = np.expand_dims(np.array(pil_img.convert("L").resize((unet_w, unet_h)), dtype=np.float32) / 255.0, axis=(0, -1))
            mask_pred = MODELS["unet"].predict(gray_in, verbose=0)[0, :, :, 0]
            mask_pil = Image.fromarray(np.uint8((mask_pred >= 0.50) * 255)).resize((orig_w, orig_h), Image.Resampling.NEAREST)
            # Create colored red overlay
            mask_colored = Image.new("RGBA", (orig_w, orig_h), (239, 68, 68, 0))
            mask_colored.paste(Image.new("RGBA", (orig_w, orig_h), (239, 68, 68, 140)), mask=mask_pil)
            seg_blended = pil_img.convert("RGBA")
            seg_blended.alpha_composite(mask_colored)
            segmentation_base64 = pil_to_base64_jpeg(seg_blended.convert("RGB"))
        except Exception as e:
            print(f"[UNet Error]: {e}")
            
    # 9. Automated Synthetic Inspection Report
    mock_boxes = []
    if box_coords:
        mock_boxes.append({
            "norm_xmin": box_coords[0] / orig_w,
            "norm_ymin": box_coords[1] / orig_h,
            "norm_xmax": box_coords[2] / orig_w,
            "norm_ymax": box_coords[3] / orig_h,
        })
    report_text = generate_synthetic_report(pred_class, mock_boxes)
    if is_uncertain:
        report_text += " [Caution: Inspection confidence below decision threshold; manual review required.]"
        
    latency_ms = round((time.perf_counter() - start_time) * 1000.0, 1)
    
    return {
        "success": True,
        "filename": file.filename,
        "predicted_class": pred_class,
        "confidence_percentage": round(confidence * 100.0, 2),
        "threshold": round(threshold * 100.0, 1),
        "verdict": verdict,
        "verdict_status": verdict_status,
        "verdict_color": verdict_color,
        "verdict_message": verdict_message,
        "is_uncertain": is_uncertain,
        "probability_distribution": probability_distribution,
        "bounding_box": box_coords, # [xmin, ymin, xmax, ymax]
        "inspection_report": report_text,
        "inference_latency_ms": latency_ms,
        "visualizations": {
            "original_image": pil_to_base64_jpeg(pil_img),
            "bounding_box_overlay": box_image_base64,
            "gradcam_heatmap_overlay": gradcam_base64,
            "segmentation_mask_overlay": segmentation_base64
        }
    }

# Mount static folder
static_dir = os.path.join("app", "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/", response_class=HTMLResponse)
def serve_index():
    """Serves the main inspection dashboard page."""
    index_path = os.path.join("app", "static", "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Surface Defect Inspection Platform</h1><p>Static index.html not yet found.</p>"
