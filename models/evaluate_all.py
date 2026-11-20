"""
models/evaluate_all.py
Industrial Surface Defect Inspection Platform (CSAI3027P)
Unified evaluation script generating outputs/metrics/test_metrics.json in one command.
Evaluates the primary fine-tuned ResNet50 model on the held-out test split,
computes overall accuracy, macro F1, per-class metrics, confusion matrix,
inference latency/throughput, and industrial False-Reject Rate across thresholds.
Also aggregates localization and segmentation benchmarks.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))
import time
import json
import datetime
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score

def run_evaluation():
    print("=" * 65)
    print("UNIFIED TEST EVALUATION SUITE: GENERATING test_metrics.json")
    print("=" * 65)
    
    # 1. Load Split Manifest
    manifest_path = os.path.join("data", "splits", "split_manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Missing split manifest at {manifest_path}. Run prepare_data.py first.")
        
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
        
    classes = manifest["classes"]
    class_to_id = manifest["class_to_id"]
    test_records = [r for r in manifest["records"] if r["split"] == "test"]
    print(f"[DATA] Loaded {len(test_records)} test samples across {len(classes)} classes.")

    # 2. Load Primary Classifier
    clf_path = os.path.join("models", "checkpoints", "best_resnet50.keras")
    if not os.path.exists(clf_path):
        raise FileNotFoundError(f"Model checkpoint not found at {clf_path}. Train model first.")
        
    print(f"[MODEL] Loading fine-tuned classifier from {clf_path}...")
    model = tf.keras.models.load_model(clf_path)

    # 3. Batch Evaluation & Latency Profiling
    y_true = []
    y_pred_probs = []
    latencies = []
    
    print("[EVAL] Running inference across held-out test split...")
    for i, r in enumerate(test_records):
        with Image.open(r["image_path"]) as im:
            im_resized = im.resize((200, 200))
            im_arr = np.array(im_resized.convert("RGB"), dtype=np.float32) / 255.0
            
        inp = np.expand_dims(im_arr, axis=0)
        
        t0 = time.perf_counter()
        pred = model.predict(inp, verbose=0)
        t1 = time.perf_counter()
        
        latencies.append((t1 - t0) * 1000.0) # ms
        y_pred_probs.append(pred[0])
        y_true.append(r["class_id"])

    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # 4. Compute Metrics
    test_acc = float(np.mean(y_pred == y_true))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    mean_latency = float(np.mean(latencies[5:])) if len(latencies) > 5 else float(np.mean(latencies))
    fps = float(1000.0 / mean_latency) if mean_latency > 0 else 0.0

    clf_dict = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    cm = confusion_matrix(y_true, y_pred).tolist()

    # 5. Dual False-Reject Rate (FRR) & Triage Analysis across Operating Thresholds
    # In NEU-DET, all test samples contain defects. We define:
    # - Triage Escalation Rate: proportion of samples routed to human review (max_prob < threshold)
    # - Silent Escape Rate: proportion of misclassified samples that bypassed human triage (y_pred != y_true AND max_prob >= threshold)
    # - Total Non-Pass Rate: proportion of samples requiring review or misclassified (y_pred != y_true OR max_prob < threshold)
    max_confs = np.max(y_pred_probs, axis=1)
    thresholds = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90]
    frr_analysis = {}
    total_samples = len(y_true)
    for th in thresholds:
        flagged_mask = (max_confs < th)
        flagged_count = int(np.sum(flagged_mask))
        triage_rate = float(flagged_count / total_samples)
        
        silent_escape_mask = (y_pred != y_true) & (~flagged_mask)
        silent_escape_count = int(np.sum(silent_escape_mask))
        silent_escape_rate = float(silent_escape_count / total_samples)
        
        non_pass_mask = (y_pred != y_true) | flagged_mask
        non_pass_count = int(np.sum(non_pass_mask))
        total_non_pass_rate = float(non_pass_count / total_samples)
        
        frr_analysis[f"threshold_{th:.2f}"] = {
            "threshold": th,
            "flagged_for_triage": flagged_count,
            "triage_escalation_rate": triage_rate,
            "silent_escape_count": silent_escape_count,
            "silent_escape_rate": silent_escape_rate,
            "total_non_pass_count": non_pass_count,
            "total_non_pass_rate": total_non_pass_rate,
            "definition_note": "Triage = max_prob < threshold; Silent Escape = wrong class & max_prob >= threshold."
        }

    # 6. Load Secondary Detector & U-Net Metrics from executed metric files (NO HARDCODED FALLBACKS)
    det_json = os.path.join("outputs", "metrics", "detector_test_metrics.json")
    if os.path.exists(det_json):
        with open(det_json, "r") as f:
            det_data = json.load(f)
            detector_metrics = det_data.get("test_metrics", det_data)
        detector_source = det_json
        print(f"[METRICS] Successfully loaded detector metrics from {det_json}")
    else:
        detector_metrics = {}
        detector_source = "UNAVAILABLE - Run models/detector.py"
        print(f"[WARNING] Detector metrics file not found at {det_json}.")

    unet_json = os.path.join("outputs", "metrics", "unet_test_metrics.json")
    if os.path.exists(unet_json):
        with open(unet_json, "r") as f:
            unet_data = json.load(f)
            unet_metrics = unet_data.get("test_metrics", unet_data)
        unet_source = unet_json
        print(f"[METRICS] Successfully loaded U-Net metrics from {unet_json}")
    else:
        unet_metrics = {}
        unet_source = "UNAVAILABLE - Run models/unet_segmentation.py"
        print(f"[WARNING] U-Net metrics file not found at {unet_json}.")

    # 7. Export Unified JSON (strictly real executed values, null if not executed)
    unified_metrics = {
        "report_name": "CSAI3027P Industrial Surface Defect Inspection Platform - Test Metrics",
        "timestamp": datetime.datetime.now().isoformat(),
        "evaluation_dataset": "NEU-DET (Held-out Test Split)",
        "total_test_samples": len(test_records),
        "primary_classifier": {
            "architecture": "ResNet50 (Two-Stage Transfer Learning with Frozen BatchNormalization)",
            "test_accuracy": test_acc,
            "test_accuracy_percentage": f"{test_acc * 100:.2f}%",
            "test_macro_f1": macro_f1,
            "mean_latency_ms": round(mean_latency, 2),
            "throughput_fps": round(fps, 2),
            "per_class": {cls_name: clf_dict[cls_name] for cls_name in classes},
            "confusion_matrix": cm,
            "false_reject_rate_analysis": frr_analysis
        },
        "defect_detector": {
            "architecture": "ResNet50 Multi-Task Shared Bounding Box Head",
            "status": "Available" if detector_metrics else "Not evaluated",
            "source_file": detector_source,
            "mAP_50": detector_metrics.get("mAP_50"),
            "mean_iou": detector_metrics.get("mean_iou"),
            "per_class_ap_50": detector_metrics.get("per_class_ap_50"),
            "throughput_fps": detector_metrics.get("throughput_fps")
        },
        "unet_segmentation": {
            "architecture": "Lightweight U-Net (ResNet Encoder, Weak Pseudo-Masks)",
            "status": "Available" if unet_metrics else "Not evaluated",
            "source_file": unet_source,
            "mean_iou": unet_metrics.get("mean_iou"),
            "dice_score": unet_metrics.get("mean_dice", unet_metrics.get("dice_score")),
            "throughput_fps": unet_metrics.get("throughput_fps")
        }
    }

    out_path = os.path.join("outputs", "metrics", "test_metrics.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(unified_metrics, f, indent=2)

    print(f"\n[SUMMARY] Results on this test split ({len(test_records)} images):")
    print(f"  - Classifier Accuracy: {test_acc * 100:.2f}% (Macro F1: {macro_f1:.4f})")
    print(f"  - Latency per Image:   {mean_latency:.2f} ms ({fps:.1f} FPS on CPU)")
    if unified_metrics["defect_detector"]["mAP_50"] is not None:
        print(f"  - Detector mAP@0.5:    {unified_metrics['defect_detector']['mAP_50']*100:.2f}% (Mean IoU: {unified_metrics['defect_detector']['mean_iou']:.4f})")
    if unified_metrics["unet_segmentation"]["mean_iou"] is not None:
        print(f"  - U-Net Pseudo IoU:    {unified_metrics['unet_segmentation']['mean_iou']:.4f} (Dice: {unified_metrics['unet_segmentation']['dice_score']:.4f})")
    print(f"[SAVE] Exported unified test metrics to {out_path}")

if __name__ == "__main__":
    run_evaluation()
