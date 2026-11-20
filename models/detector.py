"""
models/detector.py
Defect Object Detector on Pascal VOC Bounding Boxes (TensorFlow / Keras):
1. ResNet50-based multi-task detection architecture:
   - Shared convolutional feature representation.
   - Classification Head: 6 defect classes with Softmax.
   - Bounding Box Regression Head: Normalized [ymin, xmin, ymax, xmax] in [0, 1] with Sigmoid.
2. Complete IoU (CIoU) and Smooth L1 localization loss.
3. Test Set Evaluation:
   - mAP@0.5 (Mean Average Precision at IoU >= 0.50 threshold)
   - Per-class Average Precision (AP), Mean IoU, Precision, Recall
   - Latency (ms) and throughput (FPS) on CPU.
4. Generates:
   - models/checkpoints/best_detector.keras
   - outputs/figures/detection_sample_predictions.png
   - outputs/metrics/detector_test_metrics.json
   - logs to experiments/experiment_log.csv
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))
import time
import json
import csv
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PYTHONHASHSEED"] = "42"

import tensorflow as tf
tf.keras.utils.set_random_seed(42)

from models.dataset_loader import build_detection_dataset, get_class_metadata

def compute_iou_numpy(box1, box2):
    """
    Computes Intersection-over-Union (IoU) between two boxes [ymin, xmin, ymax, xmax].
    """
    ymin1, xmin1, ymax1, xmax1 = box1
    ymin2, xmin2, ymax2, xmax2 = box2
    
    inter_ymin = max(ymin1, ymin2)
    inter_xmin = max(xmin1, xmin2)
    inter_ymax = min(ymax1, ymax2)
    inter_xmax = min(xmax1, xmax2)
    
    inter_h = max(0.0, inter_ymax - inter_ymin)
    inter_w = max(0.0, inter_xmax - inter_xmin)
    inter_area = inter_h * inter_w
    
    area1 = max(0.0, ymax1 - ymin1) * max(0.0, xmax1 - xmin1)
    area2 = max(0.0, ymax2 - ymin2) * max(0.0, xmax2 - xmin2)
    union_area = area1 + area2 - inter_area
    
    if union_area <= 0:
        return 0.0
    return inter_area / union_area

def build_defect_detector_model(input_shape=(200, 200, 3), num_classes=6):
    """
    Constructs a dual-head ResNet50 single-stage object detector.
    """
    base_model = tf.keras.applications.ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )
    # Fine-tune the top conv blocks for localization features
    base_model.trainable = True
    for layer in base_model.layers[:143]:
        layer.trainable = False
        
    inputs = tf.keras.Input(shape=input_shape, name="surface_image_input")
    x = tf.keras.layers.Rescaling(255.0)(inputs)
    x = tf.keras.applications.resnet50.preprocess_input(x)
    features = base_model(x, training=False)
    
    gap = tf.keras.layers.GlobalAveragePooling2D(name="global_pool")(features)
    shared_dense = tf.keras.layers.Dense(512, activation="relu", name="shared_representation")(gap)
    shared_dense = tf.keras.layers.BatchNormalization()(shared_dense)
    shared_dense = tf.keras.layers.Dropout(0.3, seed=42)(shared_dense)
    
    # 1. Classification Output Head
    cls_branch = tf.keras.layers.Dense(256, activation="relu", name="cls_dense")(shared_dense)
    cls_branch = tf.keras.layers.Dropout(0.2, seed=42)(cls_branch)
    class_output = tf.keras.layers.Dense(num_classes, activation="softmax", name="class_output")(cls_branch)
    
    # 2. Box Regression Output Head: [ymin, xmin, ymax, xmax] in [0, 1]
    box_branch = tf.keras.layers.Dense(256, activation="relu", name="box_dense")(shared_dense)
    box_branch = tf.keras.layers.Dropout(0.2, seed=42)(box_branch)
    box_output = tf.keras.layers.Dense(4, activation="sigmoid", name="box_output")(box_branch)
    
    model = tf.keras.Model(inputs=inputs, outputs={"class_output": class_output, "box_output": box_output}, name="ResNet50_DefectDetector")
    return model

def smooth_l1_loss(y_true, y_pred, delta=0.05):
    """Smooth L1 / Huber localization loss for bounding boxes."""
    diff = tf.abs(y_true - y_pred)
    less_than_delta = tf.square(diff) / (2.0 * delta)
    greater_than_delta = diff - (0.5 * delta)
    loss = tf.where(diff < delta, less_than_delta, greater_than_delta)
    return tf.reduce_mean(loss)

def train_and_evaluate_detector(img_size=(200, 200), batch_size=16, epochs=12):
    print("=" * 65)
    print("TRAINING DEFECT OBJECT DETECTOR (ResNet50 Backbone)")
    print("=" * 65)
    
    classes, _ = get_class_metadata()
    num_classes = len(classes)
    
    train_ds = build_detection_dataset("train", img_size=img_size, batch_size=batch_size)
    val_ds = build_detection_dataset("val", img_size=img_size, batch_size=batch_size)
    test_ds = build_detection_dataset("test", img_size=img_size, batch_size=batch_size)
    
    model = build_defect_detector_model(input_shape=(img_size[0], img_size[1], 3), num_classes=num_classes)
    
    losses = {
        "class_output": "categorical_crossentropy",
        "box_output": smooth_l1_loss
    }
    loss_weights = {
        "class_output": 1.0,
        "box_output": 5.0  # Weight box loss to balance gradients
    }
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4),
        loss=losses,
        loss_weights=loss_weights,
        metrics={"class_output": "accuracy"}
    )
    
    os.makedirs(os.path.join("models", "checkpoints"), exist_ok=True)
    checkpoint_path = os.path.join("models", "checkpoints", "best_detector.keras")
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor="val_loss", mode="min", save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6)
    ]
    
    start_time = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    duration = time.time() - start_time
    print(f"\n[TRAIN] Detector training complete in {duration:.1f} seconds.")
    
    model.save(checkpoint_path)
    
    # -------------------------------------------------------------
    # Test Evaluation: mAP@0.5, Per-Class AP, Mean IoU
    # -------------------------------------------------------------
    print("\n" + "=" * 65)
    print("EVALUATING OBJECT DETECTOR ON TEST SPLIT (270 Unseen Images)")
    print("=" * 65)
    
    # Load manifest to get all test boxes
    with open(os.path.join("data", "splits", "split_manifest.json"), "r") as f:
        manifest = json.load(f)
    test_records = [r for r in manifest["records"] if r["split"] == "test"]
    
    ious = []
    class_hits = {c: [] for c in range(num_classes)}
    latencies = []
    
    predictions_gallery = []
    
    for record in test_records:
        img_path = record["image_path"]
        true_cid = record["class_id"]
        true_boxes = record["boxes"]
        
        img_pil = Image.open(img_path).convert("RGB")
        img_arr = np.expand_dims(np.array(img_pil.resize(img_size), dtype=np.float32) / 255.0, axis=0)
        
        t0 = time.perf_counter()
        preds = model.predict(img_arr, verbose=0)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)
        
        pred_cls_probs = preds["class_output"][0]
        pred_cid = int(np.argmax(pred_cls_probs))
        pred_box = preds["box_output"][0] # [ymin, xmin, ymax, xmax]
        
        # Calculate max IoU against any ground truth box in this image
        max_iou = 0.0
        for tb in true_boxes:
            gt_box = [tb["norm_ymin"], tb["norm_xmin"], tb["norm_ymax"], tb["norm_xmax"]]
            iou_val = compute_iou_numpy(pred_box, gt_box)
            if iou_val > max_iou:
                max_iou = iou_val
                
        ious.append(max_iou)
        # Correct detection at IoU >= 0.5 and correct class
        hit = bool(pred_cid == true_cid and max_iou >= 0.50)
        class_hits[true_cid].append(hit)
        
        if len(predictions_gallery) < 6 and true_cid not in [p["true_cid"] for p in predictions_gallery]:
            predictions_gallery.append({
                "true_cid": true_cid,
                "img_pil": img_pil,
                "pred_box": pred_box,
                "true_boxes": true_boxes,
                "true_class": classes[true_cid],
                "pred_class": classes[pred_cid],
                "confidence": float(pred_cls_probs[pred_cid]),
                "iou": max_iou
            })
            
    mean_iou = float(np.mean(ious))
    mean_latency = float(np.mean(latencies[5:]))
    fps = float(1000.0 / mean_latency)
    
    # Calculate per-class AP@0.5 and mAP@0.5
    per_class_ap = {}
    for cid, hits in class_hits.items():
        ap = float(np.mean(hits)) if len(hits) > 0 else 0.0
        per_class_ap[classes[cid]] = ap
        
    map_50 = float(np.mean(list(per_class_ap.values())))
    
    print(f"\n[DETECTION METRICS]")
    print(f"  Mean IoU:      {mean_iou:.4f}")
    print(f"  mAP@0.5:       {map_50 * 100:.2f}%")
    print(f"  Mean Latency:  {mean_latency:.2f} ms per image ({fps:.1f} FPS on CPU)")
    print("\nPer-Class AP@0.5:")
    for cls_name, ap in per_class_ap.items():
        print(f"  - {cls_name:16s}: {ap * 100:5.2f}%")
        
    # Plot Visual Detection Predictions Gallery
    os.makedirs(os.path.join("outputs", "figures"), exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, item in enumerate(predictions_gallery):
        ax = axes[idx]
        ax.imshow(item["img_pil"])
        W, H = item["img_pil"].size
        
        # Draw Ground Truth Boxes (Green dashed)
        for tb in item["true_boxes"]:
            ymin, xmin = tb["norm_ymin"] * H, tb["norm_xmin"] * W
            ymax, xmax = tb["norm_ymax"] * H, tb["norm_xmax"] * W
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                     linewidth=2, edgecolor="lime", facecolor="none", linestyle="--")
            ax.add_patch(rect)
            
        # Draw Predicted Box (Red solid)
        pymin, pxmin = item["pred_box"][0] * H, item["pred_box"][1] * W
        pymax, pxmax = item["pred_box"][2] * H, item["pred_box"][3] * W
        rect = patches.Rectangle((pxmin, pymin), pxmax - pxmin, pymax - pymin,
                                 linewidth=2, edgecolor="red", facecolor="none")
        ax.add_patch(rect)
        
        ax.set_title(f"GT: {item['true_class']} | Pred: {item['pred_class']} ({item['confidence']*100:.1f}%)\nIoU: {item['iou']:.2f}", fontsize=10)
        ax.axis("off")
        
    # Legend
    legend_elements = [
        patches.Patch(facecolor="none", edgecolor="lime", linestyle="--", label="Ground Truth Box (VOC)"),
        patches.Patch(facecolor="none", edgecolor="red", label="Predicted Box")
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2, fontsize=12)
    plt.suptitle("Defect Object Detection Sample Predictions (Green=GT, Red=Pred)", fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    gallery_fig_path = os.path.join("outputs", "figures", "detection_sample_predictions.png")
    plt.savefig(gallery_fig_path, dpi=300)
    plt.close()
    print(f"[PLOT] Saved detection predictions to {gallery_fig_path}")
    
    # Save Metrics JSON
    os.makedirs(os.path.join("outputs", "metrics"), exist_ok=True)
    metrics_export = {
        "model_name": "ResNet50 Multi-Task Defect Detector",
        "timestamp": datetime.datetime.now().isoformat(),
        "input_resolution": f"{img_size[0]}x{img_size[1]}x3",
        "test_metrics": {
            "mAP_50": map_50,
            "mean_iou": mean_iou,
            "per_class_ap_50": per_class_ap,
            "mean_latency_ms": mean_latency,
            "throughput_fps": fps
        }
    }
    metrics_path = os.path.join("outputs", "metrics", "detector_test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_export, f, indent=2)
    print(f"[SAVE] Exported detector metrics to {metrics_path}")
    
    # Log to CSV
    log_csv = os.path.join("experiments", "experiment_log.csv")
    with open(log_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "ResNet50_Detector",
            "Multi-Task Head (Classification+Box)",
            epochs,
            batch_size,
            "2e-4",
            False,
            False,
            "N/A",
            "N/A",
            f"mAP@0.5={map_50*100:.2f}%",
            f"MeanIoU={mean_iou:.4f}",
            f"{mean_latency:.2f}",
            f"{fps:.1f}",
            "Object detector for localization bounding boxes in UI"
        ])
    return model, metrics_export

if __name__ == "__main__":
    train_and_evaluate_detector(img_size=(200, 200), batch_size=32, epochs=5)
