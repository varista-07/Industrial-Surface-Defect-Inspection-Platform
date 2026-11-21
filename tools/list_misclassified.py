"""
tools/list_misclassified.py
Industrial Surface Defect Inspection Platform (CSAI3027P)

Evaluates best_resnet50.keras and best_inceptionv3.keras on the held-out
test split (270 images) from data/splits/split_manifest.json using the exact
same preprocessing as models/evaluate_all.py (200x200, RGB, [0, 1]).

Exports misclassified sample details to:
- outputs/metrics/misclassified_resnet50_seed42.csv
- outputs/metrics/misclassified_inceptionv3_seed42.csv
with columns: file, true_class, predicted_class, confidence
"""

import os
import sys
import json
import csv
import numpy as np
from PIL import Image

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf

def evaluate_and_list_misclassified():
    manifest_path = os.path.join("data", "splits", "split_manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Missing split manifest at {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as fp:
        manifest = json.load(fp)

    classes = manifest["classes"]
    test_records = [r for r in manifest["records"] if r["split"] == "test"]
    print(f"[DATA] Loaded {len(test_records)} test records across {len(classes)} classes.")

    # Pre-load all test images with identical preprocessing
    print("[PREPROCESS] Pre-loading 270 test images (200x200, RGB, normalized /255.0)...")
    images = []
    filenames = []
    true_classes = []
    for r in test_records:
        img_path = r["image_path"]
        filenames.append(os.path.basename(img_path))
        true_classes.append(r["primary_class"])
        with Image.open(img_path) as im:
            im_resized = im.resize((200, 200))
            im_arr = np.array(im_resized.convert("RGB"), dtype=np.float32) / 255.0
            images.append(im_arr)

    x_test = np.stack(images, axis=0)

    models_to_evaluate = [
        ("ResNet50_seed42", os.path.join("models", "checkpoints", "best_resnet50.keras"), "outputs/metrics/misclassified_resnet50_seed42.csv"),
        ("InceptionV3_seed42", os.path.join("models", "checkpoints", "best_inceptionv3.keras"), "outputs/metrics/misclassified_inceptionv3_seed42.csv"),
    ]

    for model_name, ckpt_path, out_csv in models_to_evaluate:
        if not os.path.exists(ckpt_path):
            print(f"[WARNING] Checkpoint {ckpt_path} not found. Skipping {model_name}.")
            continue

        print(f"\n[EVAL] Loading {model_name} from {ckpt_path}...")
        model = tf.keras.models.load_model(ckpt_path, compile=False)

        preds = model.predict(x_test, batch_size=32, verbose=1)
        pred_ids = np.argmax(preds, axis=1)

        misclassified = []
        correct = 0

        for i in range(len(test_records)):
            pred_class = classes[pred_ids[i]]
            conf = float(preds[i][pred_ids[i]])
            true_class = true_classes[i]

            if pred_class == true_class:
                correct += 1
            else:
                misclassified.append({
                    "file": filenames[i],
                    "true_class": true_class,
                    "predicted_class": pred_class,
                    "confidence": f"{conf:.6f}"
                })

        accuracy = correct / len(test_records)
        print(f"[{model_name}] Correct: {correct}/{len(test_records)} ({accuracy*100:.2f}%), Misclassified: {len(misclassified)}")

        # Write to CSV
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=["file", "true_class", "predicted_class", "confidence"])
            writer.writeheader()
            for row in misclassified:
                writer.writerow(row)
        print(f"[EXPORT] Saved {len(misclassified)} rows to {out_csv}")
        if misclassified:
            print("[SAMPLES] Misclassified samples:")
            for m in misclassified:
                print(f"  - {m['file']}: true={m['true_class']}, pred={m['predicted_class']}, conf={m['confidence']}")

    print("\n[NOTE] For seeds 101 and 202: checkpoints and per-image predictions were not saved during multi-seed evaluation (only aggregate accuracy recorded in experiments/seed_robustness_results.json).")

if __name__ == "__main__":
    evaluate_and_list_misclassified()
