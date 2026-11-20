"""
models/run_seed_experiments.py
Industrial Surface Defect Inspection Platform (CSAI3027P)
Test-accuracy sanity check: Re-running ResNet50 pipeline across 2 additional random seeds (Seed 101, Seed 202).
Verifies zero cross-split leakage for every seed (grouping duplicate hashes into train),
trains two-stage ResNet50 with frozen BatchNormalization, evaluates on holdout test split,
and computes the 95% Student's t confidence interval across all 3 seeds (42, 101, 202).
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))
import time
import json
import random
import hashlib
import glob
from collections import defaultdict
import numpy as np
import tensorflow as tf
from PIL import Image

def get_dataset_root():
    config_file = os.path.join("config", "dataset_path.txt")
    with open(config_file, "r") as f:
        rel_path = f.read().strip()
    return os.path.abspath(rel_path)

def compute_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def partition_dataset_for_seed(seed):
    dataset_root = get_dataset_root()
    images_dir = os.path.join(dataset_root, "IMAGES")
    image_paths = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
    
    classes = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
    class_to_id = {c: i for i, c in enumerate(classes)}
    
    class_groups = defaultdict(list)
    for p in image_paths:
        base = os.path.basename(p)
        stem = os.path.splitext(base)[0]
        c = stem.rsplit("_", 1)[0]
        class_groups[c].append({
            "path": p,
            "filename": base,
            "stem": stem,
            "class": c,
            "class_id": class_to_id[c],
            "sha256": compute_sha256(p)
        })
        
    train_records, val_records, test_records = [], [], []
    
    for c in classes:
        recs = class_groups[c]
        hash_to_recs = defaultdict(list)
        for r in recs:
            hash_to_recs[r["sha256"]].append(r)
            
        rng = random.Random(seed + class_to_id[c])
        train_cluster = []
        regular_clusters = []
        for h, cluster in hash_to_recs.items():
            if len(cluster) > 1:
                train_cluster.extend(cluster)
            else:
                regular_clusters.append(cluster[0])
                
        rng.shuffle(regular_clusters)
        
        n_train_needed = 210 - len(train_cluster)
        c_train = train_cluster + regular_clusters[:n_train_needed]
        c_val = regular_clusters[n_train_needed:n_train_needed + 45]
        c_test = regular_clusters[n_train_needed + 45:]
        
        assert len(c_train) == 210
        assert len(c_val) == 45
        assert len(c_test) == 45
        
        train_records.extend(c_train)
        val_records.extend(c_val)
        test_records.extend(c_test)
        
    # Verify zero leakage
    train_h = set(r["sha256"] for r in train_records)
    val_h = set(r["sha256"] for r in val_records)
    test_h = set(r["sha256"] for r in test_records)
    assert len(train_h.intersection(val_h)) == 0, "Train-Val leakage detected!"
    assert len(train_h.intersection(test_h)) == 0, "Train-Test leakage detected!"
    assert len(val_h.intersection(test_h)) == 0, "Val-Test leakage detected!"
    
    return train_records, val_records, test_records

def load_data_tensors(records):
    images = []
    labels = []
    for r in records:
        with Image.open(r["path"]) as im:
            im_resized = im.resize((200, 200))
            im_arr = np.array(im_resized.convert("RGB"), dtype=np.float32) / 255.0
        images.append(im_arr)
        labels.append(r["class_id"])
    x = np.array(images, dtype=np.float32)
    y = tf.keras.utils.to_categorical(np.array(labels), num_classes=6)
    return x, y

def train_and_eval_seed(seed, epochs_s1=3, epochs_s2=2):
    print(f"\n" + "=" * 65)
    print(f"RUNNING RESNET50 PIPELINE WITH RANDOM SEED {seed}")
    print("=" * 65)
    
    tf.keras.utils.set_random_seed(seed)
    
    train_recs, val_recs, test_recs = partition_dataset_for_seed(seed)
    print(f"[SPLIT SEED {seed}] Train: {len(train_recs)}, Val: {len(val_recs)}, Test: {len(test_recs)} (Zero leakage confirmed)")
    
    x_train, y_train = load_data_tensors(train_recs)
    x_val, y_val = load_data_tensors(val_recs)
    x_test, y_test = load_data_tensors(test_recs)
    
    # Build ResNet50
    base_model = tf.keras.applications.ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(200, 200, 3)
    )
    base_model.trainable = False
    
    inputs = tf.keras.layers.Input(shape=(200, 200, 3))
    x = tf.keras.layers.Rescaling(255.0)(inputs)
    x = tf.keras.applications.resnet50.preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(6, activation="softmax")(x)
    model = tf.keras.models.Model(inputs, outputs)
    
    # Stage 1: Feature Extraction
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    print(f"[STAGE 1] Training classification head ({epochs_s1} epochs)...")
    model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs_s1,
        batch_size=32,
        verbose=1
    )
    
    # Stage 2: Fine-Tuning conv5 with ALL BatchNormalization layers frozen
    print(f"[STAGE 2] Fine-tuning top conv5 blocks with frozen BatchNorm ({epochs_s2} epochs)...")
    base_model.trainable = True
    fine_tune_at = 143 # conv5 begins
    for i, layer in enumerate(base_model.layers):
        if i < fine_tune_at or isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
            
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True)
    ]
    model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs_s2,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate strictly on unseen test split
    print(f"[EVAL SEED {seed}] Evaluating on held-out test split (270 images)...")
    preds = model.predict(x_test, batch_size=32, verbose=0)
    y_pred = np.argmax(preds, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    acc = float(np.mean(y_pred == y_true))
    print(f"[RESULT SEED {seed}] Test Split Accuracy: {acc * 100:.2f}%")
    return acc

def run_seed_robustness_study():
    # Baseline seed 42 from primary executed run: 1.0000 (100.00%)
    seed_results = {
        "seed_42": 1.0000
    }
    
    seeds_to_run = [101, 202]
    for s in seeds_to_run:
        acc = train_and_eval_seed(s, epochs_s1=3, epochs_s2=2)
        seed_results[f"seed_{s}"] = round(acc, 4)
        
    accuracies = list(seed_results.values())
    n = len(accuracies)
    mean_acc = float(np.mean(accuracies))
    std_acc = float(np.std(accuracies, ddof=1)) # sample std
    
    # Student's t critical value for df=n-1=2 at 95% confidence (two-tailed) = 4.3027
    t_crit = 4.3027
    se = std_acc / np.sqrt(n)
    ci_lower = max(0.0, mean_acc - t_crit * se)
    ci_upper = min(1.0, mean_acc + t_crit * se)
    
    summary = {
        "seeds_evaluated": seed_results,
        "n_seeds": n,
        "mean_accuracy": round(mean_acc, 4),
        "mean_accuracy_pct": f"{mean_acc * 100:.2f}%",
        "std_accuracy": round(std_acc, 4),
        "standard_error": round(se, 4),
        "confidence_level": 0.95,
        "t_critical": t_crit,
        "ci_95": [round(ci_lower, 4), round(ci_upper, 4)],
        "ci_95_str": f"[{ci_lower*100:.2f}%, {ci_upper*100:.2f}%]",
        "leakage_audit_summary": {
            "duplicate_pair_isolated": ["patches_101.jpg", "patches_105.jpg"],
            "cross_split_hash_collisions": 0,
            "test_set_untouched_in_training": True
        }
    }
    
    out_file = os.path.join("experiments", "seed_robustness_results.json")
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
        
    print("\n" + "=" * 65)
    print("SEED ROBUSTNESS & TEST ACCURACY SUMMARY:")
    print("=" * 65)
    for k, v in seed_results.items():
        print(f"  - {k}: {v * 100:.2f}%")
    print(f"  - Mean Accuracy across {n} seeds: {mean_acc * 100:.2f}% (std: {std_acc * 100:.2f}%)")
    print(f"  - 95% Confidence Interval: {summary['ci_95_str']}")
    print(f"[SAVE] Saved robustness summary to {out_file}")

if __name__ == "__main__":
    run_seed_robustness_study()
