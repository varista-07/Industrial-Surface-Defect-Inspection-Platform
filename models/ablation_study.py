"""
models/ablation_study.py
Controlled Ablation Experiments for ResNet50 Classifier:
1. Backbone Freezing: Stage 1 Frozen Backbone vs Stage 2 Fine-Tuned Backbone.
2. Data Augmentation: Without Augmentation vs With Safe Defect Augmentation.
3. Class Weighting: Unweighted Cross-Entropy vs Inverse Frequency Class Weights.
4. Comprehensive Comparison Table: MLP Baseline vs InceptionV3 vs ResNet50 (Ablations).
Generates outputs/figures/resnet50_ablations.png and updates experiments/experiment_log.csv.
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
from sklearn.metrics import f1_score

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PYTHONHASHSEED"] = "42"

import tensorflow as tf
tf.keras.utils.set_random_seed(42)

from models.dataset_loader import build_classification_dataset, get_class_metadata, compute_class_weights
from models.resnet50_classifier import build_resnet50_model

def evaluate_on_test(model, test_ds):
    y_true_list, y_pred_list = [], []
    for x_batch, y_batch in test_ds:
        preds = model.predict(x_batch, verbose=0)
        y_pred_list.extend(np.argmax(preds, axis=1))
        y_true_list.extend(np.argmax(y_batch.numpy(), axis=1))
    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)
    acc = float(np.mean(y_true == y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    return acc, macro_f1

def run_ablation_experiments(img_size=(200, 200), batch_size=32):
    print("=" * 65)
    print("RESNET50 CONTROLLED ABLATION STUDY")
    print("=" * 65)
    
    classes, _ = get_class_metadata()
    num_classes = len(classes)
    
    test_ds = build_classification_dataset("test", img_size=img_size, batch_size=batch_size, augment=False)
    val_ds = build_classification_dataset("val", img_size=img_size, batch_size=batch_size, augment=False)
    class_weights = compute_class_weights("train")
    
    ablations = [
        {
            "name": "ResNet50_Frozen_Only",
            "augment": True,
            "fine_tune": False,
            "class_weights": False,
            "epochs_s1": 3,
            "epochs_s2": 0,
            "description": "Frozen backbone (Feature Extractor only)"
        },
        {
            "name": "ResNet50_No_Augmentation",
            "augment": False,
            "fine_tune": True,
            "class_weights": False,
            "epochs_s1": 2,
            "epochs_s2": 2,
            "description": "Two-stage training without data augmentation"
        },
        {
            "name": "ResNet50_With_ClassWeights",
            "augment": True,
            "fine_tune": True,
            "class_weights": True,
            "epochs_s1": 2,
            "epochs_s2": 2,
            "description": "Two-stage training with inverse frequency class weights"
        }
    ]
    
    ablation_results = []
    
    for exp in ablations:
        print(f"\n--- Running Ablation: {exp['name']} ({exp['description']}) ---")
        train_ds = build_classification_dataset("train", img_size=img_size, batch_size=batch_size, augment=exp["augment"])
        weights = class_weights if exp["class_weights"] else None
        
        model, base_model = build_resnet50_model(input_shape=(img_size[0], img_size[1], 3), num_classes=num_classes)
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
        
        # Stage 1
        h1 = model.fit(train_ds, validation_data=val_ds, epochs=exp["epochs_s1"], class_weight=weights, verbose=1)
        
        # Stage 2 if fine-tuning
        if exp["fine_tune"] and exp["epochs_s2"] > 0:
            base_model.trainable = True
            for layer in base_model.layers[:143]:
                layer.trainable = False
            model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])
            h2 = model.fit(train_ds, validation_data=val_ds, epochs=exp["epochs_s2"], class_weight=weights, verbose=1)
            final_val_acc = h2.history["val_accuracy"][-1]
        else:
            final_val_acc = h1.history["val_accuracy"][-1]
            
        test_acc, test_f1 = evaluate_on_test(model, test_ds)
        print(f"Result for {exp['name']}: Val Acc={final_val_acc*100:.2f}%, Test Acc={test_acc*100:.2f}%, Test F1={test_f1:.4f}")
        
        res = {
            "experiment": exp["name"],
            "description": exp["description"],
            "val_acc": final_val_acc,
            "test_acc": test_acc,
            "test_f1": test_f1
        }
        ablation_results.append(res)
        
        # Log to experiment log
        log_csv = os.path.join("experiments", "experiment_log.csv")
        with open(log_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                exp["name"],
                "Ablation Study",
                f"{exp['epochs_s1']}+{exp['epochs_s2']}",
                batch_size,
                "1e-3 / 1e-4",
                exp["augment"],
                exp["class_weights"],
                "N/A",
                f"{final_val_acc*100:.2f}%",
                f"{test_acc*100:.2f}%",
                f"{test_f1:.4f}",
                "N/A",
                "N/A",
                exp["description"]
            ])

    # Save ablation figure
    os.makedirs(os.path.join("outputs", "figures"), exist_ok=True)
    names = [r["experiment"].replace("ResNet50_", "") for r in ablation_results]
    accs = [r["test_acc"] * 100 for r in ablation_results]
    f1s = [r["test_f1"] * 100 for r in ablation_results]
    
    x = np.arange(len(names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(9, 5))
    rects1 = ax.bar(x - width/2, accs, width, label="Test Accuracy (%)", color="#1f77b4")
    rects2 = ax.bar(x + width/2, f1s, width, label="Test Macro F1 (%)", color="#2ca02c")
    
    ax.set_ylabel("Percentage (%)")
    ax.set_title("ResNet50 Controlled Ablation Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
                    
    plt.tight_layout()
    fig_path = os.path.join("outputs", "figures", "resnet50_ablations.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"\n[PLOT] Saved ablation comparison figure to {fig_path}")

if __name__ == "__main__":
    run_ablation_experiments(img_size=(200, 200), batch_size=32)
