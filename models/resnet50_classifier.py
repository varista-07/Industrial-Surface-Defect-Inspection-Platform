"""
models/resnet50_classifier.py
Two-Stage Transfer Learning with ResNet50 on NEU-DET (TensorFlow / Keras):
1. Stage 1: Feature Extraction with frozen ImageNet backbone.
2. Stage 2: Fine-tuning top convolutional blocks (conv5_block*) with reduced LR.
3. Rigorous Evaluation on the TEST split (270 unseen images):
   - Accuracy, Macro F1, Per-class Precision / Recall / F1.
   - Confusion Matrix visualization saved to outputs/figures/resnet50_confusion_matrix.png.
   - Stage 1 + Stage 2 Learning Curves saved to outputs/figures/resnet50_learning_curves.png.
   - Inference latency and throughput (FPS) benchmark on CPU.
4. Model weights saved to models/checkpoints/best_resnet50.keras.
5. Logs experiment configuration and metrics to experiments/experiment_log.csv.
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
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# Set environment variables for reproducibility and cleaner logging
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PYTHONHASHSEED"] = "42"

import tensorflow as tf
# Ensure seeds are pinned to 42
tf.keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()

from models.dataset_loader import build_classification_dataset, get_class_metadata, compute_class_weights

def build_resnet50_model(input_shape=(200, 200, 3), num_classes=6):
    """
    Build ResNet50 model with custom classification head for industrial defects.
    """
    base_model = tf.keras.applications.ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )
    # Stage 1: Freeze all base backbone layers
    base_model.trainable = False
    
    inputs = tf.keras.Input(shape=input_shape, name="surface_image_input")
    # ResNet50 preprocess_input expects inputs in range [0, 255] or handles scaling
    # Our data loader delivers [0, 1], so we scale to [0, 255] then apply resnet50 preprocess
    x = tf.keras.layers.Rescaling(255.0)(inputs)
    x = tf.keras.applications.resnet50.preprocess_input(x)
    
    features = base_model(x, training=False)
    
    # Classification head
    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(features)
    x = tf.keras.layers.BatchNormalization(name="head_bn")(x)
    x = tf.keras.layers.Dense(256, activation="relu", name="dense_features")(x)
    x = tf.keras.layers.Dropout(0.4, seed=42, name="dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", dtype="float32", name="defect_probabilities")(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="ResNet50_DefectClassifier")
    return model, base_model

def train_and_evaluate_resnet50(
    img_size=(200, 200),
    batch_size=32,
    stage1_epochs=8,
    stage2_epochs=10,
    use_augmentation=True,
    use_class_weights=False
):
    print("=" * 65)
    print("RESNET50 TWO-STAGE TRANSFER LEARNING PIPELINE")
    print("=" * 65)
    
    classes, class_to_id = get_class_metadata()
    num_classes = len(classes)
    
    # 1. Build Datasets
    print("\n[DATA] Building tf.data pipelines...")
    train_ds = build_classification_dataset("train", img_size=img_size, batch_size=batch_size, augment=use_augmentation)
    val_ds = build_classification_dataset("val", img_size=img_size, batch_size=batch_size, augment=False)
    test_ds = build_classification_dataset("test", img_size=img_size, batch_size=batch_size, augment=False)
    
    class_weights = compute_class_weights("train") if use_class_weights else None
    if class_weights:
        print(f"[DATA] Using class weights: {class_weights}")
        
    # 2. Build Model
    model, base_model = build_resnet50_model(input_shape=(img_size[0], img_size[1], 3), num_classes=num_classes)
    os.makedirs(os.path.join("models", "checkpoints"), exist_ok=True)
    checkpoint_path = os.path.join("models", "checkpoints", "best_resnet50.keras")
    
    # -------------------------------------------------------------
    # STAGE 1: Feature Extraction (Frozen Backbone)
    # -------------------------------------------------------------
    print("\n" + "-" * 60)
    print(f"STAGE 1: Training Classification Head ({stage1_epochs} epochs, lr=1e-3)")
    print("-" * 60)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    callbacks_stage1 = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5)
    ]
    
    start_time = time.time()
    history_s1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=stage1_epochs,
        class_weight=class_weights,
        callbacks=callbacks_stage1,
        verbose=1
    )
    
    # -------------------------------------------------------------
    # STAGE 2: Fine-Tuning (Unfreeze conv5_block* layers)
    # -------------------------------------------------------------
    print("\n" + "-" * 60)
    print(f"STAGE 2: Fine-Tuning Top Conv Blocks ({stage2_epochs} epochs, lr=1e-4)")
    print("-" * 60)
    
    base_model.trainable = True
    # Freeze all layers except conv5_block*, and keep ALL BatchNormalization layers frozen
    # Freezing BN prevents batch statistic disruption on small custom domain batches
    fine_tune_at = 143  # ResNet50 has 175 layers; layer 143 begins conv5_block1
    for i, layer in enumerate(base_model.layers):
        if i < fine_tune_at or isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        
    trainable_count = sum(int(np.prod(v.shape)) for v in model.trainable_weights)
    print(f"[STAGE 2] Unfrozen conv5 blocks with all BatchNormalization layers frozen. Trainable weights: {trainable_count:,}")
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    callbacks_stage2 = [
        tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor="val_accuracy", save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6)
    ]
    
    history_s2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=stage2_epochs,
        class_weight=class_weights,
        callbacks=callbacks_stage2,
        verbose=1
    )
    training_duration = time.time() - start_time
    print(f"\n[TRAIN] Two-stage training complete in {training_duration:.1f} seconds.")
    
    # Save final model if checkpoint not created or to ensure full model format
    model.save(checkpoint_path)
    print(f"[SAVE] Saved best ResNet50 model to {checkpoint_path}")
    
    # Combine training curves
    acc = history_s1.history["accuracy"] + history_s2.history["accuracy"]
    val_acc = history_s1.history["val_accuracy"] + history_s2.history["val_accuracy"]
    loss = history_s1.history["loss"] + history_s2.history["loss"]
    val_loss = history_s1.history["val_loss"] + history_s2.history["val_loss"]
    
    # Plot Learning Curves
    os.makedirs(os.path.join("outputs", "figures"), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(loss, label="Training Loss", color="#1f77b4", lw=2)
    axes[0].plot(val_loss, label="Validation Loss", color="#ff7f0e", lw=2)
    axes[0].axvline(len(history_s1.history["loss"]) - 1, color="gray", linestyle="--", label="Start Fine-Tuning")
    axes[0].set_title("ResNet50 Cross-Entropy Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend()
    
    axes[1].plot([a * 100 for a in acc], label="Training Accuracy (%)", color="#2ca02c", lw=2)
    axes[1].plot([va * 100 for va in val_acc], label="Validation Accuracy (%)", color="#d62728", lw=2)
    axes[1].axvline(len(history_s1.history["loss"]) - 1, color="gray", linestyle="--", label="Start Fine-Tuning")
    axes[1].set_title("ResNet50 Accuracy (%)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend()
    
    curve_fig_path = os.path.join("outputs", "figures", "resnet50_learning_curves.png")
    plt.savefig(curve_fig_path, dpi=300)
    plt.close()
    print(f"[PLOT] Saved learning curves to {curve_fig_path}")
    
    # -------------------------------------------------------------
    # 3. Comprehensive Evaluation on TEST Split
    # -------------------------------------------------------------
    print("\n" + "=" * 65)
    print("EVALUATING FINAL MODEL ON TEST SPLIT (270 Unseen Images)")
    print("=" * 65)
    
    # Collect all test predictions and true labels
    y_true_list = []
    y_pred_probs = []
    
    # Latency benchmark
    latencies = []
    for x_batch, y_batch in test_ds:
        for i in range(len(x_batch)):
            single_img = tf.expand_dims(x_batch[i], axis=0)
            t0 = time.perf_counter()
            pred = model.predict(single_img, verbose=0)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0) # ms
            
            y_pred_probs.append(pred[0])
            y_true_list.append(np.argmax(y_batch[i].numpy()))
            
    y_true = np.array(y_true_list)
    y_pred_probs = np.array(y_pred_probs)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    test_accuracy = float(np.mean(y_pred == y_true))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    mean_latency = float(np.mean(latencies[5:])) # exclude warmup
    fps = float(1000.0 / mean_latency)
    
    print(f"Test Accuracy:  {test_accuracy * 100:.2f}%")
    print(f"Test Macro F1:  {macro_f1:.4f}")
    print(f"Mean Latency:   {mean_latency:.2f} ms per image ({fps:.1f} FPS on CPU)")
    
    # Classification Report
    clf_report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    print("\nPer-Class Performance Report:")
    for cls_name in classes:
        p = clf_report[cls_name]["precision"]
        r = clf_report[cls_name]["recall"]
        f = clf_report[cls_name]["f1-score"]
        print(f"  - {cls_name:16s} | Precision: {p*100:5.1f}% | Recall: {r*100:5.1f}% | F1: {f*100:5.1f}%")

    # False-Reject Rate (FRR) & Triage Analysis across confidence thresholds
    # Definition 1: Missed Defect Rate (defective sample falling below acceptance threshold)
    # Definition 2: Triage Escalation Rate (samples routed to manual visual inspection)
    max_confs = np.max(y_pred_probs, axis=1)
    thresholds = [0.50, 0.60, 0.70, 0.80, 0.90]
    frr_analysis = {}
    print("\nIndustrial False-Reject Rate (FRR) & Manual Triage Analysis:")
    for th in thresholds:
        uncertain_count = int(np.sum(max_confs < th))
        triage_rate = float(uncertain_count / len(y_true))
        frr_analysis[f"threshold_{th:.2f}"] = {
            "threshold": th,
            "uncertain_samples": uncertain_count,
            "triage_escalation_rate": triage_rate,
            "missed_defect_rate": triage_rate
        }
        print(f"  - Threshold {th:.2f}: {uncertain_count}/{len(y_true)} flagged samples ({triage_rate*100:.2f}% triage rate)")
        
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6.5))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("ResNet50 Confusion Matrix (Test Split)", fontsize=13)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=35, ha="right", fontsize=10)
    plt.yticks(tick_marks, classes, fontsize=10)
    
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontweight="bold")
    plt.ylabel("True Ground Truth Class", fontsize=11)
    plt.xlabel("Predicted Class", fontsize=11)
    plt.tight_layout()
    cm_fig_path = os.path.join("outputs", "figures", "resnet50_confusion_matrix.png")
    plt.savefig(cm_fig_path, dpi=300)
    plt.close()
    print(f"[PLOT] Saved confusion matrix to {cm_fig_path}")
    
    # 4. Save Metrics JSON
    os.makedirs(os.path.join("outputs", "metrics"), exist_ok=True)
    metrics_export = {
        "model_name": "ResNet50 Two-Stage Transfer Learning",
        "timestamp": datetime.datetime.now().isoformat(),
        "input_resolution": f"{img_size[0]}x{img_size[1]}x3",
        "hyperparameters": {
            "stage1_epochs": stage1_epochs,
            "stage2_epochs": stage2_epochs,
            "batch_size": batch_size,
            "stage1_lr": 1e-3,
            "stage2_lr": 1e-4,
            "optimizer": "Adam",
            "augmentation": use_augmentation,
            "class_weights": use_class_weights
        },
        "test_metrics": {
            "accuracy": test_accuracy,
            "macro_f1": macro_f1,
            "per_class": {cls: clf_report[cls] for cls in classes},
            "mean_latency_ms": mean_latency,
            "throughput_fps": fps,
            "confusion_matrix": cm.tolist(),
            "false_reject_rate_analysis": frr_analysis
        }
    }
    metrics_path = os.path.join("outputs", "metrics", "resnet50_test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_export, f, indent=2)
    print(f"[SAVE] Exported ResNet50 metrics to {metrics_path}")
    
    # 5. Log to experiments/experiment_log.csv
    log_csv_path = os.path.join("experiments", "experiment_log.csv")
    file_exists = os.path.exists(log_csv_path)
    with open(log_csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp", "model_architecture", "stage", "epochs", "batch_size",
                "lr", "augmentation", "class_weights", "train_acc", "val_acc",
                "test_acc", "test_macro_f1", "latency_ms", "throughput_fps", "notes"
            ])
        writer.writerow([
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "ResNet50",
            "Two-Stage (Frozen+FineTune)",
            f"{stage1_epochs}+{stage2_epochs}",
            batch_size,
            "1e-3 -> 1e-4",
            use_augmentation,
            use_class_weights,
            f"{acc[-1]*100:.2f}%",
            f"{val_acc[-1]*100:.2f}%",
            f"{test_accuracy*100:.2f}%",
            f"{macro_f1:.4f}",
            f"{mean_latency:.2f}",
            f"{fps:.1f}",
            "Primary classifier for web inspection platform"
        ])
    print(f"[LOG] Logged experiment to {log_csv_path}")
    return model, metrics_export

if __name__ == "__main__":
    train_and_evaluate_resnet50(
        img_size=(200, 200),
        batch_size=32,
        stage1_epochs=4,
        stage2_epochs=4,
        use_augmentation=True,
        use_class_weights=False
    )
