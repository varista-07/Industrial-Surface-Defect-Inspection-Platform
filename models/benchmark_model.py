"""
models/benchmark_model.py
Benchmark InceptionV3 Transfer Learning Model on NEU-DET (TensorFlow / Keras):
1. InceptionV3 backbone with ImageNet pretrained weights.
2. Trained on identical 70/15/15 stratified data splits.
3. Evaluated on the TEST split for direct architectural benchmark against ResNet50 and MLP.
4. Generates:
   - models/checkpoints/best_inceptionv3.keras
   - outputs/figures/inceptionv3_learning_curves.png
   - outputs/metrics/inceptionv3_test_metrics.json
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
from sklearn.metrics import classification_report, confusion_matrix, f1_score

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PYTHONHASHSEED"] = "42"

import tensorflow as tf
tf.keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()

from models.dataset_loader import build_classification_dataset, get_class_metadata

def build_inceptionv3_model(input_shape=(200, 200, 3), num_classes=6):
    """
    Build InceptionV3 architecture with custom classification head.
    """
    base_model = tf.keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )
    base_model.trainable = False
    
    inputs = tf.keras.Input(shape=input_shape, name="surface_image_input")
    x = tf.keras.layers.Rescaling(255.0)(inputs)
    x = tf.keras.applications.inception_v3.preprocess_input(x)
    
    features = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(features)
    x = tf.keras.layers.BatchNormalization(name="head_bn")(x)
    x = tf.keras.layers.Dense(256, activation="relu", name="dense_features")(x)
    x = tf.keras.layers.Dropout(0.4, seed=42, name="dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", dtype="float32", name="defect_probabilities")(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="InceptionV3_Benchmark")
    return model, base_model

def train_and_evaluate_inceptionv3(img_size=(200, 200), batch_size=32, epochs=12):
    print("=" * 65)
    print("INCEPTIONV3 BENCHMARK MODEL TRAINING")
    print("=" * 65)
    
    classes, _ = get_class_metadata()
    num_classes = len(classes)
    
    train_ds = build_classification_dataset("train", img_size=img_size, batch_size=batch_size, augment=True)
    val_ds = build_classification_dataset("val", img_size=img_size, batch_size=batch_size, augment=False)
    test_ds = build_classification_dataset("test", img_size=img_size, batch_size=batch_size, augment=False)
    
    model, base_model = build_inceptionv3_model(input_shape=(img_size[0], img_size[1], 3), num_classes=num_classes)
    os.makedirs(os.path.join("models", "checkpoints"), exist_ok=True)
    checkpoint_path = os.path.join("models", "checkpoints", "best_inceptionv3.keras")
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor="val_accuracy", save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5)
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
    print(f"\n[TRAIN] InceptionV3 training complete in {duration:.1f} seconds.")
    
    model.save(checkpoint_path)
    
    # Plot learning curves
    os.makedirs(os.path.join("outputs", "figures"), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    
    axes[0].plot(history.history["loss"], label="Train Loss", color="#1f77b4", lw=2)
    axes[0].plot(history.history["val_loss"], label="Val Loss", color="#ff7f0e", lw=2)
    axes[0].set_title("InceptionV3 Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend()
    
    axes[1].plot([a * 100 for a in history.history["accuracy"]], label="Train Acc (%)", color="#2ca02c", lw=2)
    axes[1].plot([va * 100 for va in history.history["val_accuracy"]], label="Val Acc (%)", color="#d62728", lw=2)
    axes[1].set_title("InceptionV3 Accuracy (%)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend()
    
    fig_path = os.path.join("outputs", "figures", "inceptionv3_learning_curves.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[PLOT] Saved InceptionV3 curves to {fig_path}")
    
    # Test evaluation
    y_true_list, y_pred_probs, latencies = [], [], []
    for x_batch, y_batch in test_ds:
        for i in range(len(x_batch)):
            single = tf.expand_dims(x_batch[i], axis=0)
            t0 = time.perf_counter()
            pred = model.predict(single, verbose=0)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)
            y_pred_probs.append(pred[0])
            y_true_list.append(np.argmax(y_batch[i].numpy()))
            
    y_true = np.array(y_true_list)
    y_pred = np.argmax(np.array(y_pred_probs), axis=1)
    
    test_acc = float(np.mean(y_pred == y_true))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    mean_latency = float(np.mean(latencies[5:]))
    fps = float(1000.0 / mean_latency)
    
    print(f"\n[BENCHMARK] InceptionV3 Test Accuracy: {test_acc*100:.2f}% | Macro F1: {macro_f1:.4f} | Latency: {mean_latency:.2f} ms ({fps:.1f} FPS)")
    
    clf_rep = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    
    # Save Metrics JSON
    metrics_export = {
        "model_name": "InceptionV3 Benchmark",
        "timestamp": datetime.datetime.now().isoformat(),
        "input_resolution": f"{img_size[0]}x{img_size[1]}x3",
        "hyperparameters": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": 1e-3,
            "optimizer": "Adam"
        },
        "test_metrics": {
            "accuracy": test_acc,
            "macro_f1": macro_f1,
            "per_class": {cls: clf_rep[cls] for cls in classes},
            "mean_latency_ms": mean_latency,
            "throughput_fps": fps
        }
    }
    metrics_path = os.path.join("outputs", "metrics", "inceptionv3_test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_export, f, indent=2)
    print(f"[SAVE] Exported InceptionV3 metrics to {metrics_path}")
    
    # Log to CSV
    log_csv = os.path.join("experiments", "experiment_log.csv")
    with open(log_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "InceptionV3",
            "Transfer Learning (Frozen)",
            epochs,
            batch_size,
            "1e-3",
            True,
            False,
            f"{history.history['accuracy'][-1]*100:.2f}%",
            f"{history.history['val_accuracy'][-1]*100:.2f}%",
            f"{test_acc*100:.2f}%",
            f"{macro_f1:.4f}",
            f"{mean_latency:.2f}",
            f"{fps:.1f}",
            "Comparative benchmark against ResNet50 and MLP baseline"
        ])
    return model, metrics_export

if __name__ == "__main__":
    train_and_evaluate_inceptionv3(img_size=(200, 200), batch_size=32, epochs=6)
