"""
models/dataset_loader.py
TensorFlow tf.data.Dataset Pipeline for Industrial Surface Defect Inspection:
1. Classification data loader: Resizes, converts to 3-channel RGB, normalizes [0, 1].
2. Safe defect-preserving data augmentation: flips, subtle rotation, slight contrast/brightness.
3. Multi-task Detection data loader: outputs classification targets and normalized bounding box targets.
4. Class weights computation for balanced empirical loss scaling.
"""

import os
import json
import numpy as np

def get_class_metadata():
    """Retrieve ordered class names and class-to-id mapping."""
    split_manifest = os.path.join("data", "splits", "split_manifest.json")
    with open(split_manifest, "r") as f:
        data = json.load(f)
    return data["classes"], data["class_to_id"]

def compute_class_weights(split="train"):
    """
    Compute inverse frequency class weights to address any slight class imbalance.
    Returns: dict mapping class_id -> float weight.
    """
    split_manifest = os.path.join("data", "splits", "split_manifest.json")
    with open(split_manifest, "r") as f:
        data = json.load(f)
        
    records = [r for r in data["records"] if r["split"] == split]
    class_counts = {}
    for r in records:
        cid = r["class_id"]
        class_counts[cid] = class_counts.get(cid, 0) + 1
        
    total_samples = len(records)
    n_classes = len(class_counts)
    class_weights = {}
    for cid, count in class_counts.items():
        # Balanced formula: N_total / (N_classes * count)
        class_weights[cid] = total_samples / (n_classes * count)
    return class_weights

def build_classification_dataset(split="train", img_size=(200, 200), batch_size=32, augment=False, seed=42):
    """
    Build high-performance tf.data.Dataset for defect classification.
    """
    import tensorflow as tf
    
    split_manifest = os.path.join("data", "splits", "split_manifest.json")
    with open(split_manifest, "r") as f:
        data = json.load(f)
        
    records = [r for r in data["records"] if r["split"] == split]
    image_paths = [os.path.abspath(r["image_path"]) for r in records]
    class_ids = [r["class_id"] for r in records]
    
    # One-hot encode targets
    num_classes = len(data["classes"])
    one_hot_labels = tf.keras.utils.to_categorical(class_ids, num_classes=num_classes)
    
    paths_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    labels_ds = tf.data.Dataset.from_tensor_slices(one_hot_labels)
    dataset = tf.data.Dataset.zip((paths_ds, labels_ds))
    
    def load_and_preprocess(path, label):
        # Read raw image file
        img_raw = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img_raw, channels=1)
        # Convert grayscale (1-channel) to 3-channel RGB for ImageNet transfer learning
        img = tf.image.grayscale_to_rgb(img)
        # Resize to model input dimension with bilinear interpolation
        img = tf.image.resize(img, img_size)
        # Rescale pixels from [0, 255] to [0.0, 1.0]
        img = tf.cast(img, tf.float32) / 255.0
        return img, label

    # Map loading
    dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Augmentation layers (applied only if augment=True and on training split)
    if augment and split == "train":
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=seed),
            # Small rotation (+/- 5%) to maintain defect linearity
            tf.keras.layers.RandomRotation(factor=0.05, fill_mode="reflect", seed=seed),
            # Gentle contrast (+/- 10%)
            tf.keras.layers.RandomContrast(factor=0.1, seed=seed),
        ], name="safe_defect_augmentation")
        
        dataset = dataset.map(lambda x, y: (data_augmentation(x, training=True), y),
                              num_parallel_calls=tf.data.AUTOTUNE)

    if split == "train":
        dataset = dataset.shuffle(buffer_size=min(len(records), 1000), seed=seed, reshuffle_each_iteration=True)
        
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def build_detection_dataset(split="train", img_size=(200, 200), batch_size=16, seed=42):
    """
    Build multi-task detection dataset yielding:
    - input: (Batch, H, W, 3)
    - targets: {"class_output": (Batch, 6), "box_output": (Batch, 4) in [ymin, xmin, ymax, xmax]}
    Dominant defect box (largest surface area) is selected as primary localization target.
    """
    import tensorflow as tf
    
    split_manifest = os.path.join("data", "splits", "split_manifest.json")
    with open(split_manifest, "r") as f:
        data = json.load(f)
        
    records = [r for r in data["records"] if r["split"] == split]
    num_classes = len(data["classes"])
    
    image_paths = []
    class_labels = []
    box_targets = []
    
    for r in records:
        image_paths.append(os.path.abspath(r["image_path"]))
        class_labels.append(r["class_id"])
        
        # Select dominant box by area (norm_xmax - norm_xmin) * (norm_ymax - norm_ymin)
        boxes = r["boxes"]
        if len(boxes) > 0:
            # Sort by area descending
            sorted_boxes = sorted(boxes, key=lambda b: (b["norm_xmax"] - b["norm_xmin"]) * (b["norm_ymax"] - b["norm_ymin"]), reverse=True)
            dom = sorted_boxes[0]
            # Standard TensorFlow box format: [ymin, xmin, ymax, xmax] in [0, 1]
            box_targets.append([dom["norm_ymin"], dom["norm_xmin"], dom["norm_ymax"], dom["norm_xmax"]])
        else:
            # Full image box if no annotation
            box_targets.append([0.0, 0.0, 1.0, 1.0])
            
    one_hot_classes = tf.keras.utils.to_categorical(class_labels, num_classes=num_classes)
    box_targets = np.array(box_targets, dtype=np.float32)
    
    paths_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    class_ds = tf.data.Dataset.from_tensor_slices(one_hot_classes)
    box_ds = tf.data.Dataset.from_tensor_slices(box_targets)
    dataset = tf.data.Dataset.zip((paths_ds, class_ds, box_ds))
    
    def load_img_and_targets(path, cls_target, box_target):
        img_raw = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img_raw, channels=1)
        img = tf.image.grayscale_to_rgb(img)
        img = tf.image.resize(img, img_size)
        img = tf.cast(img, tf.float32) / 255.0
        return img, {"class_output": cls_target, "box_output": box_target}

    dataset = dataset.map(load_img_and_targets, num_parallel_calls=tf.data.AUTOTUNE)
    if split == "train":
        dataset = dataset.shuffle(buffer_size=min(len(records), 1000), seed=seed)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset
