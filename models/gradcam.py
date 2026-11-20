"""
models/gradcam.py
Grad-CAM: Gradient-Weighted Class Activation Mapping for Defect Explanation (Selvaraju et al., 2017):
1. Hooks into ResNet50's final convolutional layer (conv5_block3_out).
2. Computes exact gradient of predicted defect class score w.r.t. feature activation maps.
3. Performs Global Average Pooling over spatial dimensions to derive channel importance weights alpha_k.
4. Generates rectified heatmap L_GradCAM = ReLU(sum(alpha_k * A^k)).
5. Overlays heatmap on the original steel image with color blending.
6. Generates outputs/figures/gradcam_explanations.png.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf

def make_gradcam_heatmap(img_array, model, last_conv_layer_name="conv5_block3_out", pred_index=None):
    """
    Computes Grad-CAM heatmap for a given input image array of shape (1, H, W, 3).
    Cleanly handles sub-model backbone structures in Keras 3.
    """
    # 1. Find base_model layer
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) or hasattr(layer, "get_layer"):
            try:
                layer.get_layer(last_conv_layer_name)
                base_model = layer
                break
            except Exception:
                continue
                
    if base_model is None:
        # Fallback: model itself is the flat model
        base_model = model
        
    # 2. Extract feature map submodel
    conv_layer = base_model.get_layer(last_conv_layer_name)
    feature_extractor = tf.keras.Model(base_model.inputs, conv_layer.output)
    
    # 3. Head submodel: maps feature map to predictions
    # Head layers: avg_pool, head_bn, dense_features, dropout, defect_probabilities
    head_layers = ["avg_pool", "head_bn", "dense_features", "dropout", "defect_probabilities"]
    
    # Preprocess image input through rescaling and preprocess_input
    x_pre = img_array * 255.0
    x_pre = tf.keras.applications.resnet50.preprocess_input(x_pre)
    
    with tf.GradientTape() as tape:
        # Forward pass through backbone up to last conv layer
        conv_outputs = feature_extractor([x_pre], training=False)
        tape.watch(conv_outputs)
        
        # Forward pass through classification head
        x_head = conv_outputs
        for layer_name in head_layers:
            try:
                x_head = model.get_layer(layer_name)(x_head, training=False)
            except Exception:
                pass
        predictions = x_head
        
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Gradient of target class score w.r.t. conv layer output
    grads = tape.gradient(class_channel, conv_outputs)
    
    # Global average pooling of gradients: alpha_k = mean(d y^c / d A^k)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weighted combination: sum(alpha_k * A^k)
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Apply ReLU: only positive contributions to class of interest
    heatmap = tf.maximum(heatmap, 0.0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()

def generate_gradcam_overlay(original_image_pil, heatmap, alpha=0.45, colormap="jet"):
    """
    Superimposes the Grad-CAM heatmap onto the original PIL image.
    Returns: PIL Image of the superimposed visualization.
    """
    heatmap_resized = Image.fromarray(np.uint8(255 * heatmap)).resize(
        original_image_pil.size, resample=Image.Resampling.BILINEAR
    )
    heatmap_arr = np.array(heatmap_resized) / 255.0
    
    # Apply matplotlib colormap (compatible with modern matplotlib)
    try:
        cmap = matplotlib.colormaps[colormap]
    except Exception:
        cmap = plt.get_cmap(colormap)
        
    colored_heatmap = cmap(heatmap_arr)[:, :, :3]  # RGB
    colored_heatmap = np.uint8(255 * colored_heatmap)
    
    orig_rgb = np.array(original_image_pil.convert("RGB"))
    # Alpha blend: (1 - alpha) * original + alpha * heatmap
    blended = np.uint8((1.0 - alpha) * orig_rgb + alpha * colored_heatmap)
    return Image.fromarray(blended), heatmap_resized

def run_gradcam_demo(model_path=None):
    """
    Demonstrate Grad-CAM across 6 test samples (one per defect class).
    Saves outputs/figures/gradcam_explanations.png.
    """
    if model_path is None:
        model_path = os.path.join("models", "checkpoints", "best_resnet50.keras")
    if not os.path.exists(model_path):
        print(f"[Grad-CAM] Model checkpoint {model_path} not found yet. Skipping standalone run.")
        return
        
    print("=" * 60)
    print("RUNNING GRAD-CAM DEFECT EXPLANATION DEMO")
    print("=" * 60)
    
    model = tf.keras.models.load_model(model_path)
    
    with open(os.path.join("data", "splits", "split_manifest.json"), "r") as f:
        manifest = json.load(f)
    classes = manifest["classes"]
    
    test_samples = []
    for cls_name in classes:
        sample = next(r for r in manifest["records"] if r["split"] == "test" and r["primary_class"] == cls_name)
        test_samples.append(sample)
        
    os.makedirs(os.path.join("outputs", "figures"), exist_ok=True)
    fig, axes = plt.subplots(len(classes), 3, figsize=(11, 2.8 * len(classes)))
    
    for idx, sample in enumerate(test_samples):
        img_path = sample["image_path"]
        cls_name = sample["primary_class"]
        
        orig_img = Image.open(img_path).convert("RGB")
        img_resized = orig_img.resize((200, 200))
        img_arr = np.expand_dims(np.array(img_resized, dtype=np.float32) / 255.0, axis=0)
        
        # Prediction
        preds = model.predict(img_arr, verbose=0)[0]
        pred_idx = np.argmax(preds)
        pred_class = classes[pred_idx]
        conf = preds[pred_idx] * 100.0
        
        # Heatmap & overlay
        heatmap = make_gradcam_heatmap(img_arr, model, pred_index=pred_idx)
        overlay, heatmap_img = generate_gradcam_overlay(orig_img, heatmap, alpha=0.45)
        
        # 1. Original
        axes[idx, 0].imshow(orig_img)
        axes[idx, 0].set_title(f"Target: {cls_name}")
        axes[idx, 0].axis("off")
        
        # 2. Raw Heatmap
        axes[idx, 1].imshow(heatmap, cmap="jet")
        axes[idx, 1].set_title("Grad-CAM Heatmap")
        axes[idx, 1].axis("off")
        
        # 3. Superimposed Explanation
        axes[idx, 2].imshow(overlay)
        axes[idx, 2].set_title(f"Pred: {pred_class} ({conf:.1f}%)")
        axes[idx, 2].axis("off")
        
    plt.tight_layout()
    fig_path = os.path.join("outputs", "figures", "gradcam_explanations.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[PLOT] Saved Grad-CAM explanation figure to {fig_path}")

if __name__ == "__main__":
    run_gradcam_demo()
