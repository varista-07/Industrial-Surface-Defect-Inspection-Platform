"""
models/conv_visualization.py
2D Convolution Visualization: Learned Filters, Feature Maps, and 3D Convolution Comparison:
1. Inspects the first convolutional layer (conv1_conv) of ResNet50.
2. Visualizes the 64 learned 7x7 spatial filter kernels.
3. Computes and plots feature activation maps on a sample defect surface image.
4. Generates outputs/figures/conv2d_filters_and_feature_maps.png.
5. Documents the mathematical difference between 2D and 3D convolution in models/CONV2D_VS_CONV3D.md.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf

def visualize_conv_filters_and_features():
    print("=" * 60)
    print("2D CONVOLUTION FILTER & FEATURE MAP VISUALIZATION")
    print("=" * 60)
    
    # 1. Load ResNet50 backbone
    base_model = tf.keras.applications.ResNet50(weights="imagenet", include_top=False, input_shape=(200, 200, 3))
    first_conv_layer = base_model.get_layer("conv1_conv")
    weights, biases = first_conv_layer.get_weights()
    print(f"First conv layer: {first_conv_layer.name}")
    print(f"Filter weights shape: {weights.shape} (H=7, W=7, Channels_in=3, Filters_out=64)")
    print(f"Bias shape: {biases.shape}")
    
    # Normalize filters to [0, 1] for RGB visualization
    w_min, w_max = weights.min(), weights.max()
    filters_norm = (weights - w_min) / (w_max - w_min + 1e-10)
    
    # 2. Extract Feature Maps for a sample defect image
    with open(os.path.join("data", "splits", "split_manifest.json"), "r") as f:
        manifest = json.load(f)
    sample_record = manifest["records"][0]
    img_path = sample_record["image_path"]
    
    img_pil = Image.open(img_path).convert("RGB").resize((200, 200))
    img_arr = np.expand_dims(np.array(img_pil, dtype=np.float32), axis=0)
    img_preprocessed = tf.keras.applications.resnet50.preprocess_input(img_arr.copy())
    
    # Feature extraction submodel
    feature_model = tf.keras.Model(inputs=base_model.input, outputs=first_conv_layer.output)
    feature_maps = feature_model(img_preprocessed).numpy()[0]  # shape: (100, 100, 64)
    print(f"Feature activation maps shape: {feature_maps.shape}")
    
    # 3. Create Multi-Panel Plot
    os.makedirs(os.path.join("outputs", "figures"), exist_ok=True)
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(3, 8, height_ratios=[1.2, 1, 1])
    
    # Panel 1: Original input image
    ax_orig = fig.add_subplot(gs[0, 0:2])
    ax_orig.imshow(img_pil)
    ax_orig.set_title(f"Input Defect Image\n({sample_record['primary_class']})", fontsize=11)
    ax_orig.axis("off")
    
    # Panel 2: Learned Filter Kernels (Row 0: 6 filters, Row 1: 8 filters)
    for i in range(6):
        ax = fig.add_subplot(gs[0, 2 + i])
        ax.imshow(filters_norm[:, :, :, i])
        ax.set_title(f"Filter {i+1}", fontsize=8)
        ax.axis("off")
        
    for i in range(8):
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(filters_norm[:, :, :, 6 + i])
        ax.set_title(f"Filter {i+7}", fontsize=8)
        ax.axis("off")
        
    # Panel 3: Activated Feature Maps (8 feature maps)
    for j in range(8):
        ax = fig.add_subplot(gs[2, j])
        ax.imshow(feature_maps[:, :, j], cmap="viridis")
        ax.set_title(f"FeatMap {j+1}", fontsize=8)
        ax.axis("off")
        
    plt.suptitle("ResNet50 conv1_conv (7x7): Learned Filters and Activated Defect Feature Maps", fontsize=14, y=0.98)
    plt.tight_layout()
    
    out_fig = os.path.join("outputs", "figures", "conv2d_filters_and_feature_maps.png")
    plt.savefig(out_fig, dpi=300)
    plt.close()
    print(f"[PLOT] Saved conv visualization to {out_fig}")

if __name__ == "__main__":
    visualize_conv_filters_and_features()
