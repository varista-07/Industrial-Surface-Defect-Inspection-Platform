"""
advanced/defect_transformer.py
Transformer-Based Industrial Inspection Report Encoder (TensorFlow / Keras):
1. Automatic rule-based synthetic inspection report generation from VOC annotations:
   - Derives morphological descriptors (aspect ratio, surface area, localized coordinates).
   - Generates natural-language reports (e.g. "Critical elongated scratch near top edge").
   - Explicitly documented: Reports are synthetically generated from spatial bounding boxes.
2. Custom Transformer Encoder:
   - Token & Learned Positional Embedding.
   - Multi-Head Self-Attention (4 heads, key_dim=32).
   - Pre-LN / Post-LN Residual Feed-Forward block.
   - Predicts defect class or severity from text.
3. Attention weight visualization across token sequences.
4. Generates outputs/figures/transformer_attention_weights.png and outputs/metrics/transformer_test_metrics.json.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
import datetime
import numpy as np
import matplotlib.pyplot as plt

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PYTHONHASHSEED"] = "42"

import tensorflow as tf
tf.keras.utils.set_random_seed(42)

def generate_synthetic_report(class_name, boxes):
    """
    Synthesize natural language defect inspection report from class and bounding boxes.
    DISCLAIMER: These inspection reports are synthetically generated from VOC metadata.
    """
    if len(boxes) == 0:
        return "Clean steel surface inspected with no visible surface defects detected."
        
    # Dominant box
    sorted_boxes = sorted(boxes, key=lambda b: (b["norm_xmax"] - b["norm_xmin"]) * (b["norm_ymax"] - b["norm_ymin"]), reverse=True)
    dom = sorted_boxes[0]
    
    w = dom["norm_xmax"] - dom["norm_xmin"]
    h = dom["norm_ymax"] - dom["norm_ymin"]
    area = w * h
    aspect = w / (h + 1e-6)
    
    # Severity based on cumulative defect area
    if area > 0.35:
        severity = "Severe"
    elif area > 0.15:
        severity = "Moderate"
    else:
        severity = "Minor"
        
    # Morphology
    if aspect > 2.5 or aspect < 0.4:
        morphology = "elongated"
    elif area < 0.08:
        morphology = "isolated pinpoint"
    else:
        morphology = "irregular clustered"
        
    # Position
    cy = (dom["norm_ymin"] + dom["norm_ymax"]) / 2.0
    cx = (dom["norm_xmin"] + dom["norm_xmax"]) / 2.0
    
    v_pos = "top" if cy < 0.35 else ("bottom" if cy > 0.65 else "central")
    h_pos = "left" if cx < 0.35 else ("right" if cx > 0.65 else "region")
    position_str = f"near {v_pos} {h_pos}"
    
    # Clean class text
    c_clean = class_name.replace("_", " ")
    
    report = f"{severity} {morphology} {c_clean} defect identified {position_str} spanning approximately {int(area*100)} percent surface area."
    return report

class TransformerEncoderBlock(tf.keras.layers.Layer):
    """Multi-Head Self-Attention + Position-Wise Feed-Forward Layer."""
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"),
            tf.keras.layers.Dense(embed_dim),
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training=False, return_attention_scores=False):
        if return_attention_scores:
            attn_output, attn_scores = self.att(inputs, inputs, return_attention_scores=True)
        else:
            attn_output = self.att(inputs, inputs)
            attn_scores = None
            
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        if return_attention_scores:
            return out2, attn_scores
        return out2

class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    """Token Embedding combined with learned positional encodings."""
    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_emb = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = tf.keras.layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

def build_transformer_model(vocab_size, maxlen=25, embed_dim=64, num_heads=4, ff_dim=128, num_classes=6):
    inputs = tf.keras.Input(shape=(maxlen,), dtype="int32")
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    
    transformer_block = TransformerEncoderBlock(embed_dim, num_heads, ff_dim, rate=0.1, name="transformer_encoder")
    x = transformer_block(x)
    
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="DefectReportTransformer")
    return model, transformer_block, embedding_layer

def train_and_visualize_transformer(epochs=12, batch_size=32):
    print("=" * 65)
    print("SYNTHETIC REPORT GENERATION & TRANSFORMER ENCODER")
    print("=" * 65)
    
    with open(os.path.join("data", "splits", "split_manifest.json"), "r") as f:
        manifest = json.load(f)
        
    classes = manifest["classes"]
    records = manifest["records"]
    
    # 1. Generate Synthetic Text Reports for all records
    texts = []
    labels = []
    splits = []
    
    print(f"[GEN] Generating synthetic inspection reports for {len(records)} records...")
    for r in records:
        rep = generate_synthetic_report(r["primary_class"], r["boxes"])
        texts.append(rep)
        labels.append(r["class_id"])
        splits.append(r["split"])
        
    print(f"Sample Report (crazing):    '{texts[0]}'")
    print(f"Sample Report (scratches):  '{next(t for t, l in zip(texts, labels) if l==5)}'")
    
    # 2. Text Vectorization
    maxlen = 25
    vocab_size = 200
    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=vocab_size,
        output_sequence_length=maxlen,
        standardize="lower_and_strip_punctuation",
        split="whitespace"
    )
    vectorizer.adapt(texts)
    vocab = vectorizer.get_vocabulary()
    print(f"[VOCAB] Built inspection vocabulary of size: {len(vocab)}")
    
    # Tokenize
    X_all = vectorizer(np.array(texts)).numpy()
    y_all = tf.keras.utils.to_categorical(labels, num_classes=len(classes))
    
    # Partition by manifest split
    train_mask = np.array([s == "train" for s in splits])
    val_mask = np.array([s == "val" for s in splits])
    test_mask = np.array([s == "test" for s in splits])
    
    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_val, y_val = X_all[val_mask], y_all[val_mask]
    X_test, y_test = X_all[test_mask], y_all[test_mask]
    
    # 3. Build & Train Model
    model, transformer_layer, emb_layer = build_transformer_model(
        vocab_size=len(vocab), maxlen=maxlen, embed_dim=64, num_heads=4, ff_dim=128, num_classes=len(classes)
    )
    
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n[TRANSFORMER TEST] Accuracy on report classification: {test_acc*100:.2f}% (Loss: {test_loss:.4f})")
    
    # 4. Extract Attention Weights & Visualize
    sample_text = texts[0]
    sample_tokens = [vocab[idx] if idx < len(vocab) else "" for idx in X_all[0]]
    # Filter empty padding tokens
    valid_len = sum(1 for t in sample_tokens if t != "")
    display_tokens = sample_tokens[:valid_len]
    
    # Extract attention scores directly in eager execution
    sample_input = np.expand_dims(X_all[0], axis=0)
    emb_out = emb_layer(sample_input)
    _, attn_scores = transformer_layer(emb_out, return_attention_scores=True)
    scores = attn_scores.numpy()[0]  # shape: (num_heads, maxlen, maxlen)
    
    # Plot 4-head attention maps
    os.makedirs(os.path.join("outputs", "figures"), exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    
    for head_idx in range(4):
        ax = axes[head_idx]
        head_scores = scores[head_idx, :valid_len, :valid_len]
        im = ax.imshow(head_scores, cmap="Blues", interpolation="nearest")
        ax.set_xticks(range(valid_len))
        ax.set_yticks(range(valid_len))
        ax.set_xticklabels(display_tokens, rotation=90, fontsize=8)
        ax.set_yticklabels(display_tokens, fontsize=8)
        ax.set_title(f"Attention Head {head_idx+1}")
        
    plt.suptitle(f"Transformer Self-Attention Heatmaps on Synthetic Report\n'{sample_text}'", fontsize=11, y=1.02)
    plt.tight_layout()
    fig_path = os.path.join("outputs", "figures", "transformer_attention_weights.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved transformer attention visualization to {fig_path}")
    
    # Save Metrics JSON
    os.makedirs(os.path.join("outputs", "metrics"), exist_ok=True)
    metrics_export = {
        "module_name": "Synthetic Inspection Report Transformer Encoder",
        "timestamp": datetime.datetime.now().isoformat(),
        "disclaimer": "Reports are synthetically constructed from Pascal VOC bounding box metadata for research demonstration.",
        "test_metrics": {
            "test_accuracy": float(test_acc),
            "test_loss": float(test_loss),
            "vocab_size": len(vocab),
            "max_sequence_length": maxlen,
            "num_attention_heads": 4
        }
    }
    metrics_path = os.path.join("outputs", "metrics", "transformer_test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_export, f, indent=2)
    print(f"[SAVE] Exported transformer metrics to {metrics_path}")
    return model

if __name__ == "__main__":
    train_and_visualize_transformer(epochs=8, batch_size=32)
