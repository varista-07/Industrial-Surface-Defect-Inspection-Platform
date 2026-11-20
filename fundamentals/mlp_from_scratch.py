"""
fundamentals/mlp_from_scratch.py
Modular Multilayer Perceptron (MLP) from Scratch using NumPy:
1. Modular layers: LinearLayer, ReLU, Dropout, BatchNormalization1D, SoftmaxCrossEntropy.
2. Exact analytical backpropagation and gradient caching.
3. Parameter vs Hyperparameter comparison table.
4. Downsampled 32x32 NEU-DET image loader (1024-dim input -> 6 classes).
5. BASELINE model training, evaluation on test split, and metric export.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

np.random.seed(42)

class LinearLayer:
    """Fully Connected Linear Layer: Z = X @ W + b"""
    def __init__(self, in_features, out_features, init_type="he"):
        self.in_features = in_features
        self.out_features = out_features
        self.init_type = init_type
        
        # Initializations
        if init_type == "zeros":
            self.W = np.zeros((in_features, out_features), dtype=np.float32)
        elif init_type == "random":
            self.W = np.random.randn(in_features, out_features).astype(np.float32) * 0.05
        elif init_type == "xavier":
            # Glorot / Xavier: Var = 2 / (fan_in + fan_out)
            limit = np.sqrt(6.0 / (in_features + out_features))
            self.W = np.random.uniform(-limit, limit, (in_features, out_features)).astype(np.float32)
        elif init_type == "he":
            # He / Kaiming: Std = sqrt(2 / fan_in)
            std = np.sqrt(2.0 / in_features)
            self.W = (np.random.randn(in_features, out_features) * std).astype(np.float32)
        else:
            raise ValueError(f"Unknown init_type: {init_type}")
            
        self.b = np.zeros(out_features, dtype=np.float32)
        
        # Gradients
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.X_cache = None

    def forward(self, X):
        self.X_cache = X
        return np.dot(X, self.W) + self.b

    def backward(self, dZ, l2_reg=0.0):
        # dZ shape: (Batch, out_features)
        self.dW = np.dot(self.X_cache.T, dZ) + l2_reg * self.W
        self.db = np.sum(dZ, axis=0)
        dX = np.dot(dZ, self.W.T)
        return dX

class ReLU:
    """Element-wise Rectified Linear Unit activation."""
    def __init__(self):
        self.Z_cache = None
        
    def forward(self, Z):
        self.Z_cache = Z
        return np.maximum(0.0, Z)
        
    def backward(self, dA):
        dZ = dA * np.where(self.Z_cache > 0, 1.0, 0.0)
        return dZ

class Dropout:
    """Inverted Dropout layer for regularization."""
    def __init__(self, drop_rate=0.2):
        self.drop_rate = drop_rate
        self.mask = None
        
    def forward(self, X, training=True):
        if not training or self.drop_rate <= 0.0:
            return X
        keep_prob = 1.0 - self.drop_rate
        self.mask = (np.random.rand(*X.shape) < keep_prob) / keep_prob
        return X * self.mask
        
    def backward(self, dOut):
        if self.mask is None:
            return dOut
        return dOut * self.mask

class BatchNormalization1D:
    """Mini-batch Normalization for fully connected layers."""
    def __init__(self, dim, momentum=0.9, eps=1e-5):
        self.gamma = np.ones(dim, dtype=np.float32)
        self.beta = np.zeros(dim, dtype=np.float32)
        self.dgamma = np.zeros_like(self.gamma)
        self.dbeta = np.zeros_like(self.beta)
        self.momentum = momentum
        self.eps = eps
        
        self.running_mean = np.zeros(dim, dtype=np.float32)
        self.running_var = np.ones(dim, dtype=np.float32)
        self.cache = None

    def forward(self, X, training=True):
        if training:
            mean = np.mean(X, axis=0)
            var = np.var(X, axis=0)
            x_norm = (X - mean) / np.sqrt(var + self.eps)
            self.running_mean = self.momentum * self.running_mean + (1.0 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1.0 - self.momentum) * var
            self.cache = (X, x_norm, mean, var)
        else:
            x_norm = (X - self.running_mean) / np.sqrt(self.running_var + self.eps)
        return self.gamma * x_norm + self.beta

    def backward(self, dOut):
        X, x_norm, mean, var = self.cache
        N = X.shape[0]
        std_inv = 1.0 / np.sqrt(var + self.eps)
        
        self.dgamma = np.sum(dOut * x_norm, axis=0)
        self.dbeta = np.sum(dOut, axis=0)
        
        dx_norm = dOut * self.gamma
        dvar = np.sum(dx_norm * (X - mean) * -0.5 * (std_inv ** 3), axis=0)
        dmean = np.sum(dx_norm * -std_inv, axis=0) + dvar * np.mean(-2.0 * (X - mean), axis=0)
        dX = dx_norm * std_inv + dvar * 2.0 * (X - mean) / N + dmean / N
        return dX

class SoftmaxCrossEntropy:
    """Softmax Activation combined with Cross-Entropy Loss."""
    @staticmethod
    def forward(logits, y_indices):
        # Stable softmax
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exp_vals = np.exp(shifted)
        probs = exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)
        
        N = len(y_indices)
        eps = 1e-12
        correct_logprobs = -np.log(probs[np.arange(N), y_indices] + eps)
        loss = np.mean(correct_logprobs)
        return probs, loss
        
    @staticmethod
    def backward(probs, y_indices):
        # Combined gradient of batch mean cross-entropy: dL/dZ = (probs - 1_y) / N
        N = len(y_indices)
        dZ = probs.copy()
        dZ[np.arange(N), y_indices] -= 1.0
        return dZ / float(N)

class MLPScratch:
    """
    Complete Multi-Layer Perceptron architecture from scratch.
    """
    def __init__(self, layer_dims=[1024, 256, 64, 6], init_type="he", drop_rate=0.2, use_bn=False, l2_reg=1e-4):
        self.layer_dims = layer_dims
        self.l2_reg = l2_reg
        self.use_bn = use_bn
        
        self.layers = []
        for i in range(len(layer_dims) - 1):
            din, dout = layer_dims[i], layer_dims[i+1]
            self.layers.append(LinearLayer(din, dout, init_type=init_type))
            if i < len(layer_dims) - 2:
                if use_bn:
                    self.layers.append(BatchNormalization1D(dout))
                self.layers.append(ReLU())
                if drop_rate > 0.0:
                    self.layers.append(Dropout(drop_rate))
                    
    def forward(self, X, training=True):
        out = X
        for layer in self.layers:
            if isinstance(layer, (Dropout, BatchNormalization1D)):
                out = layer.forward(out, training=training)
            else:
                out = layer.forward(out)
        return out
        
    def backward(self, dZ):
        grad = dZ
        for layer in reversed(self.layers):
            if isinstance(layer, LinearLayer):
                grad = layer.backward(grad, l2_reg=self.l2_reg)
            else:
                grad = layer.backward(grad)
        return grad

    def get_parameter_table(self):
        """Generate detailed parameter vs hyperparameter table."""
        param_details = []
        total_trainable = 0
        for idx, layer in enumerate(self.layers):
            if isinstance(layer, LinearLayer):
                w_params = layer.W.size
                b_params = layer.b.size
                layer_total = w_params + b_params
                total_trainable += layer_total
                param_details.append({
                    "layer": f"Linear_{idx+1}",
                    "shape": f"W:{layer.W.shape}, b:{layer.b.shape}",
                    "trainable_params": layer_total
                })
            elif isinstance(layer, BatchNormalization1D):
                bn_params = layer.gamma.size + layer.beta.size
                total_trainable += bn_params
                param_details.append({
                    "layer": f"BatchNorm_{idx+1}",
                    "shape": f"gamma:{layer.gamma.shape}, beta:{layer.beta.shape}",
                    "trainable_params": bn_params
                })
        return param_details, total_trainable

def load_downsampled_dataset(target_size=(32, 32)):
    """
    Load NEU-DET images, downsample to 32x32 grayscale, and flatten to 1024-dim vectors.
    Caches processed dataset to data/splits/neu_det_32x32.npz for fast loading.
    """
    cache_path = os.path.join("data", "splits", f"neu_det_{target_size[0]}x{target_size[1]}.npz")
    split_manifest = os.path.join("data", "splits", "split_manifest.json")
    if not os.path.exists(split_manifest):
        raise FileNotFoundError("Split manifest missing! Run prepare_data.py first.")
        
    with open(split_manifest, "r") as f:
        data = json.load(f)
    class_names = data["classes"]
    
    if os.path.exists(cache_path):
        cached = np.load(cache_path)
        X_train, y_train = cached["X_train"], cached["y_train"]
        X_val, y_val = cached["X_val"], cached["y_val"]
        X_test, y_test = cached["X_test"], cached["y_test"]
        return X_train, y_train, X_val, y_val, X_test, y_test, class_names

    splits = {"train": ([], []), "val": ([], []), "test": ([], [])}
    for record in data["records"]:
        split_name = record["split"]
        img_path = record["image_path"]
        class_id = record["class_id"]
        
        with Image.open(img_path).convert("L") as im:
            im_resized = im.resize(target_size, Image.Resampling.BILINEAR)
            arr = np.array(im_resized, dtype=np.float32) / 255.0  # Normalize to [0, 1]
            flat = arr.flatten()
            
        splits[split_name][0].append(flat)
        splits[split_name][1].append(class_id)
        
    X_train = np.array(splits["train"][0], dtype=np.float32)
    y_train = np.array(splits["train"][1], dtype=np.int64)
    X_val = np.array(splits["val"][0], dtype=np.float32)
    y_val = np.array(splits["val"][1], dtype=np.int64)
    X_test = np.array(splits["test"][0], dtype=np.float32)
    y_test = np.array(splits["test"][1], dtype=np.int64)
    
    # Save cache
    np.savez_compressed(cache_path, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test)
    print(f"[CACHE] Saved preprocessed dataset to {cache_path}")
    
    print(f"[DATA] Downsampled 32x32 dataset loaded:")
    print(f"  Train: X={X_train.shape}, y={y_train.shape}")
    print(f"  Val:   X={X_val.shape}, y={y_val.shape}")
    print(f"  Test:  X={X_test.shape}, y={y_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test, class_names

def train_baseline_mlp(epochs=60, batch_size=32, lr=0.02):
    print("=" * 60)
    print("TRAINING FROM-SCRATCH MLP BASELINE ON 32x32 NEU-DET")
    print("=" * 60)
    
    X_train, y_train, X_val, y_val, X_test, y_test, class_names = load_downsampled_dataset((32, 32))
    
    model = MLPScratch(layer_dims=[1024, 256, 64, 6], init_type="he", drop_rate=0.2, use_bn=False, l2_reg=1e-4)
    params_table, total_params = model.get_parameter_table()
    
    print("\n--- Model Parameter Structure ---")
    for row in params_table:
        print(f"  {row['layer']:15s} | {row['shape']:28s} | Params: {row['trainable_params']}")
    print(f"Total Trainable Parameters: {total_params:,}")
    
    # Training Loop with Mini-batch SGD + Momentum (0.9)
    N = len(X_train)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    
    # Velocity caches for momentum
    v_W = [np.zeros_like(layer.W) for layer in model.layers if isinstance(layer, LinearLayer)]
    v_b = [np.zeros_like(layer.b) for layer in model.layers if isinstance(layer, LinearLayer)]
    momentum_beta = 0.9
    
    for epoch in range(epochs):
        # Shuffle training batch
        indices = np.arange(N)
        np.random.shuffle(indices)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]
        
        batch_losses = []
        for start_idx in range(0, N, batch_size):
            end_idx = min(start_idx + batch_size, N)
            xb = X_train_shuffled[start_idx:end_idx]
            yb = y_train_shuffled[start_idx:end_idx]
            
            # Forward
            logits = model.forward(xb, training=True)
            probs, loss = SoftmaxCrossEntropy.forward(logits, yb)
            batch_losses.append(loss)
            
            # Backward
            dZ = SoftmaxCrossEntropy.backward(probs, yb)
            model.backward(dZ)
            
            # SGD with Momentum update
            lin_idx = 0
            for layer in model.layers:
                if isinstance(layer, LinearLayer):
                    v_W[lin_idx] = momentum_beta * v_W[lin_idx] + lr * layer.dW
                    v_b[lin_idx] = momentum_beta * v_b[lin_idx] + lr * layer.db
                    layer.W -= v_W[lin_idx]
                    layer.b -= v_b[lin_idx]
                    lin_idx += 1
                    
        train_loss = float(np.mean(batch_losses))
        
        # Validation evaluation
        val_logits = model.forward(X_val, training=False)
        val_probs, val_loss = SoftmaxCrossEntropy.forward(val_logits, y_val)
        val_acc = float(np.mean(np.argmax(val_probs, axis=1) == y_val))
        
        # Train accuracy
        train_eval_logits = model.forward(X_train, training=False)
        train_eval_probs, _ = SoftmaxCrossEntropy.forward(train_eval_logits, y_train)
        train_acc = float(np.mean(np.argmax(train_eval_probs, axis=1) == y_train))
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(float(val_loss))
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:2d}/{epochs:2d} -> Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} | Train Acc: {train_acc*100:5.1f}%, Val Acc: {val_acc*100:5.1f}%")
            
    # Final Test Set Evaluation
    test_logits = model.forward(X_test, training=False)
    test_probs, test_loss = SoftmaxCrossEntropy.forward(test_logits, y_test)
    test_preds = np.argmax(test_probs, axis=1)
    test_acc = float(np.mean(test_preds == y_test))
    
    print("\n" + "=" * 60)
    print(f"FINAL BASELINE MLP TEST ACCURACY: {test_acc*100:.2f}% (Loss: {test_loss:.4f})")
    print("=" * 60)
    
    # Save training curves plot
    os.makedirs(os.path.join("outputs", "figures"), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    
    axes[0].plot(history["train_loss"], label="Train Loss", color="#1f77b4", lw=2)
    axes[0].plot(history["val_loss"], label="Val Loss", color="#ff7f0e", lw=2)
    axes[0].set_title("MLP Baseline Cross-Entropy Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend()
    
    axes[1].plot(np.array(history["train_acc"]) * 100, label="Train Acc (%)", color="#2ca02c", lw=2)
    axes[1].plot(np.array(history["val_acc"]) * 100, label="Val Acc (%)", color="#d62728", lw=2)
    axes[1].set_title(f"MLP Baseline Accuracy (Final Test: {test_acc*100:.1f}%)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend()
    
    plt.tight_layout()
    curve_path = os.path.join("outputs", "figures", "baseline_mlp_training_curves.png")
    plt.savefig(curve_path, dpi=300)
    plt.close()
    print(f"[PLOT] Saved baseline MLP curves to {curve_path}")
    
    # Save baseline metrics JSON
    metrics_path = os.path.join("outputs", "metrics", "baseline_mlp_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "model_name": "From-Scratch MLP (NumPy)",
            "architecture": "1024 -> 256 -> 64 -> 6",
            "total_parameters": total_params,
            "hyperparameters": {
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": lr,
                "optimizer": "SGD_Momentum_0.9",
                "dropout": 0.2,
                "l2_reg": 1e-4,
                "image_resolution": "32x32"
            },
            "metrics": {
                "final_train_loss": history["train_loss"][-1],
                "final_val_loss": history["val_loss"][-1],
                "final_val_acc": history["val_acc"][-1],
                "test_loss": float(test_loss),
                "test_accuracy": test_acc
            }
        }, f, indent=2)
    print(f"[SAVE] Exported baseline metrics to {metrics_path}")

if __name__ == "__main__":
    train_baseline_mlp(epochs=60, batch_size=32, lr=0.02)
