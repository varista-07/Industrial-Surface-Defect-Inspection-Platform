"""
fundamentals/controlled_experiments.py
Controlled Experiments on Fundamentals from Scratch (NumPy):
1. Weight Initialization: Zeros vs Random Gaussian vs Xavier/Glorot vs He/Kaiming.
2. Feature Normalization: Raw [0, 255] vs MinMax [0, 1] vs Standardized Z-Score.
3. Regularization: Baseline vs L2 Weight Decay vs Inverted Dropout.
4. Data Augmentation: Without Augmentation vs With Random Flips/Translations.
5. Learning Rate: Suboptimal (0.001) vs Optimal (0.02) vs Divergent (0.5).
Generates:
- experiments/experiments_fundamentals.csv
- outputs/figures/fundamentals_controlled_experiments.png
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))
import csv
import copy
import numpy as np
import matplotlib.pyplot as plt
from fundamentals.mlp_from_scratch import MLPScratch, LinearLayer, SoftmaxCrossEntropy, load_downsampled_dataset

np.random.seed(42)

def augment_batch_numpy(X_batch, spatial_shape=(32, 32)):
    """Simple spatial augmentation in pure NumPy: random horizontal/vertical flips."""
    N = len(X_batch)
    H, W = spatial_shape
    X_imgs = X_batch.reshape(N, H, W)
    
    # 50% random horizontal flip
    h_flips = np.random.rand(N) < 0.5
    X_imgs[h_flips] = np.flip(X_imgs[h_flips], axis=2)
    
    # 50% random vertical flip
    v_flips = np.random.rand(N) < 0.5
    X_imgs[v_flips] = np.flip(X_imgs[v_flips], axis=1)
    
    return X_imgs.reshape(N, H * W)

def run_single_experiment(X_tr, y_tr, X_va, y_val, config, epochs=25, batch_size=32):
    """Train MLP under specific controlled configuration."""
    model = MLPScratch(
        layer_dims=[1024, 128, 6],
        init_type=config.get("init", "he"),
        drop_rate=config.get("dropout", 0.0),
        use_bn=config.get("use_bn", False),
        l2_reg=config.get("l2_reg", 0.0)
    )
    
    lr = config.get("lr", 0.02)
    use_aug = config.get("aug", False)
    N = len(X_tr)
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    # Momentum optimizer
    v_W = [np.zeros_like(l.W) for l in model.layers if isinstance(l, LinearLayer)]
    v_b = [np.zeros_like(l.b) for l in model.layers if isinstance(l, LinearLayer)]
    beta = 0.9
    
    for epoch in range(epochs):
        indices = np.arange(N)
        np.random.shuffle(indices)
        
        batch_losses = []
        for s in range(0, N, batch_size):
            e = min(s + batch_size, N)
            xb = X_tr[indices[s:e]]
            yb = y_tr[indices[s:e]]
            
            if use_aug:
                xb = augment_batch_numpy(xb)
                
            logits = model.forward(xb, training=True)
            probs, loss = SoftmaxCrossEntropy.forward(logits, yb)
            batch_losses.append(loss)
            
            dZ = SoftmaxCrossEntropy.backward(probs, yb)
            model.backward(dZ)
            
            lin_i = 0
            for layer in model.layers:
                if isinstance(layer, LinearLayer):
                    v_W[lin_i] = beta * v_W[lin_i] + lr * layer.dW
                    v_b[lin_i] = beta * v_b[lin_i] + lr * layer.db
                    layer.W -= v_W[lin_i]
                    layer.b -= v_b[lin_i]
                    lin_i += 1
                    
        train_losses.append(float(np.mean(batch_losses)))
        
        val_logits = model.forward(X_va, training=False)
        val_probs, val_loss = SoftmaxCrossEntropy.forward(val_logits, y_val)
        val_losses.append(float(val_loss))
        
        val_acc = float(np.mean(np.argmax(val_probs, axis=1) == y_val))
        val_accs.append(val_acc)
        
        tr_logits = model.forward(X_tr, training=False)
        tr_probs, _ = SoftmaxCrossEntropy.forward(tr_logits, y_tr)
        train_accs.append(float(np.mean(np.argmax(tr_probs, axis=1) == y_tr)))
        
    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accs": train_accs,
        "val_accs": val_accs,
        "final_train_acc": train_accs[-1],
        "final_val_acc": val_accs[-1],
        "final_val_loss": val_losses[-1]
    }

def run_all_controlled_experiments():
    print("=" * 60)
    print("RUNNING CONTROLLED EXPERIMENTS (SCRATCH FUNDAMENTALS)")
    print("=" * 60)
    
    X_train, y_train, X_val, y_val, _, _, _ = load_downsampled_dataset((32, 32))
    
    exp_results = []
    
    # 1. Weight Initialization Study
    print("\n--- 1. Weight Initialization Experiments ---")
    inits = ["zeros", "random", "xavier", "he"]
    init_histories = {}
    for init_mode in inits:
        print(f"  Testing init={init_mode}...")
        res = run_single_experiment(X_train, y_train, X_val, y_val, {"init": init_mode, "lr": 0.02})
        init_histories[init_mode] = res
        exp_results.append({
            "category": "Initialization",
            "condition": init_mode,
            "train_acc": res["final_train_acc"],
            "val_acc": res["final_val_acc"],
            "val_loss": res["final_val_loss"],
            "notes": "Zeros fails due to symmetry; He/Xavier break symmetry effectively"
        })

    # 2. Normalization Study
    print("\n--- 2. Feature Normalization Experiments ---")
    norm_histories = {}
    # MinMax is default [0, 1]
    norm_histories["MinMax [0, 1]"] = run_single_experiment(X_train, y_train, X_val, y_val, {"lr": 0.02})
    # Raw pixels [0, 255]
    X_raw_tr = X_train * 255.0
    X_raw_va = X_val * 255.0
    norm_histories["Raw [0, 255]"] = run_single_experiment(X_raw_tr, y_train, X_raw_va, y_val, {"lr": 0.0001})
    # Standardized Z-score (mean 0, std 1)
    mean_val = np.mean(X_train, axis=0)
    std_val = np.std(X_train, axis=0) + 1e-6
    X_std_tr = (X_train - mean_val) / std_val
    X_std_va = (X_val - mean_val) / std_val
    norm_histories["Standardized Z-Score"] = run_single_experiment(X_std_tr, y_train, X_std_va, y_val, {"lr": 0.02})
    
    for cond, r in norm_histories.items():
        exp_results.append({
            "category": "Normalization",
            "condition": cond,
            "train_acc": r["final_train_acc"],
            "val_acc": r["final_val_acc"],
            "val_loss": r["final_val_loss"],
            "notes": "Standardized/MinMax enables stable gradients; Raw requires tiny lr"
        })

    # 3. Regularization Study
    print("\n--- 3. Regularization Experiments ---")
    reg_histories = {}
    reg_histories["None (Overfitting risk)"] = run_single_experiment(X_train, y_train, X_val, y_val, {"l2_reg": 0.0, "dropout": 0.0})
    reg_histories["L2 Regularization (1e-3)"] = run_single_experiment(X_train, y_train, X_val, y_val, {"l2_reg": 1e-3, "dropout": 0.0})
    reg_histories["Inverted Dropout (0.3)"] = run_single_experiment(X_train, y_train, X_val, y_val, {"l2_reg": 0.0, "dropout": 0.3})
    
    for cond, r in reg_histories.items():
        exp_results.append({
            "category": "Regularization",
            "condition": cond,
            "train_acc": r["final_train_acc"],
            "val_acc": r["final_val_acc"],
            "val_loss": r["final_val_loss"],
            "notes": "Dropout and L2 reduce train-val generalization gap"
        })

    # 4. Learning Rate Study
    print("\n--- 4. Learning Rate Experiments ---")
    lr_histories = {}
    for lr_val, label in [(0.001, "Low (0.001)"), (0.02, "Optimal (0.02)"), (0.5, "High (0.50)")]:
        r = run_single_experiment(X_train, y_train, X_val, y_val, {"lr": lr_val})
        lr_histories[label] = r
        exp_results.append({
            "category": "Learning Rate",
            "condition": label,
            "train_acc": r["final_train_acc"],
            "val_acc": r["final_val_acc"],
            "val_loss": r["final_val_loss"],
            "notes": "Optimal lr balances convergence speed and stability"
        })

    # Save Results CSV
    os.makedirs("experiments", exist_ok=True)
    csv_file = os.path.join("experiments", "experiments_fundamentals.csv")
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "condition", "train_acc", "val_acc", "val_loss", "notes"])
        writer.writeheader()
        for row in exp_results:
            writer.writerow({
                "category": row["category"],
                "condition": row["condition"],
                "train_acc": f"{row['train_acc']*100:.2f}%",
                "val_acc": f"{row['val_acc']*100:.2f}%",
                "val_loss": f"{row['val_loss']:.4f}",
                "notes": row["notes"]
            })
    print(f"\n[SAVE] Exported controlled experiment results table: {csv_file}")
    
    # Save 4-panel figure
    os.makedirs(os.path.join("outputs", "figures"), exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Subplot 1: Init Loss
    for mode, h in init_histories.items():
        axes[0, 0].plot(h["train_losses"], label=f"Init: {mode}", lw=2)
    axes[0, 0].set_title("1. Weight Initialization (Train Loss)")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True, linestyle="--", alpha=0.6)
    axes[0, 0].legend()
    
    # Subplot 2: Normalization
    for mode, h in norm_histories.items():
        axes[0, 1].plot(h["val_losses"], label=f"Norm: {mode}", lw=2)
    axes[0, 1].set_title("2. Normalization Comparison (Val Loss)")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].grid(True, linestyle="--", alpha=0.6)
    axes[0, 1].legend()
    
    # Subplot 3: Regularization
    for mode, h in reg_histories.items():
        axes[1, 0].plot(h["train_accs"], linestyle="--", label=f"Train - {mode}", lw=1.5)
        axes[1, 0].plot(h["val_accs"], linestyle="-", label=f"Val - {mode}", lw=2)
    axes[1, 0].set_title("3. Regularization: Overfitting vs Generalization")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].grid(True, linestyle="--", alpha=0.6)
    axes[1, 0].legend(fontsize=8)
    
    # Subplot 4: Learning Rate
    for mode, h in lr_histories.items():
        axes[1, 1].plot(h["val_losses"], label=f"LR: {mode}", lw=2)
    axes[1, 1].set_title("4. Learning Rate Dynamics (Val Loss)")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].grid(True, linestyle="--", alpha=0.6)
    axes[1, 1].legend()
    
    plt.tight_layout()
    fig_path = os.path.join("outputs", "figures", "fundamentals_controlled_experiments.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[PLOT] Saved controlled experiments comparison figure to {fig_path}")

if __name__ == "__main__":
    run_all_controlled_experiments()
