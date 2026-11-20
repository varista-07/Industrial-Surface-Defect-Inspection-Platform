"""
fundamentals/optimizers_from_scratch.py
Optimizers from Scratch in NumPy:
1. Batch Gradient Descent (BGD)
2. Stochastic Gradient Descent (SGD)
3. SGD with Classical Momentum
4. Nesterov Accelerated Gradient (NAG)
Compares all 4 optimizers on identical MLP weights & 32x32 NEU-DET data.
Generates outputs/figures/optimizers_comparison_loss_curves.png.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))
import copy
import numpy as np
import matplotlib.pyplot as plt
from fundamentals.mlp_from_scratch import MLPScratch, LinearLayer, SoftmaxCrossEntropy, load_downsampled_dataset

np.random.seed(42)

class OptimizerBGD:
    """Batch Gradient Descent: full batch update per step."""
    def __init__(self, layers, lr=0.05):
        self.layers = [l for l in layers if isinstance(l, LinearLayer)]
        self.lr = lr
        
    def step(self):
        for layer in self.layers:
            layer.W -= self.lr * layer.dW
            layer.b -= self.lr * layer.db

class OptimizerSGD:
    """Mini-batch / Stochastic Gradient Descent without momentum."""
    def __init__(self, layers, lr=0.02):
        self.layers = [l for l in layers if isinstance(l, LinearLayer)]
        self.lr = lr
        
    def step(self):
        for layer in self.layers:
            layer.W -= self.lr * layer.dW
            layer.b -= self.lr * layer.db

class OptimizerMomentum:
    """Classical Polyak Momentum (1964): v = beta * v + lr * grad, W -= v."""
    def __init__(self, layers, lr=0.02, beta=0.9):
        self.layers = [l for l in layers if isinstance(l, LinearLayer)]
        self.lr = lr
        self.beta = beta
        self.v_W = [np.zeros_like(l.W) for l in self.layers]
        self.v_b = [np.zeros_like(l.b) for l in self.layers]
        
    def step(self):
        for idx, layer in enumerate(self.layers):
            self.v_W[idx] = self.beta * self.v_W[idx] + self.lr * layer.dW
            self.v_b[idx] = self.beta * self.v_b[idx] + self.lr * layer.db
            layer.W -= self.v_W[idx]
            layer.b -= self.v_b[idx]

class OptimizerNesterov:
    """
    Nesterov Accelerated Gradient (NAG, 1983):
    Lookahead step: theta_lookahead = theta - beta * v
    Gradient computed at lookahead point, then update velocity and position.
    """
    def __init__(self, layers, lr=0.02, beta=0.9):
        self.layers = [l for l in layers if isinstance(l, LinearLayer)]
        self.lr = lr
        self.beta = beta
        self.v_W = [np.zeros_like(l.W) for l in self.layers]
        self.v_b = [np.zeros_like(l.b) for l in self.layers]
        
    def apply_lookahead(self):
        # Shift parameters to lookahead position before backward pass
        for idx, layer in enumerate(self.layers):
            layer.W -= self.beta * self.v_W[idx]
            layer.b -= self.beta * self.v_b[idx]
            
    def step(self):
        for idx, layer in enumerate(self.layers):
            # Undo lookahead shift first
            layer.W += self.beta * self.v_W[idx]
            layer.b += self.beta * self.v_b[idx]
            # Update velocity using gradient at lookahead
            self.v_W[idx] = self.beta * self.v_W[idx] + self.lr * layer.dW
            self.v_b[idx] = self.beta * self.v_b[idx] + self.lr * layer.db
            # Apply update
            layer.W -= self.v_W[idx]
            layer.b -= self.v_b[idx]

def run_optimizer_benchmark(epochs=35):
    print("=" * 60)
    print("OPTIMIZER BENCHMARK: BGD vs SGD vs Momentum vs Nesterov")
    print("=" * 60)
    
    X_train, y_train, X_val, y_val, _, _, _ = load_downsampled_dataset((32, 32))
    N = len(X_train)
    batch_size = 32
    
    # Initialize base model architecture
    np.random.seed(42)
    base_model = MLPScratch(layer_dims=[1024, 128, 6], init_type="he", drop_rate=0.0, use_bn=False, l2_reg=0.0)
    
    configs = [
        ("Batch GD (lr=0.05)", "bgd", 0.05),
        ("Standard SGD (lr=0.02)", "sgd", 0.02),
        ("Momentum (lr=0.02, b=0.9)", "momentum", 0.02),
        ("NAG / Nesterov (lr=0.02, b=0.9)", "nesterov", 0.02)
    ]
    
    results = {}
    
    for opt_name, opt_type, lr in configs:
        print(f"\n--- Running: {opt_name} ---")
        # Deepcopy identical initial weights for fair benchmark
        model = copy.deepcopy(base_model)
        
        if opt_type == "bgd":
            opt = OptimizerBGD(model.layers, lr=lr)
        elif opt_type == "sgd":
            opt = OptimizerSGD(model.layers, lr=lr)
        elif opt_type == "momentum":
            opt = OptimizerMomentum(model.layers, lr=lr, beta=0.9)
        elif opt_type == "nesterov":
            opt = OptimizerNesterov(model.layers, lr=lr, beta=0.9)
            
        losses = []
        for epoch in range(epochs):
            if opt_type == "bgd":
                # Full batch gradient descent
                logits = model.forward(X_train, training=True)
                probs, loss = SoftmaxCrossEntropy.forward(logits, y_train)
                dZ = SoftmaxCrossEntropy.backward(probs, y_train)
                model.backward(dZ)
                opt.step()
                losses.append(float(loss))
            else:
                # Mini-batch gradient descent
                indices = np.arange(N)
                np.random.shuffle(indices)
                epoch_batch_losses = []
                
                for start in range(0, N, batch_size):
                    end = min(start + batch_size, N)
                    xb = X_train[indices[start:end]]
                    yb = y_train[indices[start:end]]
                    
                    if opt_type == "nesterov":
                        opt.apply_lookahead()
                        
                    logits = model.forward(xb, training=True)
                    probs, loss = SoftmaxCrossEntropy.forward(logits, yb)
                    epoch_batch_losses.append(loss)
                    
                    dZ = SoftmaxCrossEntropy.backward(probs, yb)
                    model.backward(dZ)
                    opt.step()
                    
                losses.append(float(np.mean(epoch_batch_losses)))
                
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1:2d}/{epochs:2d} -> Loss: {losses[-1]:.4f}")
                
        results[opt_name] = losses
        
    # Plot comparative loss curves
    os.makedirs(os.path.join("outputs", "figures"), exist_ok=True)
    plt.figure(figsize=(9, 5.5))
    colors = ["#7f7f7f", "#1f77b4", "#2ca02c", "#d62728"]
    
    for idx, (opt_name, losses) in enumerate(results.items()):
        plt.plot(range(1, epochs + 1), losses, label=opt_name, color=colors[idx], lw=2.2)
        
    plt.title("Optimizer Convergence Comparison on NEU-DET (32x32)")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    plot_path = os.path.join("outputs", "figures", "optimizers_comparison_loss_curves.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\n[PLOT] Saved optimizer comparison curves to {plot_path}")

if __name__ == "__main__":
    run_optimizer_benchmark(epochs=35)
