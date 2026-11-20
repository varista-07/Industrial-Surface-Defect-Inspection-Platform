"""
fundamentals/neurons_demo.py
McCulloch-Pitts (M-P) Neuron and Sigmoid Neuron Implementations from Scratch:
1. McCulloch-Pitts binary threshold logic neuron (1943).
   - Logic functions: AND, OR, NOT, NAND.
   - Demonstration of XOR linear inseparability (Minsky & Papert, 1969).
2. Sigmoid Neuron:
   - Logistic activation sigma(z) = 1 / (1 + exp(-z)).
   - Analytical derivative sigma'(z) = sigma(z) * (1 - sigma(z)).
   - Binary cross-entropy training on 2D defect feature space (Contrast vs Defect Aspect Ratio).
   - Visual decision boundary and probability contour plot.
3. Generates outputs/figures/neurons_decision_boundaries.png.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

class McCullochPittsNeuron:
    """
    McCulloch-Pitts Neuron (1943)
    Computes: y = 1 if (sum(w_i * x_i) >= theta and all inhibitory inputs == 0) else 0.
    """
    def __init__(self, weights, threshold, inhibitory_indices=None):
        self.weights = np.array(weights, dtype=np.float32)
        self.threshold = float(threshold)
        self.inhibitory_indices = inhibitory_indices or []
        
    def activate(self, inputs):
        x = np.array(inputs, dtype=np.float32)
        # Inhibitory check: if any inhibitory input is active (1), output is unconditionally 0
        for idx in self.inhibitory_indices:
            if x[idx] == 1.0:
                return 0
        total_input = np.dot(self.weights, x)
        return 1 if total_input >= self.threshold else 0

def demo_mcculloch_pitts():
    print("=" * 60)
    print("1. McCULLOCH-PITTS NEURON LOGIC DEMO")
    print("=" * 60)
    
    # 2-input truth table
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    
    # AND gate: weights=[1, 1], threshold=2
    mp_and = McCullochPittsNeuron(weights=[1, 1], threshold=2)
    and_outs = [mp_and.activate(x) for x in inputs]
    print(f"AND Gate  (weights=[1,1], theta=2): {and_outs} (Expected: [0, 0, 0, 1])")
    assert and_outs == [0, 0, 0, 1], "AND gate mismatch"
    
    # OR gate: weights=[1, 1], threshold=1
    mp_or = McCullochPittsNeuron(weights=[1, 1], threshold=1)
    or_outs = [mp_or.activate(x) for x in inputs]
    print(f"OR Gate   (weights=[1,1], theta=1): {or_outs} (Expected: [0, 1, 1, 1])")
    assert or_outs == [0, 1, 1, 1], "OR gate mismatch"
    
    # NOT gate: input x1, inhibitory weight
    # x = [x1], weights=[0], threshold=0, inhibitory=[0]
    mp_not = McCullochPittsNeuron(weights=[0], threshold=0, inhibitory_indices=[0])
    not_outs = [mp_not.activate([0]), mp_not.activate([1])]
    print(f"NOT Gate  (inhibitory x1):          {not_outs} (Expected: [1, 0])")
    assert not_outs == [1, 0], "NOT gate mismatch"
    
    # NAND gate: weights=[-1, -1], threshold=-1
    # inputs: [0,0]->0>=-1 (1), [0,1]->-1>=-1 (1), [1,0]->-1>=-1 (1), [1,1]->-2>=-1 (0)
    mp_nand = McCullochPittsNeuron(weights=[-1, -1], threshold=-1)
    nand_outs = [mp_nand.activate(x) for x in inputs]
    print(f"NAND Gate (weights=[-1,-1], theta=-1): {nand_outs} (Expected: [1, 1, 1, 0])")
    assert nand_outs == [1, 1, 1, 0], "NAND gate mismatch"
    
    # XOR Inseparability proof:
    # XOR truth table: [0, 0]->0, [0, 1]->1, [1, 0]->1, [1, 1]->0
    # Requires:
    # w1*0 + w2*0 < theta  => theta > 0
    # w1*0 + w2*1 >= theta => w2 >= theta
    # w1*1 + w2*0 >= theta => w1 >= theta
    # w1*1 + w2*1 < theta  => w1 + w2 < theta
    # Adding w1>=theta and w2>=theta yields w1 + w2 >= 2*theta.
    # Since theta > 0, 2*theta > theta, creating a direct contradiction with w1 + w2 < theta.
    print("[PROOF] Single linear neuron mathematically cannot separate XOR (Minsky & Papert, 1969).")

class SigmoidNeuron:
    """
    Sigmoid Logistic Neuron trained with gradient descent on Binary Cross Entropy.
    """
    def __init__(self, in_features, lr=0.1):
        self.w = np.random.randn(in_features) * 0.1
        self.b = 0.0
        self.lr = lr
        
    @staticmethod
    def sigmoid(z):
        # Numerically stable sigmoid
        return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))
    
    @staticmethod
    def sigmoid_grad(a):
        return a * (1.0 - a)
    
    def forward(self, X):
        z = np.dot(X, self.w) + self.b
        return self.sigmoid(z)
    
    def train(self, X, y, epochs=200):
        losses = []
        N = len(y)
        for _ in range(epochs):
            y_hat = self.forward(X)
            # Binary Cross-Entropy
            eps = 1e-12
            loss = -np.mean(y * np.log(y_hat + eps) + (1.0 - y) * np.log(1.0 - y_hat + eps))
            losses.append(loss)
            
            # Gradient: dL/dz = y_hat - y
            dz = (y_hat - y) / N
            dw = np.dot(X.T, dz)
            db = np.sum(dz)
            
            self.w -= self.lr * dw
            self.b -= self.lr * db
        return losses

def demo_sigmoid_neuron():
    print("\n" + "=" * 60)
    print("2. SIGMOID NEURON & DECISION BOUNDARY DEMO")
    print("=" * 60)
    
    # Generate 2D synthetic defect features: [Defect Contrast, Aspect Ratio]
    # Class 0: Mild scratches (low contrast, high aspect ratio)
    # Class 1: Severe inclusion/patches (high contrast, low aspect ratio)
    N_samples = 100
    X0 = np.random.randn(N_samples, 2) * 0.5 + np.array([-1.0, 1.0])
    X1 = np.random.randn(N_samples, 2) * 0.5 + np.array([1.0, -1.0])
    X = np.vstack([X0, X1])
    y = np.array([0] * N_samples + [1] * N_samples)
    
    neuron = SigmoidNeuron(in_features=2, lr=0.5)
    losses = neuron.train(X, y, epochs=150)
    
    acc = np.mean((neuron.forward(X) >= 0.5) == y) * 100.0
    print(f"Sigmoid Neuron final loss: {losses[-1]:.4f}, Accuracy: {acc:.1f}%")
    print(f"Learned weights: {neuron.w}, bias: {neuron.b:.4f}")
    
    # Plotting decision boundary and probability contours
    os.makedirs(os.path.join("outputs", "figures"), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Subplot 1: Loss curve
    axes[0].plot(losses, color="#1f77b4", lw=2)
    axes[0].set_title("Sigmoid Neuron BCE Loss vs Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    
    # Subplot 2: Decision boundary in feature space
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = neuron.forward(grid).reshape(xx.shape)
    
    contour = axes[1].contourf(xx, yy, probs, levels=25, cmap="RdBu_r", alpha=0.7)
    cbar = plt.colorbar(contour, ax=axes[1])
    cbar.set_label("Defect Class 1 Probability P(y=1|x)")
    axes[1].contour(xx, yy, probs, levels=[0.5], colors="black", linewidths=2, linestyles="--")
    
    axes[1].scatter(X[y == 0, 0], X[y == 0, 1], c="blue", edgecolors="k", label="Class 0 (Mild Scratch)")
    axes[1].scatter(X[y == 1, 0], X[y == 1, 1], c="red", edgecolors="k", label="Class 1 (Severe Inclusion)")
    axes[1].set_title("Sigmoid Neuron Decision Boundary & Sigmoid Contours")
    axes[1].set_xlabel("Defect Contrast Feature ($x_1$)")
    axes[1].set_ylabel("Defect Aspect Ratio ($x_2$)")
    axes[1].legend(loc="upper right")
    
    plt.tight_layout()
    fig_path = os.path.join("outputs", "figures", "neurons_decision_boundaries.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[PLOT] Saved neuron figures to {fig_path}")

if __name__ == "__main__":
    demo_mcculloch_pitts()
    demo_sigmoid_neuron()
