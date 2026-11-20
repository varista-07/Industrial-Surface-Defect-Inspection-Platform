"""
fundamentals/unsupervised_learning.py
Unsupervised, Associative, and Energy-Based Neural Learning from Scratch:
1. Classical Hebbian Learning (Hebb, 1949) & demonstration of unbounded weight growth.
2. Modified Hebbian / Oja's Subspace Rule (Oja, 1982) & convergence to the 1st Principal Component.
3. Competitive Learning / Winner-Take-All (WTA) defect cluster centroid discovery.
4. Compact Restricted Boltzmann Machine (RBM) / Energy-Based Model via Contrastive Divergence (CD-1).
Generates outputs/figures/unsupervised_and_energy_learning.png.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

def demo_hebbian_and_oja():
    print("=" * 60)
    print("1. HEBBIAN VS OJA'S MODIFIED HEBBIAN LEARNING")
    print("=" * 60)
    
    # 2D synthetic correlated defect features (e.g., length vs width of surface cracks)
    N = 200
    mean = [0.0, 0.0]
    cov = [[2.0, 1.6], [1.6, 1.5]]  # Strongly correlated along diagonal
    X = np.random.multivariate_normal(mean, cov, N)
    
    # True 1st Principal Component via SVD
    _, _, Vt = np.linalg.svd(X)
    true_pc1 = Vt[0]
    if true_pc1[0] < 0:
        true_pc1 = -true_pc1
    print(f"True 1st Principal Component (PCA): {true_pc1}")
    
    # 1. Classical Hebbian Learning: dw = eta * y * x
    w_hebb = np.random.randn(2)
    w_hebb /= np.linalg.norm(w_hebb)
    eta_hebb = 0.01
    hebb_norms = []
    
    for epoch in range(10):
        for x in X:
            y = np.dot(w_hebb, x)
            w_hebb += eta_hebb * y * x
            hebb_norms.append(np.linalg.norm(w_hebb))
    print(f"Classical Hebbian initial norm: 1.0 -> Final norm: {hebb_norms[-1]:.2f} (Exponential Divergence)")
    
    # 2. Oja's Rule: dw = eta * y * (x - y * w)
    w_oja = np.random.randn(2)
    w_oja /= np.linalg.norm(w_oja)
    eta_oja = 0.02
    oja_norms = []
    oja_alignments = []
    
    for epoch in range(25):
        for x in X:
            y = np.dot(w_oja, x)
            w_oja += eta_oja * y * (x - y * w_oja)
            oja_norms.append(np.linalg.norm(w_oja))
            
            # Cosine similarity to true PC1
            align = abs(np.dot(w_oja, true_pc1) / (np.linalg.norm(w_oja) * np.linalg.norm(true_pc1)))
            oja_alignments.append(align)
            
    print(f"Oja's Rule Final Weight: {w_oja}, Norm: {np.linalg.norm(w_oja):.4f}")
    print(f"Alignment to True PC1: {oja_alignments[-1]:.6f} (Matches First Principal Component!)")
    return X, true_pc1, w_oja, hebb_norms, oja_norms, oja_alignments

def demo_competitive_learning():
    print("\n" + "=" * 60)
    print("2. COMPETITIVE LEARNING (WINNER-TAKE-ALL)")
    print("=" * 60)
    
    # 3 clusters representing 3 distinct surface defect profiles:
    # Cluster 1: Pitted surface (localized, small area)
    # Cluster 2: Rolled-in scale (medium area, irregular)
    # Cluster 3: Scratches (long, high aspect ratio)
    c1 = np.random.randn(50, 2) * 0.3 + np.array([-2.0, -1.0])
    c2 = np.random.randn(50, 2) * 0.4 + np.array([0.0, 2.0])
    c3 = np.random.randn(50, 2) * 0.3 + np.array([2.5, -0.5])
    X = np.vstack([c1, c2, c3])
    
    # 3 codebook weight vectors (prototypes)
    K = 3
    W_prototypes = np.random.uniform(-1, 1, (K, 2))
    eta = 0.05
    
    for epoch in range(40):
        # Decay learning rate
        lr = eta * (1.0 - epoch / 40)
        np.random.shuffle(X)
        for x in X:
            # Winner-take-all: argmin Euclidean distance
            dists = np.linalg.norm(W_prototypes - x, axis=1)
            winner = np.argmin(dists)
            # Update winner towards sample
            W_prototypes[winner] += lr * (x - W_prototypes[winner])
            
    print(f"Learned defect cluster prototypes:\n{W_prototypes}")
    return X, W_prototypes

class CompactRBM:
    """
    Restricted Boltzmann Machine trained via 1-step Contrastive Divergence (CD-1).
    Energy function: E(v, h) = -v^T W h - a^T v - b^T h
    """
    def __init__(self, n_visible=16, n_hidden=8, lr=0.05):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.lr = lr
        
        # Small Gaussian init
        self.W = np.random.randn(n_visible, n_hidden) * 0.05
        self.a = np.zeros(n_visible)  # visible bias
        self.b = np.zeros(n_hidden)   # hidden bias

    def sample_h(self, v):
        p_h = sigmoid(np.dot(v, self.W) + self.b)
        h_sample = (np.random.rand(*p_h.shape) < p_h).astype(np.float32)
        return p_h, h_sample

    def sample_v(self, h):
        p_v = sigmoid(np.dot(h, self.W.T) + self.a)
        v_sample = (np.random.rand(*p_v.shape) < p_v).astype(np.float32)
        return p_v, v_sample

    def train_cd1(self, X, epochs=60):
        recon_errors = []
        N = len(X)
        for epoch in range(epochs):
            # 1. Positive phase (data driven)
            v0 = X
            p_h0, h0 = self.sample_h(v0)
            
            # 2. Negative phase (model hallucination / reconstruction via Gibbs sampling)
            p_v1, v1 = self.sample_v(h0)
            p_h1, h1 = self.sample_h(v1)
            
            # 3. Parameter updates: E_data[v h^T] - E_model[v h^T]
            pos_assoc = np.dot(v0.T, p_h0) / N
            neg_assoc = np.dot(v1.T, p_h1) / N
            
            self.W += self.lr * (pos_assoc - neg_assoc)
            self.a += self.lr * np.mean(v0 - v1, axis=0)
            self.b += self.lr * np.mean(p_h0 - p_h1, axis=0)
            
            recon_err = np.mean((v0 - p_v1) ** 2)
            recon_errors.append(recon_err)
            
        return recon_errors

def demo_rbm():
    print("\n" + "=" * 60)
    print("3. RESTRICTED BOLTZMANN MACHINE (RBM) / ENERGY-BASED MODEL")
    print("=" * 60)
    
    # 4x4 binary defect patch patterns (horizontal scratch, vertical scratch)
    N_samples = 200
    X_patterns = []
    for _ in range(N_samples // 2):
        # Horizontal line at random row
        img = np.zeros((4, 4))
        r = np.random.randint(0, 4)
        img[r, :] = 1.0
        X_patterns.append(img.flatten())
    for _ in range(N_samples // 2):
        # Vertical line at random col
        img = np.zeros((4, 4))
        c = np.random.randint(0, 4)
        img[:, c] = 1.0
        X_patterns.append(img.flatten())
        
    X_bin = np.array(X_patterns, dtype=np.float32)
    rbm = CompactRBM(n_visible=16, n_hidden=6, lr=0.1)
    errors = rbm.train_cd1(X_bin, epochs=80)
    
    print(f"RBM Initial Reconstruction MSE: {errors[0]:.4f} -> Final MSE: {errors[-1]:.4f}")
    return errors, rbm

def run_all_unsupervised():
    X_corr, pc1, w_oja, hebb_norms, oja_norms, oja_alignments = demo_hebbian_and_oja()
    X_comp, W_proto = demo_competitive_learning()
    rbm_errors, rbm = demo_rbm()
    
    # Visual multi-panel figure
    os.makedirs(os.path.join("outputs", "figures"), exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    
    # Subplot 1: Oja vs Hebbian weight norm stability
    axes[0, 0].plot(hebb_norms[:500], label="Classical Hebbian (Diverging)", color="#d62728", lw=2)
    axes[0, 0].plot(oja_norms[:500], label="Oja's Rule (Normalized to 1.0)", color="#2ca02c", lw=2)
    axes[0, 0].set_title("1. Hebbian Instability vs Oja Stability")
    axes[0, 0].set_xlabel("Update Step")
    axes[0, 0].set_ylabel("Weight Vector Norm $||w||_2$")
    axes[0, 0].grid(True, linestyle="--", alpha=0.6)
    axes[0, 0].legend()
    
    # Subplot 2: Oja convergence to PC1
    axes[0, 1].plot(oja_alignments, color="#1f77b4", lw=2)
    axes[0, 1].set_title("2. Oja Alignment to 1st Principal Component")
    axes[0, 1].set_xlabel("Update Step")
    axes[0, 1].set_ylabel("Cosine Similarity to True PC1")
    axes[0, 1].grid(True, linestyle="--", alpha=0.6)
    axes[0, 1].set_ylim(0, 1.05)
    
    # Subplot 3: Competitive learning prototypes
    axes[1, 0].scatter(X_comp[:, 0], X_comp[:, 1], c="#aec7e8", edgecolors="k", alpha=0.6, label="Defect Feature Samples")
    axes[1, 0].scatter(W_proto[:, 0], W_proto[:, 1], c="red", marker="X", s=200, edgecolors="black", label="Learned Prototypes (WTA)")
    axes[1, 0].set_title("3. Competitive Learning (Cluster Centroids)")
    axes[1, 0].set_xlabel("Feature $x_1$")
    axes[1, 0].set_ylabel("Feature $x_2$")
    axes[1, 0].grid(True, linestyle="--", alpha=0.6)
    axes[1, 0].legend()
    
    # Subplot 4: RBM Reconstruction Error
    axes[1, 1].plot(rbm_errors, color="#9467bd", lw=2)
    axes[1, 1].set_title("4. Restricted Boltzmann Machine Energy CD-1 Loss")
    axes[1, 1].set_xlabel("CD-1 Epoch")
    axes[1, 1].set_ylabel("Reconstruction MSE")
    axes[1, 1].grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    fig_path = os.path.join("outputs", "figures", "unsupervised_and_energy_learning.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"\n[PLOT] Saved unsupervised learning comparison to {fig_path}")

if __name__ == "__main__":
    run_all_unsupervised()
