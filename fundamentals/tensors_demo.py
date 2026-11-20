"""
fundamentals/tensors_demo.py
Demonstration of Tensor Fundamentals from Scratch using NumPy:
1. Tensor creation across ranks (0D to 4D batch tensors).
2. Broadcasting arithmetic with visual examples.
3. Inner products, matrix multiplication, and Einstein summation (einsum).
4. Reshaping, transposition, and spatial flattening.
5. Numerical gradient checking via finite differences.
Saves visual figures to outputs/figures/tensors_and_broadcasting.png.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

def tensor_creation_demo():
    print("=" * 60)
    print("1. TENSOR CREATION ACROSS RANKS")
    print("=" * 60)
    
    # Rank-0: Scalar
    t0 = np.array(42.0)
    # Rank-1: Vector (e.g., class label vector or bias)
    t1 = np.array([0.1, 0.4, 0.5])
    # Rank-2: Matrix (e.g., single grayscale image patch or weight matrix)
    t2 = np.random.randn(4, 4)
    # Rank-3: 3D Tensor (e.g., single RGB image [Height, Width, Channels])
    t3 = np.random.randn(28, 28, 3)
    # Rank-4: 4D Tensor (e.g., batch of images [Batch, Height, Width, Channels])
    t4 = np.random.randn(32, 28, 28, 3)
    
    print(f"Rank-0 Scalar: shape={t0.shape}, ndim={t0.ndim}, dtype={t0.dtype}")
    print(f"Rank-1 Vector: shape={t1.shape}, ndim={t1.ndim}, dtype={t1.dtype}")
    print(f"Rank-2 Matrix: shape={t2.shape}, ndim={t2.ndim}, dtype={t2.dtype}")
    print(f"Rank-3 Tensor: shape={t3.shape}, ndim={t3.ndim}, dtype={t3.dtype}")
    print(f"Rank-4 Batch:  shape={t4.shape}, ndim={t4.ndim}, dtype={t4.dtype}")
    return t0, t1, t2, t3, t4

def broadcasting_demo():
    print("\n" + "=" * 60)
    print("2. BROADCASTING ARITHMETIC RULES")
    print("=" * 60)
    
    # Practical DL scenario: Normalizing a batch of feature maps with channel means
    # Batch of 4 images, 3x3 spatial, 2 channels: shape (4, 3, 3, 2)
    features = np.ones((4, 3, 3, 2), dtype=np.float32) * 10.0
    channel_means = np.array([2.0, 5.0], dtype=np.float32)  # shape (2,)
    
    # Broadcasting expands (2,) to (1, 1, 1, 2) and matches (4, 3, 3, 2)
    normalized = features - channel_means
    print(f"Features shape:      {features.shape}")
    print(f"Channel means shape: {channel_means.shape}")
    print(f"Result shape:        {normalized.shape}")
    print(f"Channel 0 sample value (expected 8.0): {normalized[0, 0, 0, 0]}")
    print(f"Channel 1 sample value (expected 5.0): {normalized[0, 0, 0, 1]}")
    
    # Visualizing 2D broadcasting
    A = np.arange(4).reshape(4, 1)  # (4, 1)
    B = np.arange(3).reshape(1, 3)  # (1, 3)
    C = A + B                       # (4, 3)
    
    os.makedirs(os.path.join("outputs", "figures"), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    im0 = axes[0].imshow(A, cmap="Blues", aspect="auto")
    axes[0].set_title(f"A: shape (4, 1)")
    for i in range(4):
        axes[0].text(0, i, f"{A[i, 0]}", ha="center", va="center", color="black", fontweight="bold")
        
    im1 = axes[1].imshow(B, cmap="Greens", aspect="auto")
    axes[1].set_title(f"B: shape (1, 3)")
    for j in range(3):
        axes[1].text(j, 0, f"{B[0, j]}", ha="center", va="center", color="black", fontweight="bold")
        
    im2 = axes[2].imshow(C, cmap="Purples", aspect="auto")
    axes[2].set_title(f"A + B: shape (4, 3)")
    for i in range(4):
        for j in range(3):
            axes[2].text(j, i, f"{C[i, j]}", ha="center", va="center", color="white" if C[i,j]>3 else "black")
            
    plt.tight_layout()
    plot_path = os.path.join("outputs", "figures", "tensors_and_broadcasting.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[PLOT] Saved broadcasting figure to {plot_path}")
    return A, B, C

def inner_products_and_einsum_demo():
    print("\n" + "=" * 60)
    print("3. INNER PRODUCTS, MATRIX MULTIPLICATION & EINSUM")
    print("=" * 60)
    
    # Linear layer forward pass: X (Batch, In_features) @ W (In_features, Out_features) + b
    X = np.random.randn(8, 16)
    W = np.random.randn(16, 6)
    b = np.random.randn(6)
    
    # Method 1: Standard matrix multiplication @
    Y1 = X @ W + b
    # Method 2: np.matmul
    Y2 = np.matmul(X, W) + b
    # Method 3: Einstein summation (einsum)
    Y3 = np.einsum("bi,io->bo", X, W) + b
    
    assert np.allclose(Y1, Y2) and np.allclose(Y1, Y3), "Matrix multiplication methods mismatch!"
    print(f"Input batch X: {X.shape}, Weights W: {W.shape}, Bias b: {b.shape}")
    print(f"Output Y shape: {Y1.shape}")
    print(f"Verification: np.matmul and np.einsum match @ operator within {np.max(np.abs(Y1 - Y3)):.2e}")

def numerical_gradient_check_demo():
    print("\n" + "=" * 60)
    print("4. NUMERICAL GRADIENT CHECKING VIA FINITE DIFFERENCES")
    print("=" * 60)
    
    # Objective: f(X) = 0.5 * sum((X @ W + b - Y_true)**2)
    # Analytical gradient: df/dX = (X @ W + b - Y_true) @ W^T
    X = np.random.randn(4, 5)
    W = np.random.randn(5, 3)
    b = np.random.randn(3)
    Y_true = np.random.randn(4, 3)
    
    def loss_func(x):
        pred = x @ W + b
        return 0.5 * np.sum((pred - Y_true) ** 2)
    
    # Analytical gradient
    pred = X @ W + b
    grad_analytical = (pred - Y_true) @ W.T
    
    # Numerical gradient using centered difference: df/dx = (f(x + h) - f(x - h)) / (2*h)
    h = 1e-6
    grad_numerical = np.zeros_like(X)
    it = np.nditer(X, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        orig_val = X[idx]
        
        X[idx] = orig_val + h
        loss_plus = loss_func(X)
        
        X[idx] = orig_val - h
        loss_minus = loss_func(X)
        
        grad_numerical[idx] = (loss_plus - loss_minus) / (2.0 * h)
        X[idx] = orig_val
        it.iternext()
        
    rel_error = np.linalg.norm(grad_analytical - grad_numerical) / (
        np.linalg.norm(grad_analytical) + np.linalg.norm(grad_numerical) + 1e-12
    )
    print(f"Analytical gradient norm: {np.linalg.norm(grad_analytical):.6f}")
    print(f"Numerical gradient norm:  {np.linalg.norm(grad_numerical):.6f}")
    print(f"Relative error:           {rel_error:.2e}")
    assert rel_error < 1e-6, f"Gradient check failed with relative error {rel_error}"
    print("[PASS] Tensor gradient checking verified successfully (< 1e-6 relative error).")

if __name__ == "__main__":
    tensor_creation_demo()
    broadcasting_demo()
    inner_products_and_einsum_demo()
    numerical_gradient_check_demo()
