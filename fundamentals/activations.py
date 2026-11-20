"""
fundamentals/activations.py
Comparative Study of Activation Functions and Gradient Flow:
1. Functions: Step (Heaviside), Sigmoid, Tanh, ReLU, LeakyReLU.
2. Exact analytical gradients for backpropagation.
3. Vanishing gradient simulation: deep chain rule gradient propagation through 10 layers.
4. Generates outputs/figures/activation_functions_and_gradients.png.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

def step(z):
    return np.where(z >= 0, 1.0, 0.0)

def step_grad(z):
    # Derivative is zero almost everywhere; undefined at 0 (Dirac delta)
    return np.zeros_like(z)

def sigmoid(z):
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))

def sigmoid_grad(z):
    s = sigmoid(z)
    return s * (1.0 - s)

def tanh(z):
    return np.tanh(z)

def tanh_grad(z):
    t = np.tanh(z)
    return 1.0 - t ** 2

def relu(z):
    return np.maximum(0.0, z)

def relu_grad(z):
    return np.where(z > 0, 1.0, 0.0)

def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)

def leaky_relu_grad(z, alpha=0.01):
    return np.where(z > 0, 1.0, alpha)

def plot_activations_and_gradient_flow():
    print("=" * 60)
    print("ACTIVATIONS & GRADIENT FLOW COMPARISON")
    print("=" * 60)
    
    z = np.linspace(-6, 6, 500)
    
    funcs = [
        ("Step (Heaviside)", step(z), step_grad(z), "#7f7f7f"),
        ("Sigmoid", sigmoid(z), sigmoid_grad(z), "#1f77b4"),
        ("Tanh", tanh(z), tanh_grad(z), "#ff7f0e"),
        ("ReLU", relu(z), relu_grad(z), "#2ca02c"),
        ("LeakyReLU (a=0.01)", leaky_relu(z), leaky_relu_grad(z), "#d62728")
    ]
    
    os.makedirs(os.path.join("outputs", "figures"), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    
    # 1. Activation outputs
    for name, f_val, _, color in funcs:
        axes[0].plot(z, f_val, label=name, color=color, lw=2)
    axes[0].set_title(r"Activation Functions $\sigma(z)$")
    axes[0].set_xlabel("Input $z$")
    axes[0].set_ylabel(r"Output $\sigma(z)$")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(loc="upper left")
    axes[0].set_ylim(-1.5, 3.5)
    
    # 2. Analytical gradients
    for name, _, g_val, color in funcs:
        axes[1].plot(z, g_val, label=f"d({name})/dz", color=color, lw=2)
    axes[1].set_title(r"Analytical Derivatives $\sigma'(z)$")
    axes[1].set_xlabel("Input $z$")
    axes[1].set_ylabel(r"Gradient $\sigma'(z)$")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(loc="upper right")
    axes[1].set_ylim(-0.1, 1.2)
    
    # 3. Simulated gradient magnitude across 10 stacked layers
    # Demonstrating Vanishing Gradient Problem: dL/dh_0 = prod_{l=1}^L (W_l^T * sigma'(z_l))
    num_layers = 10
    layers = np.arange(1, num_layers + 1)
    
    # Typical operating point around moderate activation input |z| = 2.0
    z_eval = 2.0
    grad_magnitudes = {
        "Sigmoid (max 0.25)": [sigmoid_grad(z_eval) ** l for l in layers],
        "Tanh (max 1.0, at z=2 is ~0.07)": [tanh_grad(z_eval) ** l for l in layers],
        "ReLU (constant 1.0 for z>0)": [relu_grad(z_eval) ** l for l in layers],
        "LeakyReLU (constant 1.0)": [leaky_relu_grad(z_eval) ** l for l in layers]
    }
    
    for label, mags in grad_magnitudes.items():
        axes[2].semilogy(layers, mags, "o-", label=label, lw=2)
    axes[2].set_title("Gradient Flow Decay Across 10 Deep Layers ($z=2.0$)")
    axes[2].set_xlabel("Depth / Number of Backprop Layers")
    axes[2].set_ylabel("Gradient Magnitude (Log Scale)")
    axes[2].grid(True, linestyle="--", alpha=0.6)
    axes[2].legend(loc="lower left")
    
    plt.tight_layout()
    fig_path = os.path.join("outputs", "figures", "activation_functions_and_gradients.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[PLOT] Saved activation comparison figure to {fig_path}")

if __name__ == "__main__":
    plot_activations_and_gradient_flow()
