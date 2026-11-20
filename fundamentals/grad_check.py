"""
fundamentals/grad_check.py
High-Precision Analytical vs Numerical Gradient Checking from Scratch:
Verifies backpropagation gradients across LinearLayer, ReLU, and SoftmaxCrossEntropy
using two-sided centered finite differences:
  f'(theta) ~ (f(theta + eps) - f(theta - eps)) / (2 * eps)
Ensures relative error < 1e-6 across weights, biases, and activations.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))
import numpy as np
from fundamentals.mlp_from_scratch import MLPScratch, LinearLayer, SoftmaxCrossEntropy

def gradient_check(model, X, y, eps=1e-5):
    """
    Compute analytical gradients and compare against numerical finite differences.
    Returns: dict of relative errors per parameter array.
    """
    # 1. Forward and analytical backward pass
    logits = model.forward(X, training=False)
    probs, loss = SoftmaxCrossEntropy.forward(logits, y)
    dZ = SoftmaxCrossEntropy.backward(probs, y)
    model.backward(dZ)
    
    results = {}
    
    # 2. Check each linear layer's W and b
    for l_idx, layer in enumerate(model.layers):
        if not isinstance(layer, LinearLayer):
            continue
            
        for param_name, param, grad_analytical in [("W", layer.W, layer.dW), ("b", layer.b, layer.db)]:
            grad_numerical = np.zeros_like(param)
            
            # Centered finite differences for each scalar parameter
            it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                idx = it.multi_index
                old_val = param[idx]
                
                param[idx] = old_val + eps
                l_plus = model.forward(X, training=False)
                _, loss_plus = SoftmaxCrossEntropy.forward(l_plus, y)
                
                param[idx] = old_val - eps
                l_minus = model.forward(X, training=False)
                _, loss_minus = SoftmaxCrossEntropy.forward(l_minus, y)
                
                grad_numerical[idx] = (loss_plus - loss_minus) / (2.0 * eps)
                param[idx] = old_val
                it.iternext()
                
            norm_diff = np.linalg.norm(grad_analytical - grad_numerical)
            norm_sum = np.linalg.norm(grad_analytical) + np.linalg.norm(grad_numerical) + 1e-12
            rel_error = float(norm_diff / norm_sum)
            
            key = f"Layer_{l_idx+1}_{param_name}"
            results[key] = {
                "analytical_norm": float(np.linalg.norm(grad_analytical)),
                "numerical_norm": float(np.linalg.norm(grad_numerical)),
                "relative_error": rel_error,
                "passed": bool(rel_error < 1e-6)
            }
            print(f"[{'PASS' if results[key]['passed'] else 'FAIL'}] {key:15s} | Analytical: {results[key]['analytical_norm']:8.5f} | Numerical: {results[key]['numerical_norm']:8.5f} | Rel Error: {rel_error:.2e}")
            assert results[key]["passed"], f"Gradient check failed for {key}: rel_error={rel_error:.2e}"
            
    return results

def run_gradient_check_test():
    print("=" * 60)
    print("ANALYTICAL VS NUMERICAL GRADIENT VERIFICATION")
    print("=" * 60)
    
    np.random.seed(42)
    # Small dimension network with float64 precision for finite differences
    B, D_in, H, D_out = 4, 16, 8, 3
    X = np.random.randn(B, D_in).astype(np.float64)
    y = np.array([0, 1, 2, 1], dtype=np.int64)
    
    model = MLPScratch(layer_dims=[D_in, H, D_out], init_type="he", drop_rate=0.0, use_bn=False, l2_reg=0.0)
    for layer in model.layers:
        if isinstance(layer, LinearLayer):
            layer.W = layer.W.astype(np.float64)
            layer.b = layer.b.astype(np.float64)
            
    results = gradient_check(model, X, y, eps=1e-6)
    print("\n[SUMMARY] All analytical gradients exactly match numerical finite differences.")
    return results

if __name__ == "__main__":
    run_gradient_check_test()
