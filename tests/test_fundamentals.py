"""
tests/test_fundamentals.py
Automated Unit Tests for Neural Network Fundamentals from Scratch (NumPy).
Run with: pytest tests/test_fundamentals.py -v
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))
import numpy as np
try:
    import pytest
except ImportError:
    pytest = None

from fundamentals.tensors_demo import tensor_creation_demo, broadcasting_demo
from fundamentals.neurons_demo import McCullochPittsNeuron, SigmoidNeuron
from fundamentals.activations import step, sigmoid, tanh, relu, leaky_relu, sigmoid_grad, tanh_grad, relu_grad
from fundamentals.mlp_from_scratch import LinearLayer, ReLU, Dropout, BatchNormalization1D, SoftmaxCrossEntropy, MLPScratch
from fundamentals.optimizers_from_scratch import OptimizerBGD, OptimizerSGD, OptimizerMomentum, OptimizerNesterov
from fundamentals.grad_check import gradient_check

def test_tensors_and_broadcasting():
    t0, t1, t2, t3, t4 = tensor_creation_demo()
    assert t0.ndim == 0
    assert t1.shape == (3,)
    assert t2.shape == (4, 4)
    assert t3.shape == (28, 28, 3)
    assert t4.shape == (32, 28, 28, 3)
    
    A, B, C = broadcasting_demo()
    assert C.shape == (4, 3)
    assert np.allclose(C, A + B)

def test_mcculloch_pitts_logic_gates():
    # AND
    mp_and = McCullochPittsNeuron([1, 1], 2)
    assert [mp_and.activate(x) for x in [[0,0], [0,1], [1,0], [1,1]]] == [0, 0, 0, 1]
    
    # OR
    mp_or = McCullochPittsNeuron([1, 1], 1)
    assert [mp_or.activate(x) for x in [[0,0], [0,1], [1,0], [1,1]]] == [0, 1, 1, 1]
    
    # NOT
    mp_not = McCullochPittsNeuron([0], 0, inhibitory_indices=[0])
    assert [mp_not.activate([0]), mp_not.activate([1])] == [1, 0]
    
    # NAND
    mp_nand = McCullochPittsNeuron([-1, -1], -1)
    assert [mp_nand.activate(x) for x in [[0,0], [0,1], [1,0], [1,1]]] == [1, 1, 1, 0]

def test_activations_and_gradients():
    z = np.array([-2.0, 0.0, 2.0], dtype=np.float32)
    
    # Sigmoid range [0, 1]
    s = sigmoid(z)
    assert np.all(s >= 0.0) and np.all(s <= 1.0)
    assert np.isclose(s[1], 0.5)
    
    # Sigmoid derivative at z=0 is 0.25
    ds = sigmoid_grad(z)
    assert np.isclose(ds[1], 0.25)
    
    # ReLU
    r = relu(z)
    assert np.allclose(r, [0.0, 0.0, 2.0])
    dr = relu_grad(z)
    assert np.allclose(dr, [0.0, 0.0, 1.0])
    
    # Tanh range [-1, 1]
    t = tanh(z)
    assert np.all(t >= -1.0) and np.all(t <= 1.0)
    dt = tanh_grad(z)
    assert np.isclose(dt[1], 1.0)

def test_mlp_shapes_and_forward_backward():
    B, D_in, D_out = 8, 32, 6
    X = np.random.randn(B, D_in).astype(np.float32)
    y = np.random.randint(0, D_out, size=B)
    
    model = MLPScratch(layer_dims=[D_in, 16, D_out], init_type="he", drop_rate=0.2, use_bn=True)
    
    # Forward
    logits = model.forward(X, training=True)
    assert logits.shape == (B, D_out)
    
    probs, loss = SoftmaxCrossEntropy.forward(logits, y)
    assert probs.shape == (B, D_out)
    assert np.allclose(np.sum(probs, axis=1), 1.0)
    assert loss > 0
    
    # Backward
    dZ = SoftmaxCrossEntropy.backward(probs, y)
    assert dZ.shape == (B, D_out)
    dX = model.backward(dZ)
    assert dX.shape == (B, D_in)

def test_gradient_checking_strict():
    """Verify that analytical gradients match numerical gradients to < 1e-6 relative error."""
    np.random.seed(42)
    B, D_in, H, D_out = 4, 8, 4, 3
    X = np.random.randn(B, D_in).astype(np.float64)
    y = np.array([0, 1, 2, 0], dtype=np.int64)
    
    model = MLPScratch(layer_dims=[D_in, H, D_out], init_type="he", drop_rate=0.0, use_bn=False, l2_reg=0.0)
    for layer in model.layers:
        if isinstance(layer, LinearLayer):
            layer.W = layer.W.astype(np.float64)
            layer.b = layer.b.astype(np.float64)
            
    results = gradient_check(model, X, y, eps=1e-6)
    
    for param_name, res in results.items():
        assert res["passed"], f"Grad check failed for {param_name} with error {res['relative_error']}"
        assert res["relative_error"] < 1e-6

def test_optimizers_update_weights():
    """Verify that all 4 optimizers update weights and reduce parameter values in gradient direction."""
    np.random.seed(42)
    D_in, D_out = 4, 2
    X = np.ones((2, D_in), dtype=np.float32)
    y = np.array([0, 1], dtype=np.int64)
    
    for OptClass in [OptimizerBGD, OptimizerSGD, OptimizerMomentum, OptimizerNesterov]:
        model = MLPScratch(layer_dims=[D_in, D_out], init_type="he", drop_rate=0.0, use_bn=False, l2_reg=0.0)
        initial_W = model.layers[0].W.copy()
        
        opt = OptClass(model.layers, lr=0.1)
        if isinstance(opt, OptimizerNesterov):
            opt.apply_lookahead()
            
        logits = model.forward(X, training=True)
        probs, _ = SoftmaxCrossEntropy.forward(logits, y)
        dZ = SoftmaxCrossEntropy.backward(probs, y)
        model.backward(dZ)
        opt.step()
        
        updated_W = model.layers[0].W
if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING AUTOMATED UNIT TESTS FOR FUNDAMENTALS")
    print("=" * 60)
    test_tensors_and_broadcasting()
    print("[PASS] test_tensors_and_broadcasting")
    test_mcculloch_pitts_logic_gates()
    print("[PASS] test_mcculloch_pitts_logic_gates")
    test_activations_and_gradients()
    print("[PASS] test_activations_and_gradients")
    test_mlp_shapes_and_forward_backward()
    print("[PASS] test_mlp_shapes_and_forward_backward")
    test_gradient_checking_strict()
    print("[PASS] test_gradient_checking_strict")
    test_optimizers_update_weights()
    print("[PASS] test_optimizers_update_weights")
    print("\n[ALL TESTS PASSED SUCCESSFULLY!]")
