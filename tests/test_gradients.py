"""
Numerical gradient checks for the from-scratch backpropagation.

Every analytic gradient (activations, losses, and full networks) is compared
against a central finite-difference estimate:

    dL/dw ~= (L(w + eps) - L(w - eps)) / (2 * eps)
"""

import numpy as np
import pytest

from src.activations import get_activation
from src.losses import get_loss
from src.network import NeuralNetwork

EPS = 1e-6
RTOL = 1e-5
ATOL = 1e-8


def numerical_grad(f, x):
    """Central finite-difference gradient of scalar function f at array x (modified in place, then restored)."""
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        original = x[idx]
        x[idx] = original + EPS
        f_plus = f()
        x[idx] = original - EPS
        f_minus = f()
        x[idx] = original
        grad[idx] = (f_plus - f_minus) / (2 * EPS)
    return grad


def away_from_zero(rng, shape, margin=0.1):
    """Random values with |x| >= margin, so ReLU's kink at 0 is never inside the finite-difference window."""
    x = rng.uniform(margin, 2.0, size=shape)
    return x * rng.choice([-1.0, 1.0], size=shape)


@pytest.mark.parametrize("name", ["relu", "sigmoid", "tanh", "linear"])
def test_activation_backward_matches_numerical(name):
    rng = np.random.RandomState(0)
    x = away_from_zero(rng, (5, 3))
    upstream = rng.normal(size=(5, 3))  # arbitrary dL/dy, so L = sum(upstream * y)

    activation = get_activation(name)
    activation.forward(x)
    analytic = activation.backward(upstream)

    numeric = numerical_grad(lambda: np.sum(upstream * get_activation(name).forward(x)), x)

    np.testing.assert_allclose(analytic, numeric, rtol=RTOL, atol=ATOL)


def test_mse_backward_matches_numerical():
    rng = np.random.RandomState(1)
    y_true = rng.normal(size=(8, 1))
    y_pred = rng.normal(size=(8, 1))
    loss = get_loss("mse")

    analytic = loss.backward(y_true, y_pred)
    numeric = numerical_grad(lambda: loss.forward(y_true, y_pred), y_pred)

    np.testing.assert_allclose(analytic, numeric, rtol=RTOL, atol=ATOL)


def test_bce_backward_matches_numerical():
    rng = np.random.RandomState(2)
    y_true = rng.randint(0, 2, size=(8, 1)).astype(float)
    y_pred = rng.uniform(0.05, 0.95, size=(8, 1))  # stay clear of the clipping region
    loss = get_loss("bce")

    analytic = loss.backward(y_true, y_pred)
    numeric = numerical_grad(lambda: loss.forward(y_true, y_pred), y_pred)

    np.testing.assert_allclose(analytic, numeric, rtol=RTOL, atol=ATOL)


NETWORK_CASES = [
    # (id, loss, hidden layers, output activation)
    ("A1-style_bce", "bce", [("sigmoid", 6)], "sigmoid"),
    ("A2-style_bce", "bce", [("relu", 6), ("relu", 6), ("relu", 6)], "sigmoid"),
    ("A1-style_mse", "mse", [("sigmoid", 6)], "linear"),
    ("A2-style_mse", "mse", [("relu", 6), ("relu", 6), ("relu", 6)], "linear"),
    ("tanh_mse", "mse", [("tanh", 5), ("tanh", 4)], "linear"),
]


@pytest.mark.parametrize("loss_name,hidden,output_activation",
                         [c[1:] for c in NETWORK_CASES], ids=[c[0] for c in NETWORK_CASES])
def test_network_weight_gradients_match_numerical(loss_name, hidden, output_activation):
    rng = np.random.RandomState(3)
    input_dim, batch = 4, 10

    layers = [{"type": "dense", "units": units, "activation": act} for act, units in hidden]
    layers.append({"type": "dense", "units": 1, "activation": output_activation})

    network = NeuralNetwork(random_seed=42)
    network.build_from_config({"input_dimension": input_dim, "layers": layers})
    loss = get_loss(loss_name)

    X = rng.normal(size=(batch, input_dim))
    if loss_name == "bce":
        y = rng.randint(0, 2, size=(batch, 1)).astype(float)
    else:
        y = rng.normal(size=(batch, 1))

    # Analytic gradients via backprop
    y_pred = network.forward(X)
    network.backward(loss.backward(y, y_pred))
    dense_layers = network.get_trainable_layers()
    analytic = [layer.dweights.copy() for layer in dense_layers]

    def loss_value():
        return loss.forward(y, network.forward(X))

    for layer, analytic_grad in zip(dense_layers, analytic):
        numeric_grad = numerical_grad(loss_value, layer.weights)
        np.testing.assert_allclose(analytic_grad, numeric_grad, rtol=1e-4, atol=1e-7)
