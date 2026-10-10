"""
Tests for the Muon optimizer and its Newton-Schulz orthogonalization.
"""

import numpy as np
import pytest

from nn_numpy.nn.layers import BatchNorm, Dense
from nn_numpy.nn.optimizers import AdaBelief, Muon, get_optimizer, newton_schulz_orthogonalize

# ------------------------------------------------------------------
# Newton-Schulz orthogonalization
# ------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(8, 32), (32, 8), (16, 16), (32, 1)])
def test_newton_schulz_pushes_singular_values_towards_one(shape):
    """
    The iteration keeps the shape and moves every singular value close to 1.

    Parameters
    ----------
    shape : tuple of (int, int)
        Shape of the random input matrix: wide, tall, square and a column.

    Notes
    -----
    The quintic coefficients oscillate around 1 rather than converging
    exactly, so the bound is the band Muon is known to produce, [0.6, 1.25].
    """
    matrix = np.random.RandomState(0).normal(size=shape)
    result = newton_schulz_orthogonalize(matrix)

    assert result.shape == shape
    singular_values = np.linalg.svd(result, compute_uv=False)
    assert singular_values.min() > 0.6
    assert singular_values.max() < 1.25


def test_newton_schulz_keeps_the_singular_vectors():
    """
    The result shares the input's singular vectors (O = U S' V^T for G = U S V^T).

    Notes
    -----
    If both matrices share U and V, then O^T G = V S' S V^T, which is
    symmetric with positive eigenvalues. A rotation or reflection of the
    singular vectors would break that.
    """
    matrix = np.random.RandomState(1).normal(size=(6, 10))
    product = newton_schulz_orthogonalize(matrix).T @ matrix

    np.testing.assert_allclose(product, product.T, atol=1e-10)
    eigenvalues = np.linalg.eigvalsh(product)
    assert eigenvalues[-6:].min() > 0  # rank 6: the top 6 eigenvalues are positive


def test_newton_schulz_handles_a_zero_matrix():
    """
    An all-zero input returns zeros instead of dividing by zero.
    """
    result = newton_schulz_orthogonalize(np.zeros((4, 3)))

    assert np.all(np.isfinite(result))
    assert np.all(result == 0)


# ------------------------------------------------------------------
# Muon routing and update rule
# ------------------------------------------------------------------


def make_dense_with_grads(fan_in=4, fan_out=6, seed=0):
    """
    Build a Dense layer with a bias and give it a backward pass.

    Parameters
    ----------
    fan_in, fan_out : int, default=4, 6
        Layer shape.
    seed : int, default=0
        Seed for the weights and the input batch.

    Returns
    -------
    Dense
        Layer whose dweights (fan_in, fan_out) and dbias (1, fan_out) are set.

    Notes
    -----
    Processing:
    1. Create the layer with a seeded initialization and a bias.
    2. Run forward on a random batch of 5 rows and backward with a random
       upstream gradient.
    """
    rng = np.random.RandomState(seed)
    layer = Dense(fan_in, fan_out, random_state=rng, use_bias=True)
    layer.forward(rng.normal(size=(5, fan_in)))
    layer.backward(rng.normal(size=(5, fan_out)))
    return layer


@pytest.mark.parametrize("fan_in,fan_out", [(4, 6), (6, 4)])
def test_first_muon_step_on_weights_matches_the_rule(fan_in, fan_out):
    """
    The first weight step is learning_rate * shape factor * orthogonalized direction.

    Parameters
    ----------
    fan_in, fan_out : int
        Layer shape; (4, 6) has a shape factor sqrt(6 / 4), (6, 4) has 1.

    Notes
    -----
    On the first step the buffer is (1 - m) * g, so the Nesterov direction
    is (1 - m) * g + m * (1 - m) * g = (1 - m) * (1 + m) * g.
    """
    layer = make_dense_with_grads(fan_in, fan_out)
    g = layer.dweights.copy()
    before = layer.weights.copy()
    learning_rate, momentum = 0.02, 0.95

    Muon(learning_rate=learning_rate, momentum=momentum).update(layer)

    direction = (1 - momentum) * (1 + momentum) * g
    scale = np.sqrt(max(1.0, fan_out / fan_in))
    expected = before - learning_rate * scale * newton_schulz_orthogonalize(direction)
    np.testing.assert_allclose(layer.weights, expected, atol=1e-12)


def test_without_nesterov_the_step_uses_the_buffer():
    """
    With nesterov=False, the first direction is the buffer (1 - m) * g.
    """
    layer = make_dense_with_grads(4, 4)
    g = layer.dweights.copy()
    before = layer.weights.copy()

    Muon(learning_rate=0.02, momentum=0.9, nesterov=False).update(layer)

    expected = before - 0.02 * newton_schulz_orthogonalize(0.1 * g)
    np.testing.assert_allclose(layer.weights, expected, atol=1e-12)


def test_the_bias_follows_the_adabelief_rule():
    """
    Muon hands the bias to AdaBelief: three steps give AdaBelief's exact result.

    Notes
    -----
    Two identical layers receive the same gradients for three steps; one
    is updated by Muon, the other by a plain AdaBelief with the same
    learning rate. Their biases must be identical, while the weights differ.
    """
    muon_layer = make_dense_with_grads(seed=3)
    adabelief_layer = make_dense_with_grads(seed=3)
    muon, adabelief = Muon(learning_rate=0.05), AdaBelief(learning_rate=0.05)
    rng = np.random.RandomState(4)

    for _ in range(3):
        dweights, dbias = rng.normal(size=(4, 6)), rng.normal(size=(1, 6))
        for layer in (muon_layer, adabelief_layer):
            layer.dweights, layer.dbias = dweights.copy(), dbias.copy()
        muon.update(muon_layer)
        adabelief.update(adabelief_layer)

    np.testing.assert_array_equal(muon_layer.bias, adabelief_layer.bias)
    assert not np.allclose(muon_layer.weights, adabelief_layer.weights)


def test_batch_norm_parameters_use_the_fallback():
    """
    A BatchNorm layer (gamma and beta, no weights) is updated exactly like AdaBelief.
    """
    rng = np.random.RandomState(5)
    X, upstream = rng.normal(size=(8, 3)), rng.normal(size=(8, 3))
    layers = []
    for optimizer in (Muon(learning_rate=0.05), AdaBelief(learning_rate=0.05)):
        bn = BatchNorm(3)
        bn.forward(X)
        bn.backward(upstream)
        optimizer.update(bn)
        layers.append(bn)

    muon_bn, adabelief_bn = layers
    for name, param in muon_bn.get_params().items():
        np.testing.assert_array_equal(param, adabelief_bn.get_params()[name])


def test_get_optimizer_builds_muon_with_the_fixed_settings():
    """
    "muon" (any case) builds Muon with momentum 0.95, Nesterov and 5 steps.
    """
    optimizer = get_optimizer("Muon", 0.1)

    assert isinstance(optimizer, Muon)
    assert optimizer.learning_rate == 0.1
    assert optimizer.momentum == 0.95
    assert optimizer.nesterov is True
    assert optimizer.ns_steps == 5
    assert optimizer.fallback.learning_rate == 0.1


@pytest.mark.parametrize("shape", [(8, 32), (32, 8), (32, 1)])
def test_weights_match_pytorch_muon(shape):
    """
    Three Muon steps match torch.optim.Muon on the same gradients.

    Parameters
    ----------
    shape : tuple of (int, int)
        NumPy weight shape (fan_in, fan_out); PyTorch stores the transpose.

    Notes
    -----
    PyTorch runs Newton-Schulz in bfloat16 and the NumPy network in float64,
    so the weights agree to about 1e-3 rather than exactly. Weight decay is
    switched off in PyTorch (its default is 0.1) to match the NumPy rule.
    """
    torch = pytest.importorskip("torch")
    layer = Dense(shape[0], shape[1], random_state=np.random.RandomState(1), use_bias=True)
    torch_weights = torch.nn.Parameter(torch.tensor(layer.weights.T.copy()))
    torch_muon = torch.optim.Muon([torch_weights], lr=0.02, weight_decay=0.0)
    numpy_muon = Muon(learning_rate=0.02)
    rng = np.random.RandomState(2)

    for _ in range(3):
        g = rng.normal(size=shape)
        layer.dweights, layer.dbias = g.copy(), np.zeros((1, shape[1]))
        numpy_muon.update(layer)
        torch_weights.grad = torch.tensor(g.T.copy())
        torch_muon.step()

    np.testing.assert_allclose(layer.weights, torch_weights.detach().numpy().T, atol=2e-3)
