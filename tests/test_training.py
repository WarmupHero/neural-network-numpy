"""
Smoke tests for the optimizers and the training loop on tiny synthetic problems.
"""

import numpy as np
import pytest

from nn_from_scratch.nn.losses import get_loss
from nn_from_scratch.nn.network import NeuralNetwork
from nn_from_scratch.nn.optimizers import get_optimizer
from nn_from_scratch.modeling.trainer import Trainer


def make_classification_data(n=200, seed=0):
    """
    Generate a linearly separable 2-feature binary classification problem.

    Parameters
    ----------
    n : int, default=200
        Number of samples.
    seed : int, default=0
        Seed for the random generator.

    Returns
    -------
    X : numpy.ndarray of shape (n, 2), dtype float64
        Standard-normal features.
    y : numpy.ndarray of shape (n, 1), dtype float64
        Labels: 1.0 where X[:, 0] + X[:, 1] > 0, else 0.0.

    Notes
    -----
    Processing:
    1. Draw X from a standard normal distribution.
    2. Label each row by the sign of X[:, 0] + X[:, 1] and reshape the
       labels to a column.
    """
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, 2))
    y = (X[:, 0] + X[:, 1] > 0).astype(float).reshape(-1, 1)
    return X, y


def build_classifier():
    """
    Build a small 2 -> 8 -> 1 binary classifier.

    Parameters
    ----------
    None
        The architecture and seed (42) are fixed.

    Returns
    -------
    NeuralNetwork
        Network with layers Dense(2, 8), ReLU, Dense(8, 1), Sigmoid, in
        training mode.

    Notes
    -----
    Processing:
    1. Create `NeuralNetwork(random_seed=42)`.
    2. Build it from a two-layer config without bias, batch norm or
       dropout.
    """
    network = NeuralNetwork(random_seed=42)
    network.build_from_config({
        "input_dimension": 2,
        "layers": [
            {"type": "dense", "units": 8, "activation": "relu"},
            {"type": "dense", "units": 1, "activation": "sigmoid"},
        ],
    })
    return network


@pytest.mark.parametrize("optimizer_name,learning_rate", [
    ("sgd", 0.5),
    ("momentum", 0.5),
    ("adabelief", 0.01),
])
def test_optimizer_reduces_loss(optimizer_name, learning_rate):
    """
    Each optimizer lowers the training loss on a simple problem.

    Parameters
    ----------
    optimizer_name : str
        Optimizer under test: "sgd", "momentum" or "adabelief".
    learning_rate : float
        Learning rate for that optimizer (0.5 for SGD and momentum, 0.01
        for AdaBelief).

    Notes
    -----
    The classifier takes 50 full-batch steps (forward, backward, update
    every trainable layer) on 200 samples. The final BCE must be below
    80% of the initial BCE.
    """
    X, y = make_classification_data()
    network = build_classifier()
    loss = get_loss("bce")
    optimizer = get_optimizer(optimizer_name, learning_rate)

    initial_loss = loss.forward(y, network.forward(X))

    for _ in range(50):
        y_pred = network.forward(X)
        network.backward(loss.backward(y, y_pred))
        for layer in network.get_trainable_layers():
            optimizer.update(layer)

    final_loss = loss.forward(y, network.forward(X))
    assert final_loss < 0.8 * initial_loss


def test_unknown_optimizer_raises():
    """
    Asking for an unsupported optimizer name raises ValueError.

    Notes
    -----
    `get_optimizer("rmsprop", 0.01)` must raise ValueError.
    """
    with pytest.raises(ValueError):
        get_optimizer("rmsprop", 0.01)


def test_trainer_fits_and_evaluates():
    """
    Trainer.fit records a decreasing loss history and Trainer.evaluate reports a good test BCE.

    Notes
    -----
    The classifier is trained with AdaBelief for 20 epochs (no early
    stopping) on 200 samples, validated on 50 and tested on 50. Asserted:
    all 20 epochs ran and were recorded, the last training loss is below
    the first, the test metric equals the test loss (both are BCE for
    classification), and the test BCE is below 0.3.
    """
    X, y = make_classification_data(n=300)
    X_train, y_train = X[:200], y[:200]
    X_val, y_val = X[200:250], y[200:250]
    X_test, y_test = X[250:], y[250:]

    trainer = Trainer(
        network=build_classifier(),
        loss_fn=get_loss("bce"),
        optimizer=get_optimizer("adabelief", 0.01),
        task_type="classification",
    )
    history = trainer.fit(X_train, y_train, X_val, y_val, epochs=20, batch_size=16, verbose=False)

    assert history["epochs_ran"] == 20
    assert len(history["train_loss"]) == 20
    assert history["train_loss"][-1] < history["train_loss"][0]

    results = trainer.evaluate(X_test, y_test)
    # The evaluation metric for classification is BCE, the same quantity as the loss
    assert results["test_metric"] == pytest.approx(results["test_loss"])
    assert results["test_metric"] < 0.3  # well below ln(2) ~= 0.693, the BCE of a coin flip
