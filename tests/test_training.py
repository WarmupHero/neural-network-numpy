"""
Smoke tests for the optimizers and the training loop on tiny synthetic problems.
"""

import numpy as np
import pytest

from src.losses import get_loss
from src.network import NeuralNetwork
from src.optimizers import get_optimizer
from src.train import Trainer


def make_classification_data(n=200, seed=0):
    """Linearly separable 2-feature binary problem."""
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, 2))
    y = (X[:, 0] + X[:, 1] > 0).astype(float).reshape(-1, 1)
    return X, y


def build_classifier():
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
    with pytest.raises(ValueError):
        get_optimizer("rmsprop", 0.01)


def test_trainer_fits_and_evaluates():
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
