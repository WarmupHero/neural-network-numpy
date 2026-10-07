"""
Tests for the library comparison (nn_from_scratch.benchmarks): identical
data splits, the scikit-learn wrappers, and the report's model selection.

Tests that need scikit-learn are skipped when it is not installed (it is an
optional dependency), so the core test suite runs without it.
"""

import math

import numpy as np
import pytest

from nn_from_scratch.benchmarks import report
from nn_from_scratch.benchmarks.data import load_splits
from nn_from_scratch.modeling.train import PREPROCESSORS, constant_prediction_baseline

# ------------------------------------------------------------------
# Data: the comparison must use the main pipeline's exact splits
# ------------------------------------------------------------------


@pytest.mark.parametrize("problem_name", ["classification", "regression"])
def test_benchmark_splits_match_the_main_pipeline(problem_name):
    """
    load_splits returns exactly the main pipeline's split for a seed.

    Parameters
    ----------
    problem_name : str
        "classification" or "regression".

    Notes
    -----
    The main pipeline's preprocessor is run directly with seed 43 and its
    six arrays are compared element by element with load_splits's, as is
    the constant-prediction baseline.
    """
    splits, baseline = load_splits(problem_name, 43)
    expected = PREPROCESSORS[problem_name]().get_data(random_seed=43, run_eda=False)
    for got, want in zip(splits, expected):
        np.testing.assert_array_equal(got, want)
    assert baseline == constant_prediction_baseline(problem_name, expected)


# ------------------------------------------------------------------
# Report: selection and summaries (no scikit-learn needed)
# ------------------------------------------------------------------


def run(seed, val, test, baseline=1.0, **extra):
    """
    Build a minimal library-style result record for the report tests.

    Parameters
    ----------
    seed : int
        Seed of the run.
    val : float
        Validation metric.
    test : float
        Test metric.
    baseline : float, default=1.0
        Constant-prediction test metric.
    **extra
        Any other fields, e.g. model="svm".

    Returns
    -------
    dict
        A record with "seed", "val_metric", "test_metric",
        "baseline_test_metric" and "params", plus the extra fields.

    Notes
    -----
    Processing:
    1. Fill in the fields the report reads, with an empty params dict.
    """
    return {
        "seed": seed,
        "val_metric": val,
        "test_metric": test,
        "baseline_test_metric": baseline,
        "params": {},
        **extra,
    }


def test_selection_uses_validation_not_test():
    """
    select_per_seed picks the lowest validation metric on each seed.

    Notes
    -----
    On seed 1 the run with the best test score has the worse validation
    score, so it must not be selected. Runs with a non-finite validation
    metric are ignored.
    """
    runs = [
        run(1, val=0.2, test=0.01),
        run(1, val=0.1, test=0.50),
        run(2, val=math.nan, test=0.0),
        run(2, val=0.3, test=0.30),
    ]
    selected = report.select_per_seed(runs, "val_metric")
    assert [r["test_metric"] for r in selected] == [0.50, 0.30]


def test_summarize_computes_error_removed_and_most_common_config():
    """
    summarize reports error removed and the most frequently chosen configuration.

    Notes
    -----
    With baseline 2.0 and test metrics 0.5 and 1.0, error removed is 0.75
    and 0.5. Both runs share one configuration, chosen on 2 of 2 seeds.
    """
    selected = [
        run(1, 0.1, 0.5, baseline=2.0, params={"C": 1}),
        run(2, 0.1, 1.0, baseline=2.0, params={"C": 1}),
    ]
    summary = report.summarize("x", selected, "test_accuracy")
    assert summary["error_removed"] == pytest.approx([0.75, 0.5])
    assert summary["config"] == "C=1 (2/2 seeds)"


def test_mean_pm_std_ignores_missing_values():
    """
    mean_pm_std skips None values, e.g. metrics missing from older files.

    Notes
    -----
    [1, None, 3] -> mean 2, sample std ~1.414.
    """
    assert report.mean_pm_std([1.0, None, 3.0], 2) == "2.00 ± 1.41"
    assert report.mean_pm_std([None], 2) == "-"


# ------------------------------------------------------------------
# scikit-learn wrappers
# ------------------------------------------------------------------


def make_splits(problem_name, n=120, seed=0, n_features=3):
    """
    Build small synthetic train / validation / test splits.

    Parameters
    ----------
    problem_name : str
        "classification" (0/1 integer labels) or "regression" (float
        targets).
    n : int, default=120
        Total number of samples, split 50 / 25 / 25 %.
    seed : int, default=0
        Seed for the random data.
    n_features : int, default=3
        Number of input features.

    Returns
    -------
    tuple of numpy.ndarray
        (X_train, y_train, X_val, y_val, X_test, y_test), with n_features
        features and y arrays of shape (n_part, 1), like the real pipeline.

    Notes
    -----
    Processing:
    1. Draw standard-normal features.
    2. Target: a linear function of the features (thresholded at 0 for
       classification), so the problem is learnable.
    3. Cut the arrays into the three parts.
    """
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, n_features))
    signal = X @ np.linspace(-2.0, 1.0, n_features)
    y = (signal > 0).astype(np.int64) if problem_name == "classification" else signal
    y = y.reshape(-1, 1)
    a, b = n // 2, 3 * n // 4
    return X[:a], y[:a], X[a:b], y[a:b], X[b:], y[b:]


def test_expand_grid_handles_lists_of_grids_and_tuples():
    """
    expand_grid forms every combination and turns JSON lists into tuples.

    Notes
    -----
    A union of two grids (2 + 1 combinations) gives 3 configurations, and
    the [32, 32] hidden-layer list becomes the tuple scikit-learn expects.
    """
    pytest.importorskip("sklearn")
    from nn_from_scratch.benchmarks.sklearn_models import expand_grid

    grid = [
        {"hidden_layer_sizes": [[32]], "alpha": [0.1, 1.0]},
        {"hidden_layer_sizes": [[32, 32]]},
    ]
    combos = expand_grid(grid)
    assert len(combos) == 3
    assert {"hidden_layer_sizes": (32, 32)} in combos


@pytest.mark.parametrize(
    "problem_name, model_name, grid",
    [
        ("classification", "logistic_regression", {"C": [0.1, 1.0]}),
        ("classification", "random_forest", {"n_estimators": [20]}),
        ("regression", "ridge", {"alpha": [0.1, 1.0]}),
        ("regression", "gradient_boosting", {"max_iter": [50]}),
    ],
)
def test_run_model_returns_one_scored_record_per_config(problem_name, model_name, grid):
    """
    run_model trains every configuration and records its metrics.

    Parameters
    ----------
    problem_name : str
        Problem type of the synthetic data.
    model_name : str
        scikit-learn model family to run.
    grid : dict
        A small hyperparameter grid.

    Notes
    -----
    Each record must hold finite validation / test / training metrics, the
    secondary metric, the baseline and a training time, and the models
    must beat a trivial level on this easy synthetic problem.
    """
    pytest.importorskip("sklearn")
    from nn_from_scratch.benchmarks.sklearn_models import expand_grid, run_model

    records = run_model(
        problem_name, model_name, grid, make_splits(problem_name), seed=7, baseline=0.5
    )
    assert len(records) == len(expand_grid(grid))
    secondary = "test_accuracy" if problem_name == "classification" else "test_r2"
    for record in records:
        assert record["library"] == "sklearn" and record["seed"] == 7
        for key in ("val_metric", "test_metric", "train_metric", secondary, "train_seconds"):
            assert math.isfinite(record[key])
        assert record[secondary] > 0.6


def test_runs_are_reproducible_with_the_same_seed():
    """
    Seeded models give identical results when run twice.

    Notes
    -----
    A random forest uses its random_state; with the same seed both runs
    must produce the same test metric.
    """
    pytest.importorskip("sklearn")
    from nn_from_scratch.benchmarks.sklearn_models import run_model

    splits = make_splits("classification")
    first = run_model("classification", "random_forest", {"n_estimators": [20]}, splits, 3, 0.7)
    second = run_model("classification", "random_forest", {"n_estimators": [20]}, splits, 3, 0.7)
    assert first[0]["test_metric"] == second[0]["test_metric"]


def test_unknown_model_name_raises():
    """
    build_model rejects a model name that isn't defined for the problem.

    Notes
    -----
    "ridge" is a regression model, so asking for it as a classifier fails.
    """
    pytest.importorskip("sklearn")
    from nn_from_scratch.benchmarks.sklearn_models import build_model

    with pytest.raises(ValueError):
        build_model("classification", "ridge", {}, seed=0)


# ------------------------------------------------------------------
# TensorFlow / Keras wrappers
# ------------------------------------------------------------------

# Short training so the tests stay fast.
QUICK_TRAINING = {"epochs": 3, "early_stopping": {"patience": 2, "start_from_epoch": 0}}


def test_keras_network_mirrors_the_architecture_config():
    """
    build_network creates the same layer sequence as the NumPy architecture.

    Notes
    -----
    A2-bn (3 hidden ReLU layers with batch norm, no bias) must become, per
    hidden layer, Dense (without bias) -> BatchNormalization -> Activation,
    followed by the output Dense -> Activation.
    """
    pytest.importorskip("tensorflow")
    from nn_from_scratch.benchmarks.data import load_problem_config
    from nn_from_scratch.benchmarks.keras_models import build_network

    config = load_problem_config("regression")
    model = build_network(config["architectures"]["A2-bn"], config["input_dimension"], seed=0)
    kinds = [type(layer).__name__ for layer in model.layers[1:]]  # skip the input layer
    assert kinds == ["Dense", "BatchNormalization", "Activation"] * 3 + ["Dense", "Activation"]
    assert all(not layer.use_bias for layer in model.layers if type(layer).__name__ == "Dense")


@pytest.mark.parametrize("name", ["sgd", "momentum", "adam"])
def test_keras_optimizers(name):
    """
    build_optimizer returns the matching Keras optimizer.

    Parameters
    ----------
    name : str
        Optimizer name from the benchmark config.

    Notes
    -----
    "momentum" must be SGD with momentum 0.9; unknown names are rejected.
    """
    pytest.importorskip("tensorflow")
    from nn_from_scratch.benchmarks.keras_models import build_optimizer

    optimizer = build_optimizer(name, 0.01)
    expected = {"sgd": "SGD", "momentum": "SGD", "adam": "Adam"}[name]
    assert type(optimizer).__name__ == expected
    if name == "momentum":
        assert optimizer.momentum == pytest.approx(0.9)
    with pytest.raises(ValueError):
        build_optimizer("adabelief", 0.01)


@pytest.mark.parametrize("problem_name, n_features", [("classification", 4), ("regression", 8)])
def test_keras_run_model_records_every_combination(problem_name, n_features):
    """
    run_model trains each grid combination and records finite metrics.

    Parameters
    ----------
    problem_name : str
        Problem type; its config fixes the architecture's input size.
    n_features : int
        Number of features matching that config (4 or 8).

    Notes
    -----
    Two combinations of A1, trained for 3 epochs on synthetic data, must
    each produce a record with finite metrics, the loss histories and the
    run's settings.
    """
    pytest.importorskip("tensorflow")
    from nn_from_scratch.benchmarks.keras_models import run_model

    splits = make_splits(problem_name, n_features=n_features)
    grid = {"optimizer": ["adam"], "learning_rate": [0.01], "batch_size": [16, 32]}
    records = run_model(problem_name, "A1", grid, splits, 5, 0.5, training=QUICK_TRAINING)
    assert len(records) == 2
    for record in records:
        assert record["library"] == "tensorflow" and record["model"] == "A1"
        assert not record["diverged"]
        assert record["epochs_ran"] == len(record["train_loss_history"]) <= 3
        for key in ("val_metric", "test_metric", "train_metric", "train_seconds"):
            assert math.isfinite(record[key])


def test_keras_runs_are_reproducible():
    """
    The same seed gives the same Keras result twice.

    Notes
    -----
    Seeding with keras.utils.set_random_seed and deterministic TensorFlow
    operations make weight initialization and shuffling repeatable.
    """
    pytest.importorskip("tensorflow")
    from nn_from_scratch.benchmarks.keras_models import run_model

    splits = make_splits("classification", n_features=4)
    grid = {"optimizer": ["sgd"], "learning_rate": [0.1], "batch_size": [16]}
    first = run_model("classification", "A1", grid, splits, 9, 0.5, training=QUICK_TRAINING)
    second = run_model("classification", "A1", grid, splits, 9, 0.5, training=QUICK_TRAINING)
    assert first[0]["test_metric"] == second[0]["test_metric"]
