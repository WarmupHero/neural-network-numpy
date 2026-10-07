"""
Tests for running experiments with several seeds: seed plumbing, the
constant-prediction baseline, config validation, and the multi-seed
analysis helpers.
"""

import numpy as np
import pandas as pd
import pytest

from nn_from_scratch.modeling.train import constant_prediction_baseline, get_seeds
from nn_from_scratch.analysis import error_removed, fails_baseline, mean_pm_std, select_by_validation
from nn_from_scratch.config_loader import ConfigLoader
from nn_from_scratch.nn.network import NeuralNetwork
from nn_from_scratch.features import train_val_test_split
from nn_from_scratch.config import RANDOM_SEED


# ------------------------------------------------------------------
# Seed plumbing
# ------------------------------------------------------------------

def test_split_changes_with_the_seed():
    """
    Different seeds give different train/validation/test splits.

    Notes
    -----
    A 100-row dataframe is split (stratified on "class") with seeds 42
    and 43; the sets of "x" values in the two training splits must differ.
    """
    df = pd.DataFrame({"x": np.arange(100), "class": np.arange(100) % 2})
    train_a, _, _ = train_val_test_split(df, stratify_col="class", random_seed=42)
    train_b, _, _ = train_val_test_split(df, stratify_col="class", random_seed=43)
    assert set(train_a["x"]) != set(train_b["x"])


def test_network_initialization_follows_the_seed():
    """
    Network weight initialization is reproducible from the seed and changes with it.

    Notes
    -----
    Three one-layer networks are built from the same config with seeds
    42, 42 and 43. The two seed-42 networks must have identical weights;
    the seed-43 network must have different weights.
    """
    config = {"input_dimension": 3, "layers": [{"type": "dense", "units": 4, "activation": "relu"}]}
    a, b, c = NeuralNetwork(random_seed=42), NeuralNetwork(random_seed=42), NeuralNetwork(random_seed=43)
    for net in (a, b, c):
        net.build_from_config(config)
    np.testing.assert_array_equal(a.layers[0].weights, b.layers[0].weights)
    assert not np.array_equal(a.layers[0].weights, c.layers[0].weights)


def test_get_seeds_defaults_to_the_project_seed():
    """
    get_seeds falls back to [RANDOM_SEED] and otherwise returns the configured seeds.

    Notes
    -----
    A config without "seeds" must give [RANDOM_SEED]; a config with
    "seeds": [1, 2] must give [1, 2].
    """
    assert get_seeds({"experiments": {}}) == [RANDOM_SEED]
    assert get_seeds({"experiments": {"seeds": [1, 2]}}) == [1, 2]


# ------------------------------------------------------------------
# Constant-prediction baseline
# ------------------------------------------------------------------

def splits(y_train, y_test):
    """
    Build a dataset-splits tuple that holds only training and test targets.

    Parameters
    ----------
    y_train : list of float or list of int
        Training targets.
    y_test : list of float or list of int
        Test targets.

    Returns
    -------
    tuple
        (None, y_train, None, None, None, y_test), where both targets are
        numpy.ndarray of shape (n, 1), dtype float64. This matches the
        (X_train, y_train, X_val, y_val, X_test, y_test) layout that
        `constant_prediction_baseline` expects.

    Notes
    -----
    Processing:
    1. Convert each list to a float64 array and reshape it to a column.
    2. Put them in their positions and fill the unused slots with None,
       since the baseline only reads y_train and y_test.
    """
    y_train = np.asarray(y_train, dtype=float).reshape(-1, 1)
    y_test = np.asarray(y_test, dtype=float).reshape(-1, 1)
    return None, y_train, None, None, None, y_test


def test_regression_baseline_predicts_the_training_mean():
    """
    The regression baseline predicts the training mean and is scored with MSE.

    Notes
    -----
    With training targets [1, 2, 3] and test targets [1, 5], the baseline
    must return 5.0, the MSE of always predicting 2.
    """
    # Training mean is 2; test targets 1 and 5 -> MSE = ((1-2)^2 + (5-2)^2) / 2 = 5
    assert constant_prediction_baseline("regression", splits([1, 2, 3], [1, 5])) == pytest.approx(5.0)


def test_classification_baseline_predicts_the_training_proportion():
    """
    The classification baseline predicts the training class-1 proportion and is scored with BCE.

    Notes
    -----
    With training labels [1, 0, 0, 0] the constant prediction is 0.25.
    For test labels [1, 0] the baseline must return
    -(log(0.25) + log(0.75)) / 2.
    """
    # Training proportion 0.25; BCE of predicting 0.25 for labels [1, 0]
    expected = -(np.log(0.25) + np.log(0.75)) / 2
    assert constant_prediction_baseline("classification", splits([1, 0, 0, 0], [1, 0])) == pytest.approx(expected)


# ------------------------------------------------------------------
# Config validation
# ------------------------------------------------------------------

def make_config(**experiments_extra):
    """
    Build a minimal valid classification config for ConfigLoader.validate.

    Parameters
    ----------
    **experiments_extra : dict
        Extra keys merged into the "experiments" block, for example
        `seeds=[42, 43]`.

    Returns
    -------
    dict
        Config with every required top-level, preprocessing and
        experiments key (plus `experiments_extra`), and one architecture
        "A" made of a single 1-unit sigmoid Dense layer.

    Notes
    -----
    Processing:
    1. Return a fresh dictionary literal each call, so tests cannot
       affect each other by mutating it.
    """
    return {
        "task_type": "classification",
        "input_dimension": 4,
        "loss": "bce",
        "preprocessing": {"enabled": True, "scale_features": True},
        "architectures": {"A": [{"type": "dense", "units": 1, "activation": "sigmoid"}]},
        "experiments": {
            "optimizers": ["sgd"], "learning_rates": [0.1], "batch_sizes": [16],
            "epochs": 10, "early_stopping": True, "patience": 2,
            "min_delta": 0.0, "min_epochs_before_early_stop": 0, **experiments_extra,
        },
    }


def test_config_accepts_seeds():
    """
    The config validator accepts a list of distinct non-negative integer seeds.

    Notes
    -----
    `validate` must return True for "seeds": [42, 43, 44].
    """
    assert ConfigLoader().validate(make_config(seeds=[42, 43, 44]))


@pytest.mark.parametrize("seeds", [[], [42, 42], [-1], [1.5], [True], "42", 42])
def test_config_rejects_invalid_seeds(seeds):
    """
    The config validator rejects invalid "seeds" values.

    Parameters
    ----------
    seeds : list, str or int
        Invalid value: [] (empty), [42, 42] (duplicate), [-1] (negative),
        [1.5] (not an integer), [True] (a boolean), "42" (a string, not a
        list) or 42 (an integer, not a list).

    Notes
    -----
    `validate` must raise ValueError.
    """
    with pytest.raises(ValueError):
        ConfigLoader().validate(make_config(seeds=seeds))


# ------------------------------------------------------------------
# Multi-seed analysis helpers
# ------------------------------------------------------------------

def run(val, test, baseline=10.0, diverged=False):
    """
    Build a minimal experiment-run dictionary for the analysis helpers.

    Parameters
    ----------
    val : float
        Best validation loss of the run.
    test : float
        Test metric of the run.
    baseline : float, default=10.0
        Test metric of the constant-prediction baseline.
    diverged : bool, default=False
        Whether the run diverged.

    Returns
    -------
    dict
        {"best_val_loss": val, "test_metric": test,
        "baseline_test_metric": baseline, "diverged": diverged}.

    Notes
    -----
    Processing:
    1. Put the arguments under the keys the analysis helpers read.
    """
    return {"best_val_loss": val, "test_metric": test, "baseline_test_metric": baseline, "diverged": diverged}


def test_mean_pm_std_uses_the_sample_standard_deviation():
    """
    mean_pm_std formats "mean ± sample std" and handles one or zero values.

    Notes
    -----
    [1, 2, 3] must give "2.00 ± 1.00" (the sample standard deviation,
    dividing by n - 1, is 1.0). A single value must give a standard
    deviation of 0.0, and an empty list must give "-".
    """
    assert mean_pm_std([1.0, 2.0, 3.0], digits=2) == "2.00 ± 1.00"
    assert mean_pm_std([4.0], digits=1) == "4.0 ± 0.0"
    assert mean_pm_std([]) == "-"


def test_selection_uses_validation_loss_and_skips_diverged_runs():
    """
    select_by_validation picks the lowest validation loss among runs that did not diverge.

    Notes
    -----
    Of three runs with validation losses 0.5, 0.1 and 0.3, the 0.1 run
    diverged, so the 0.3 run must be chosen (the identical object is
    returned). A list containing only a diverged run must give None.
    """
    runs = [run(val=0.5, test=9.0), run(val=0.1, test=8.0, diverged=True), run(val=0.3, test=1.0)]
    assert select_by_validation(runs) is runs[2]
    assert select_by_validation([run(0.1, 1.0, diverged=True)]) is None


def test_error_removed_and_baseline_failure():
    """
    error_removed and fails_baseline compare a run's test metric with the baseline.

    Notes
    -----
    A test metric of 2.5 against a baseline of 10.0 must give an error
    removed of 1 - 2.5 / 10 = 0.75. A run whose test metric equals the
    baseline fails it; a run slightly below the baseline (9.9) does not.
    """
    assert error_removed(run(val=0, test=2.5, baseline=10.0)) == pytest.approx(0.75)
    assert fails_baseline(run(val=0, test=10.0, baseline=10.0))
    assert not fails_baseline(run(val=0, test=9.9, baseline=10.0))
