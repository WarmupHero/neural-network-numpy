"""
Data access for the library comparisons.

Input: the experiment configs in ``configs/`` and the raw CSVs in
``data/raw/``.
Output: for a given problem and seed, exactly the same train / validation /
test split (and scaling) that the main NumPy pipeline uses, plus that
split's constant-prediction baseline.

Reusing the main pipeline's preprocessing, rather than re-implementing it,
guarantees that every library is trained and tested on identical data.
"""

import contextlib
import io

from nn_from_scratch.config_loader import ConfigLoader
from nn_from_scratch.modeling.train import (
    CONFIG_FILES,
    PREPROCESSORS,
    constant_prediction_baseline,
    get_seeds,
)


def load_problem_config(problem_name):
    """
    Load and validate the main experiment config for one problem.

    Parameters
    ----------
    problem_name : str
        "classification" or "regression".

    Returns
    -------
    dict
        The validated config, as used by the main pipeline.

    Notes
    -----
    Processing:
    1. Look up the config file name in CONFIG_FILES.
    2. Load and validate it with ConfigLoader.
    """
    return ConfigLoader().load_and_validate(CONFIG_FILES[problem_name])


def problem_seeds(problem_name):
    """
    Return the seeds the main pipeline runs for one problem.

    Parameters
    ----------
    problem_name : str
        "classification" or "regression".

    Returns
    -------
    list of int
        The config's "seeds" list, or [RANDOM_SEED] if it has none.

    Notes
    -----
    Processing:
    1. Load the problem's config.
    2. Return get_seeds(config), the same function the main pipeline uses,
       so the comparisons always cover the same seeds.
    """
    return get_seeds(load_problem_config(problem_name))


def load_splits(problem_name, seed):
    """
    Build the train / validation / test split for one problem and seed.

    Parameters
    ----------
    problem_name : str
        "classification" or "regression".
    seed : int
        Seed for the split. The same seed in the main pipeline produces the
        same split.

    Returns
    -------
    splits : tuple of numpy.ndarray
        (X_train, y_train, X_val, y_val, X_test, y_test). X arrays have
        shape (n_samples, n_features), dtype float64 (scaled with training
        statistics when the config asks for scaling). y arrays have shape
        (n_samples, 1): 0/1 integer labels for classification, float64
        targets for regression.
    baseline : float
        Test metric of always predicting a constant fitted on the training
        targets (class proportion, or mean), as in the main pipeline.

    Notes
    -----
    Processing:
    1. Load the problem's config to read its preprocessing settings.
    2. Run the problem's preprocessor with this seed, with EDA plots
       switched off and its console output silenced.
    3. Compute the constant-prediction baseline for this split.
    """
    preprocessing = load_problem_config(problem_name)["preprocessing"]

    # The preprocessors print progress; silence it to keep the output readable.
    with contextlib.redirect_stdout(io.StringIO()):
        splits = PREPROCESSORS[problem_name]().get_data(
            show_eda=False,
            preprocessing_enabled=preprocessing["enabled"],
            scale_features=preprocessing["scale_features"],
            random_seed=seed,
            run_eda=False,
        )

    return splits, constant_prediction_baseline(problem_name, splits)
