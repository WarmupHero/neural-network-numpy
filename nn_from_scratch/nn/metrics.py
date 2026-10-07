"""
Evaluation metrics reported for trained models.

The metrics reuse the training losses (BCELoss and MSELoss) so that the
reported numbers are directly comparable with the loss curves. Both
functions accept 1-D or column-vector inputs and return a plain float.
"""

import numpy as np

from nn_from_scratch.nn.losses import BCELoss, MSELoss

def binary_cross_entropy(y_true, y_pred):
    """
    Compute Binary Cross-Entropy (BCE) as the classification evaluation metric.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, 1)
        True labels, 0 or 1. Converted to float64.
    y_pred : array-like of shape (n_samples,) or (n_samples, 1)
        Predicted probabilities from the network, usually the output
        of a sigmoid activation. Converted to float64. Must have the same
        number of elements as y_true.

    Returns
    -------
    float
        Mean binary cross-entropy. Lower is better.

    Notes
    -----
    Processing:

    1. Convert both inputs to float64 NumPy arrays and reshape them to
       column vectors of shape (n_samples, 1).
    2. Compute the loss with BCELoss().forward, which clips the
       probabilities to avoid log(0).
    3. Convert the result to a Python float.

    The same BCE used as the training loss is used to evaluate the model,
    so the reported metric is directly comparable with the loss curves.

    It computes:
        -mean(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))

    Reusing BCELoss keeps a single implementation (including the
    clipping that avoids log(0)).
    """

    # Convert inputs to NumPy arrays and reshape them into column vectors,
    # so shapes like (n_samples,) and (n_samples, 1) are handled the same way.
    y_true = np.asarray(y_true, dtype=float).reshape(-1, 1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1, 1)

    return float(BCELoss().forward(y_true, y_pred))


def mean_squared_error(y_true, y_pred):
    """
    Compute Mean Squared Error (MSE) as the regression evaluation metric.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, 1)
        True target values. Converted to float64.
    y_pred : array-like of shape (n_samples,) or (n_samples, 1)
        Predicted target values. Converted to float64. Must have the same
        number of elements as y_true.

    Returns
    -------
    float
        Mean squared error. Lower is better.

    Notes
    -----
    Processing:

    1. Convert both inputs to float64 NumPy arrays and reshape them to
       column vectors of shape (n_samples, 1).
    2. Compute the loss with MSELoss().forward.
    3. Convert the result to a Python float.

    The same MSE used as the training loss is used to evaluate the model,
    so the reported metric is directly comparable with the loss curves.

    It computes:
        mean((y_true - y_pred)^2)

    Reusing MSELoss keeps a single implementation.
    """

    # Convert inputs to NumPy arrays and reshape them into column vectors,
    # so shapes like (n_samples,) and (n_samples, 1) are handled the same way.
    y_true = np.asarray(y_true, dtype=float).reshape(-1, 1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1, 1)

    return float(MSELoss().forward(y_true, y_pred))
