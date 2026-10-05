import numpy as np

from src.losses import BCELoss, MSELoss

def binary_cross_entropy(y_true, y_pred):
    """
    Compute Binary Cross-Entropy (BCE) as the classification evaluation metric.

    Parameters
    ----------
    y_true : numpy.ndarray
        True labels with shape (n_samples, 1) or (n_samples,).
        Expected values are 0 or 1.

    y_pred : numpy.ndarray
        Predicted probabilities from the network, usually the output
        of a sigmoid activation. Shape should match y_true.

    Returns
    -------
    float
        Mean binary cross-entropy. Lower is better.

    Explanation
    -----------
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
    y_true : numpy.ndarray
        True target values with shape (n_samples, 1) or (n_samples,).

    y_pred : numpy.ndarray
        Predicted target values with shape matching y_true.

    Returns
    -------
    float
        Mean squared error. Lower is better.

    Explanation
    -----------
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
