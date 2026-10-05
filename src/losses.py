import numpy as np


class MSELoss:
    """
    Mean Squared Error loss for regression.

    Formula:
        MSE = (1 / n) * sum((y_true - y_pred)^2)

    This loss measures how far the predictions are from the true values.
    Larger errors are penalized more strongly because the difference is squared.

    This class provides:
    - forward pass: compute scalar loss
    - backward pass: compute dL/dy_pred
    """

    def forward(self, y_true, y_pred):
        """
        Compute the Mean Squared Error loss.

        Parameters
        ----------
        y_true : numpy.ndarray
            True target values, shape (batch_size, 1) or similar.
        y_pred : numpy.ndarray
            Predicted values, same shape as y_true.

        Returns
        -------
        float
            Mean squared error over the batch.

        Explanation
        -----------
        For each sample:
            error = y_true - y_pred

        Then square the error so negative and positive mistakes
        do not cancel each other out.

        Finally, average over the batch.
        """
        return np.mean((y_true - y_pred) ** 2)

    def backward(self, y_true, y_pred):
        """
        Compute the gradient of MSE with respect to predictions.

        For:
            L = (1/n) * sum((y_true - y_pred)^2)

        The derivative is:
            dL/dy_pred = (2/n) * (y_pred - y_true)

        Parameters
        ----------
        y_true : numpy.ndarray
            True target values.
        y_pred : numpy.ndarray
            Predicted values.

        Returns
        -------
        numpy.ndarray
            Gradient with respect to y_pred.

        Explanation
        -----------
        This tells us how the loss changes if the prediction changes.

        If y_pred is too large:
            gradient is positive
            -> gradient descent will push prediction down

        If y_pred is too small:
            gradient is negative
            -> gradient descent will push prediction up
        """
        # Number of samples in the mini-batch
        n = y_true.shape[0]

        # Gradient of MSE with respect to predictions
        return (2 / n) * (y_pred - y_true)


class BCELoss:
    """
    Binary Cross-Entropy loss for binary classification.

    Formula:
        BCE = -(1 / n) * sum(
            y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred)
        )

    Assumes:
    - y_true contains binary labels in {0, 1}
    - y_pred contains probabilities in (0, 1),
      typically the output of a sigmoid activation

    This class provides:
    - forward pass: compute scalar loss
    - backward pass: compute dL/dy_pred
    """

    def forward(self, y_true, y_pred):
        """
        Compute the Binary Cross-Entropy loss.

        Parameters
        ----------
        y_true : numpy.ndarray
            True binary labels, shape (batch_size, 1) or similar.
        y_pred : numpy.ndarray
            Predicted probabilities, same shape as y_true.

        Returns
        -------
        float
            Binary cross-entropy loss over the batch.

        Explanation
        -----------
        BCE compares:
        - the true class label
        - the predicted probability for that class

        It penalizes confident wrong predictions very strongly.

        Example:
        - if y_true = 1 and y_pred is close to 0, loss becomes very large
        - if y_true = 1 and y_pred is close to 1, loss is small
        """
        # Small constant for numerical stability.
        # Prevents log(0), which is undefined.
        epsilon = 1e-12

        # Clip probabilities to stay safely inside (0, 1)
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)

        # Apply the BCE formula and average over the batch
        loss = -np.mean(
            y_true * np.log(y_pred_clipped) +
            (1 - y_true) * np.log(1 - y_pred_clipped)
        )

        return loss

    def backward(self, y_true, y_pred):
        """
        Compute the gradient of BCE with respect to predictions.

        For:
            L = -(1/n) * sum(
                y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred)
            )

        The derivative is:
            dL/dy_pred = -(1/n) * [y_true / y_pred - (1 - y_true) / (1 - y_pred)]

        Parameters
        ----------
        y_true : numpy.ndarray
            True binary labels.
        y_pred : numpy.ndarray
            Predicted probabilities.

        Returns
        -------
        numpy.ndarray
            Gradient with respect to y_pred.

        Explanation
        -----------
        This gradient tells us how changing the predicted probability
        affects the BCE loss.

        Important:
        BCE expects probabilities, so y_pred should normally come
        from a sigmoid output layer.
        """
        # Small constant for numerical stability.
        # Prevents division by zero when y_pred is exactly 0 or 1.
        epsilon = 1e-12

        # Clip predictions before using them in the gradient formula
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)

        # Number of samples in the mini-batch
        n = y_true.shape[0]

        # Apply the derivative formula for BCE
        grad = -(
            (y_true / y_pred_clipped) -
            ((1 - y_true) / (1 - y_pred_clipped))
        ) / n

        return grad


def get_loss(name):
    """
    Factory function that returns a loss object by name.

    Parameters
    ----------
    name : str
        Name of the loss function.

    Returns
    -------
    object
        Instance of the requested loss class.

    Supported names
    ---------------
    - "mse"
    - "mean_squared_error"
    - "mean_squarederror"
    - "meansquared_error"
    - "bce"
    - "binary_crossentropy"
    - "binary_cross_entropy"

    Raises
    ------
    ValueError
        If the loss name is not supported.

    Explanation
    -----------
    This helper lets the rest of the project choose a loss function
    using a string from a config file instead of manually creating
    the object every time.
    """
    # Make the name lowercase so inputs like "MSE" or "BCE" still work
    name = name.lower()

    # Return the requested loss object
    if name in ["mse","mean_squared_error","mean_squarederror","meansquared_error"]:
        return MSELoss()
    elif name in ["bce", "binary_crossentropy", "binary_cross_entropy"]:
        return BCELoss()
    else:
        raise ValueError(f"Unsupported loss function: {name}")