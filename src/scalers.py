import numpy as np

class StandardScaler:
    """
    Standardize features by removing the mean and scaling by the standard deviation.

    Formula
    -------
        X_scaled = (X - mean) / std

    This is done feature-by-feature, meaning each column is scaled separately.

    Why this is useful
    ------------------
    Standardization helps neural networks train more smoothly because:
    - features are put on a similar scale
    - very large-valued features do not dominate smaller-valued ones
    - optimization is usually more stable
    """

    def __init__(self):
        """
        Create an empty scaler.

        Attributes
        ----------
        mean_ : numpy.ndarray or None
            Feature-wise mean learned from the training data.
        std_ : numpy.ndarray or None
            Feature-wise standard deviation learned from the training data.
        """
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        """
        Learn the feature-wise mean and standard deviation from the data.

        Parameters
        ----------
        X : numpy.ndarray
            Input data of shape (n_samples, n_features)

        Returns
        -------
        StandardScaler
            The fitted scaler object.

        Explanation
        -----------
        axis=0 means:
        - compute one mean per column
        - compute one standard deviation per column

        So if X has shape (100, 4), then:
        - self.mean_ will have shape (4,)
        - self.std_ will have shape (4,)
        """
        # Compute the mean of each feature column
        self.mean_ = X.mean(axis=0)

        # Compute the standard deviation of each feature column
        self.std_ = X.std(axis=0)

        # Prevent division by zero for constant features.
        #
        # If a feature has std = 0, then all its values are identical.
        # Replacing 0 with a tiny number avoids numerical errors.
        # In practice, (X - mean) will be 0 for that feature anyway,
        # so the transformed values will remain 0.
        self.std_ = np.where(self.std_ == 0, 1e-8, self.std_)

        return self

    def transform(self, X):
        """
        Apply standardization using the stored mean and standard deviation.

        Parameters
        ----------
        X : numpy.ndarray
            Input data to scale.

        Returns
        -------
        numpy.ndarray
            Standardized version of X.

        Raises
        ------
        ValueError
            If fit() has not been called first.

        Important
        ---------
        In a proper ML workflow:
        - fit() should be called on training data only
        - transform() should then be applied to train/validation/test
          using the same stored training statistics
        """
        # Make sure the scaler has already learned mean and std
        if self.mean_ is None or self.std_ is None:
            raise ValueError("You must call fit() before transform()")

        # Apply the standard scaling formula feature-by-feature
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        """
        Fit the scaler on X, then immediately transform X.

        Parameters
        ----------
        X : numpy.ndarray
            Input data.

        Returns
        -------
        numpy.ndarray
            Standardized version of X.

        Explanation
        -----------
        This is just a convenience method equivalent to:

            scaler.fit(X)
            X_scaled = scaler.transform(X)
        """
        return self.fit(X).transform(X)