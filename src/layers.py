import numpy as np

class Dense:
    """
    A fully connected (dense) layer without a bias term.

    This layer performs a linear transformation:

        Z = X @ W

    Shape convention used in this project:
    - X: (batch_size, input_dim)
    - W: (input_dim, output_dim)
    - Z: (batch_size, output_dim)

    Note:
    The formula is often written as z = W x, which usually assumes
    x is a column vector. In this implementation, each sample is stored as
    a row, so the equivalent NumPy operation is X @ W.

    The layer is bias-free by design: only the weight matrix is trained.
    """

    def __init__(self, input_dim, output_dim, random_state=None):
        """
        Create a dense layer and initialize its trainable weights.

        Parameters
        ----------
        input_dim : int
            Number of input features entering the layer.
        output_dim : int
            Number of output neurons produced by the layer.
        random_state : numpy.random.RandomState or None
            Optional random generator for reproducible weight initialization.
            If None, NumPy's default random generator is used.
        """

        # Save the layer dimensions so the object remembers:
        # - how many values it expects as input
        # - how many values it should produce as output
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Choose which random number generator to use.
        # If a seeded generator is passed in, use it for reproducibility.
        # Otherwise, use NumPy's default random generator.
        rng = np.random if random_state is None else random_state

        # Initialize weights with small random values from a normal distribution.
        #
        # loc=0.0   -> mean of the distribution is 0
        # scale=0.1 -> small spread, so weights start close to 0
        #
        # Weight matrix shape:
        # (input_dim, output_dim)
        #
        # Example:
        # if input_dim = 4 and output_dim = 3,
        # then weights.shape = (4, 3)
        self.weights = rng.normal(
            loc=0.0,
            scale=0.1,
            size=(self.input_dim, self.output_dim)
        )

        # This will store the input batch from the forward pass.
        # We need it later during backpropagation to compute dL/dW.
        self.input = None

        # This will store the gradient of the loss with respect to the weights:
        # dL/dW
        #
        # It has the same shape as the weight matrix.
        self.dweights = np.zeros_like(self.weights)

    def forward(self, X):
        """
        Perform the forward pass.

        Parameters
        ----------
        X : numpy.ndarray
            Input batch of shape (batch_size, input_dim)

        Returns
        -------
        numpy.ndarray
            Output batch of shape (batch_size, output_dim)

        Math
        ----
        For a batch of inputs:
            Z = X @ W
        """

        # Store the input so it is available during the backward pass.
        self.input = X

        # Apply the linear transformation.
        #
        # The @ operator means matrix multiplication.
        # This is equivalent to:
        #   np.matmul(X, self.weights)
        # or, for 2D arrays:
        #   np.dot(X, self.weights)
        return X @ self.weights

    def backward(self, grad_output):
        """
        Perform the backward pass.

        Parameters
        ----------
        grad_output : numpy.ndarray
            Gradient of the loss with respect to this layer's output.
            Shape: (batch_size, output_dim)

        Returns
        -------
        numpy.ndarray
            Gradient of the loss with respect to this layer's input.
            Shape: (batch_size, input_dim)

        Math
        ----
        Let:
            Z = X @ W

        and suppose:
            grad_output = dL/dZ

        Then:

            dL/dW = X^T @ (dL/dZ)
            dL/dX = (dL/dZ) @ W^T
        """

        # Compute gradient with respect to the weights.
        #
        # Shape check:
        # self.input.T  -> (input_dim, batch_size)
        # grad_output   -> (batch_size, output_dim)
        # result        -> (input_dim, output_dim)
        #
        # This matches the shape of self.weights.
        self.dweights = self.input.T @ grad_output

        # Compute gradient with respect to the input.
        #
        # Shape check:
        # grad_output    -> (batch_size, output_dim)
        # self.weights.T -> (output_dim, input_dim)
        # result         -> (batch_size, input_dim)
        #
        # This tells the previous layer how the loss changes with respect
        # to its outputs.
        grad_input = grad_output @ self.weights.T

        return grad_input

    def get_params(self):
        """
        Return the trainable parameters of the layer.

        Returns
        -------
        dict
            Dictionary containing the current weights.
        """
        return {"weights": self.weights}

    def get_grads(self):
        """
        Return the gradients of the trainable parameters.

        Returns
        -------
        dict
            Dictionary containing the gradient of the weights.
        """
        return {"weights": self.dweights}