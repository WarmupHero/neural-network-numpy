import numpy as np

# --------------------
# ReLU
# --------------------

class ReLU:
    """
    Rectified Linear Unit activation.

    Forward:
        f(x) = max(0, x)

    Backward:
        f'(x) = 1 if x > 0, else 0

    Meaning:
    - Positive inputs stay unchanged
    - Zero and negative inputs become 0
    """

    def __init__(self):
        # We store the input from the forward pass because
        # the backward pass needs to know which values were
        # positive and which were not.
        self.input = None

    def forward(self, x):
        """
        Apply ReLU elementwise.

        Parameters
        ----------
        x : numpy.ndarray
            Input array.

        Returns
        -------
        numpy.ndarray
            Output after applying ReLU to each element.
        """
        # Save input for use in backward pass
        self.input = x

        # np.maximum compares each element with 0
        # and keeps the larger value
        return np.maximum(0, x)

    def backward(self, grad_output):
        """
        Backward pass through ReLU.

        Parameters
        ----------
        grad_output : numpy.ndarray
            Gradient coming from the next layer.

        Returns
        -------
        numpy.ndarray
            Gradient with respect to the input of ReLU.

        Logic
        -----
        ReLU derivative:
        - 1 where input > 0
        - 0 where input <= 0

        So we pass the gradient through only where the
        original input was positive.
        """
        # Start with a copy of the incoming gradient
        grad_input = grad_output.copy()

        # Wherever the original input was <= 0,
        # the ReLU derivative is 0, so block the gradient
        grad_input[self.input <= 0] = 0

        return grad_input


# --------------------
# Sigmoid
# --------------------

class Sigmoid:
    """
    Sigmoid activation.

    Forward:
        f(x) = 1 / (1 + exp(-x))

    Backward:
        f'(x) = sigmoid(x) * (1 - sigmoid(x))

    Meaning:
    - Converts values into the range (0, 1)
    - Commonly used in binary classification output layers
    """

    def __init__(self):
        # Store the sigmoid output from the forward pass.
        # This is useful because the derivative can be written as:
        # sigmoid(x) * (1 - sigmoid(x))
        self.output = None

    def forward(self, x):
        """
        Apply sigmoid elementwise.

        Parameters
        ----------
        x : numpy.ndarray
            Input array.

        Returns
        -------
        numpy.ndarray
            Output after applying sigmoid.
        """
        # Clip very large or very small values to avoid numerical overflow
        # inside exp(). This does not change the intended behavior in a
        # meaningful way, but makes computation safer.
        x_clipped = np.clip(x, -500, 500)

        # Apply the sigmoid formula
        self.output = 1 / (1 + np.exp(-x_clipped))

        return self.output

    def backward(self, grad_output):
        """
        Backward pass through sigmoid.

        Parameters
        ----------
        grad_output : numpy.ndarray
            Gradient coming from the next layer.

        Returns
        -------
        numpy.ndarray
            Gradient with respect to the input of sigmoid.

        Logic
        -----
        If s = sigmoid(x), then:
            ds/dx = s * (1 - s)

        By the chain rule:
            grad_input = grad_output * ds/dx
        """
        return grad_output * self.output * (1 - self.output)


# --------------------
# Tanh
# --------------------

class Tanh:
    """
    Hyperbolic tangent activation.

    Forward:
        f(x) = tanh(x)

    Backward:
        f'(x) = 1 - tanh(x)^2

    Meaning:
    - Maps inputs into the range (-1, 1)
    - Often used in hidden layers
    """

    def __init__(self):
        # Store tanh output during forward pass.
        # This makes the derivative easy to compute later.
        self.output = None

    def forward(self, x):
        """
        Apply tanh elementwise.

        Parameters
        ----------
        x : numpy.ndarray
            Input array.

        Returns
        -------
        numpy.ndarray
            Output after applying tanh.
        """
        # NumPy applies tanh to every element in the array
        self.output = np.tanh(x)
        return self.output

    def backward(self, grad_output):
        """
        Backward pass through tanh.

        Parameters
        ----------
        grad_output : numpy.ndarray
            Gradient coming from the next layer.

        Returns
        -------
        numpy.ndarray
            Gradient with respect to the input of tanh.

        Logic
        -----
        If t = tanh(x), then:
            dt/dx = 1 - t^2

        By the chain rule:
            grad_input = grad_output * (1 - t^2)
        """
        return grad_output * (1 - self.output ** 2)


# --------------------
# Linear - Identity
# --------------------

class Linear:
    """
    Linear (identity) activation.

    Forward:
        f(x) = x

    Backward:
        f'(x) = 1

    Meaning:
    - Leaves the input unchanged
    - Commonly used in regression output layers
    """

    def __init__(self):
        # No cached values are needed for linear activation,
        # but we keep the same class structure as the others
        # for consistency.
        pass

    def forward(self, x):
        """
        Return the input unchanged.

        Parameters
        ----------
        x : numpy.ndarray
            Input array.

        Returns
        -------
        numpy.ndarray
            Exactly the same input array.
        """
        return x

    def backward(self, grad_output):
        """
        Backward pass through linear activation.

        Parameters
        ----------
        grad_output : numpy.ndarray
            Gradient coming from the next layer.

        Returns
        -------
        numpy.ndarray
            Same gradient, unchanged.

        Logic
        -----
        Since f(x) = x, its derivative is 1.
        So the gradient just passes through as-is.
        """
        return grad_output


def get_activation(name):
    """
    Return an activation object based on its name.

    Parameters
    ----------
    name : str
        Name of the activation function.

    Returns
    -------
    object
        Instance of the requested activation class.

    Supported names
    ---------------
    - "relu"
    - "sigmoid"
    - "tanh"
    - "linear"

    Raises
    ------
    ValueError
        If the activation name is not supported.
    """
    # Convert to lowercase so inputs like "ReLU" or "SIGMOID"
    # still work correctly
    name = name.lower()

    # Factory pattern:
    # choose which activation class to create based on the name
    if name == "relu":
        return ReLU()
    elif name == "sigmoid":
        return Sigmoid()
    elif name == "tanh":
        return Tanh()
    elif name == "linear":
        return Linear()
    else:
        raise ValueError(f"Unsupported activation: {name}")