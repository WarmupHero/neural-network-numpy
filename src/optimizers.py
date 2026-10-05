import numpy as np


class SGD:
    """
    Stochastic Gradient Descent (SGD) optimizer.

    Update rule:
        W = W - learning_rate * dW

    This is the simplest optimizer:
    it updates the weights by moving directly in the
    opposite direction of the gradient.

    In optimization notation:
        theta_{t+1} = theta_t - eta * g_t

    where:
    - theta = parameter being updated
    - eta = learning rate
    - g_t = gradient at the current step
    """

    def __init__(self, learning_rate=0.01):
        """
        Initialize the SGD optimizer.

        Parameters
        ----------
        learning_rate : float, default=0.01
            Step size used for each weight update.
        """
        self.learning_rate = learning_rate

    def update(self, layer):
        """
        Update the weights of a layer using SGD.

        Parameters
        ----------
        layer : object
            A trainable layer that must have:
            - layer.weights
            - layer.dweights

        Explanation
        -----------
        layer.dweights stores the gradient of the loss
        with respect to the weights.

        If the gradient is positive:
            subtracting it makes the weight smaller

        If the gradient is negative:
            subtracting it makes the weight larger

        This is how gradient descent reduces the loss.
        """
        layer.weights -= self.learning_rate * layer.dweights


class MomentumSGD:
    """
    Momentum-based Stochastic Gradient Descent optimizer.

    Update rule:
        v = beta * v + (1 - beta) * dW
        W = W - learning_rate * v

    In standard notation:
        v_t = beta * v_{t-1} + (1 - beta) * g_t
        theta_{t+1} = theta_t - eta * v_t

    Idea
    ----
    Momentum keeps a running average of past gradients.
    This helps:
    - smooth noisy updates
    - accelerate movement in a consistent direction
    - reduce zig-zagging
    """

    def __init__(self, learning_rate=0.01, beta=0.9):
        """
        Initialize the Momentum SGD optimizer.

        Parameters
        ----------
        learning_rate : float, default=0.01
            Step size used for the weight update.
        beta : float, default=0.9
            Momentum coefficient.

        Notes
        -----
        The project uses beta = 0.9.
        """
        self.learning_rate = learning_rate
        self.beta = beta

        # Store one velocity matrix per layer.
        # We use id(layer) as the key so each layer keeps its own
        # momentum state independently.
        self.velocity = {}

    def update(self, layer):
        """
        Update the weights of a layer using Momentum SGD.

        Parameters
        ----------
        layer : object
            A trainable layer that must have:
            - layer.weights
            - layer.dweights

        Explanation
        -----------
        For each layer, we maintain a velocity matrix.

        First:
            velocity = beta * old_velocity + (1 - beta) * gradient

        Then:
            weights = weights - learning_rate * velocity

        So the actual update direction is not just the current gradient,
        but a smoothed version of recent gradients.
        """
        layer_id = id(layer)

        # If this is the first time this layer is being updated,
        # create a zero velocity matrix with the same shape as its weights.
        if layer_id not in self.velocity:
            self.velocity[layer_id] = np.zeros_like(layer.weights)

        # Update the momentum term (velocity)
        self.velocity[layer_id] = (
            self.beta * self.velocity[layer_id]
            + (1 - self.beta) * layer.dweights
        )

        # Update the weights using the velocity
        layer.weights -= self.learning_rate * self.velocity[layer_id]


class AdaBelief:
    """
    AdaBelief optimizer.

    AdaBelief is similar to Adam, but instead of tracking the second
    moment of the gradient itself, it tracks the second moment of the
    "belief error" between the current gradient and its running average.

    Update equations:
        m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
        s_t = beta2 * s_{t-1} + (1 - beta2) * (g_t - m_t)^2

        m_hat = m_t / (1 - beta1^t)
        s_hat = s_t / (1 - beta2^t)

        W = W - learning_rate * m_hat / (sqrt(s_hat) + epsilon)

    Idea
    ----
    - m tracks the running average of gradients
    - s tracks how surprising the current gradient is
      compared to that running average
    - if the gradient behaves as expected, updates can be more confident
    - if the gradient is unstable, updates become more cautious
    """

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Initialize the AdaBelief optimizer.

        Parameters
        ----------
        learning_rate : float, default=0.001
            Step size used for the weight update.
        beta1 : float, default=0.9
            Exponential decay rate for the first moment.
        beta2 : float, default=0.999
            Exponential decay rate for the second moment.
        epsilon : float, default=1e-8
            Small constant for numerical stability.

        Notes
        -----
        The project uses:
        - beta1 = 0.9
        - beta2 = 0.999
        - epsilon = 1e-8
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # First moment estimate for each layer.
        # This stores the running average of gradients.
        self.m = {}

        # Second moment estimate for each layer.
        # In AdaBelief, this stores the running average of
        # squared belief errors: (gradient - first_moment)^2
        self.s = {}

        # Time step for each layer.
        # Needed for bias correction because m and s start at zero.
        self.t = {}

    def update(self, layer):
        """
        Update the weights of a layer using AdaBelief.

        Parameters
        ----------
        layer : object
            A trainable layer that must have:
            - layer.weights
            - layer.dweights

        Explanation
        -----------
        Step 1:
            Read the current gradient g

        Step 2:
            Update first moment:
                m = beta1 * m + (1 - beta1) * g

        Step 3:
            Compute belief error:
                g - m

        Step 4:
            Update second moment:
                s = beta2 * s + (1 - beta2) * (belief_error^2)

        Step 5:
            Apply bias correction to m and s

        Step 6:
            Update weights using the corrected values
        """
        layer_id = id(layer)

        # Current gradient for this layer
        g = layer.dweights

        # If this is the first time the optimizer sees this layer,
        # initialize all internal state for it.
        if layer_id not in self.m:
            self.m[layer_id] = np.zeros_like(layer.weights)
            self.s[layer_id] = np.zeros_like(layer.weights)
            self.t[layer_id] = 0

        # Increase the time step for this layer
        self.t[layer_id] += 1
        t = self.t[layer_id]

        # Update first moment estimate
        # This is the exponential moving average of gradients
        self.m[layer_id] = (
            self.beta1 * self.m[layer_id]
            + (1 - self.beta1) * g
        )

        # Belief error:
        # difference between current gradient and expected gradient
        belief_error = g - self.m[layer_id]

        # Update second moment estimate using the squared belief error
        self.s[layer_id] = (
            self.beta2 * self.s[layer_id]
            + (1 - self.beta2) * (belief_error ** 2)
        )

        # Bias correction for the first moment
        # Needed because the running average starts at zero
        m_hat = self.m[layer_id] / (1 - self.beta1 ** t)

        # Bias correction for the second moment
        s_hat = self.s[layer_id] / (1 - self.beta2 ** t)

        # Final AdaBelief update
        layer.weights -= self.learning_rate * m_hat / (np.sqrt(s_hat) + self.epsilon)


def get_optimizer(name, learning_rate):
    """
    Factory function that returns an optimizer object by name.

    Parameters
    ----------
    name : str
        Name of the optimizer.
    learning_rate : float
        Learning rate used by the optimizer.

    Returns
    -------
    object
        Instance of the requested optimizer.

    Supported names
    ---------------
    - "sgd"
    - "momentum"
    - "momentumsgd"
    - "momentum_sgd"
    - "adabelief"

    Raises
    ------
    ValueError
        If the optimizer name is not supported.

    Explanation
    -----------
    This helper lets the rest of the project choose an optimizer
    from a config file using a simple string.
    """
    # Make the name lowercase so values like "SGD" still work
    name = name.lower()

    if name == "sgd":
        return SGD(learning_rate=learning_rate)

    elif name in ["momentum", "momentumsgd", "momentum_sgd"]:
        # Fixed momentum coefficient
        return MomentumSGD(learning_rate=learning_rate, beta=0.9)

    elif name == "adabelief":
        # Fixed AdaBelief hyperparameters:
        # beta1 = 0.9
        # beta2 = 0.999
        # epsilon = 1e-8
        return AdaBelief(
            learning_rate=learning_rate,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8
        )

    else:
        raise ValueError(f"Unsupported optimizer: {name}")