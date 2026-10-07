"""
Optimizers that update layer parameters from their gradients.

Three optimizers are provided: plain SGD, SGD with momentum, and
AdaBelief. Each has update(layer), which reads layer.get_params() and
layer.get_grads() and modifies the parameter arrays in place. Stateful
optimizers keep their state per parameter, keyed by (id(layer), name).
get_optimizer() builds an optimizer from a config string.
"""

import numpy as np


class SGD:
    """
    Stochastic Gradient Descent (SGD) optimizer.

    Attributes
    ----------
    learning_rate : float
        Step size used for every update.

    Notes
    -----
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

        Returns
        -------
        None
            Sets self.learning_rate.

        Notes
        -----
        Processing:

        1. Store the learning rate. SGD keeps no other state.
        """
        self.learning_rate = learning_rate

    def update(self, layer):
        """
        Update every trainable parameter of a layer using SGD.

        Parameters
        ----------
        layer : Dense or BatchNorm
            A trainable layer exposing get_params() and get_grads(), which
            return dicts of str to numpy.ndarray with matching keys and
            shapes (float64).

        Returns
        -------
        None
            Each parameter array returned by layer.get_params() is
            modified in place, so the layer's own parameters change.

        Notes
        -----
        Processing:

        1. Read the gradients with layer.get_grads().
        2. For each parameter name and array from layer.get_params(),
           subtract learning_rate * gradient in place.

        For a Dense layer the parameters are the weights and, if enabled,
        the bias. Each gradient is the derivative of the loss with respect
        to the parameter of the same name.

        If the gradient is positive:
            subtracting it makes the parameter smaller

        If the gradient is negative:
            subtracting it makes the parameter larger

        This is how gradient descent reduces the loss.
        """
        grads = layer.get_grads()
        for name, param in layer.get_params().items():
            # In-place update, so the layer's own array is modified.
            param -= self.learning_rate * grads[name]


class MomentumSGD:
    """
    Momentum-based Stochastic Gradient Descent optimizer.

    Attributes
    ----------
    learning_rate : float
        Step size used for every update.
    beta : float
        Momentum coefficient (weight of the previous velocity).
    velocity : dict of tuple (int, str) to numpy.ndarray
        One velocity array per parameter, keyed by (id(layer), name),
        with the same shape as the parameter.

    Notes
    -----
    Update rule:
        v = beta * v + (1 - beta) * dW
        W = W - learning_rate * v

    In standard notation:
        v_t = beta * v_{t-1} + (1 - beta) * g_t
        theta_{t+1} = theta_t - eta * v_t

    Idea: momentum keeps a running average of past gradients.
    This helps:

    - smooth noisy updates
    - accelerate movement in a consistent direction
    - reduce zig-zagging

    This is the exponential-moving-average form of momentum, with the
    (1 - beta) factor on the gradient; there is no bias correction, so
    the first few steps are smaller than with plain SGD.
    """

    def __init__(self, learning_rate=0.01, beta=0.9):
        """
        Initialize the Momentum SGD optimizer.

        Parameters
        ----------
        learning_rate : float, default=0.01
            Step size used for the weight update.
        beta : float, default=0.9
            Momentum coefficient. The project uses beta = 0.9.

        Returns
        -------
        None
            Sets self.learning_rate, self.beta and an empty
            self.velocity dict.

        Notes
        -----
        Processing:

        1. Store the hyperparameters.
        2. Create an empty velocity store; velocities are created lazily
           the first time each parameter is updated.
        """
        self.learning_rate = learning_rate
        self.beta = beta

        # Store one velocity array per parameter.
        # The key is (id(layer), parameter name), so each layer's weights
        # and bias keep their own momentum state independently.
        self.velocity = {}

    def update(self, layer):
        """
        Update every trainable parameter of a layer using Momentum SGD.

        Parameters
        ----------
        layer : Dense or BatchNorm
            A trainable layer exposing get_params() and get_grads(), which
            return dicts of str to numpy.ndarray with matching keys and
            shapes (float64).

        Returns
        -------
        None
            Each parameter array returned by layer.get_params() is
            modified in place, and self.velocity is updated.

        Notes
        -----
        Processing:

        1. Read the gradients with layer.get_grads().
        2. For each parameter, build the key (id(layer), name) and create
           a zero velocity of the parameter's shape if none exists yet.
        3. Update the velocity:
           velocity = beta * old_velocity + (1 - beta) * gradient.
        4. Update the parameter in place:
           parameter = parameter - learning_rate * velocity.

        So the actual update direction is not just the current gradient,
        but a smoothed version of recent gradients.

        The key uses id(layer), so the state belongs to a specific layer
        object. Using the same optimizer for a different network would
        start fresh velocities (unless Python reuses an id).
        """
        grads = layer.get_grads()
        for name, param in layer.get_params().items():
            key = (id(layer), name)

            # If this is the first time this parameter is being updated,
            # create a zero velocity array with the same shape.
            if key not in self.velocity:
                self.velocity[key] = np.zeros_like(param)

            # Update the momentum term (velocity)
            self.velocity[key] = self.beta * self.velocity[key] + (1 - self.beta) * grads[name]

            # Update the parameter in place using the velocity
            param -= self.learning_rate * self.velocity[key]


class AdaBelief:
    """
    AdaBelief optimizer.

    AdaBelief is similar to Adam, but instead of tracking the second
    moment of the gradient itself, it tracks the second moment of the
    "belief error" between the current gradient and its running average.

    Attributes
    ----------
    learning_rate : float
        Step size used for every update.
    beta1 : float
        Decay rate of the first-moment (mean gradient) average.
    beta2 : float
        Decay rate of the second-moment (squared belief error) average.
    epsilon : float
        Constant added to the denominator for numerical stability.
    m : dict of tuple (int, str) to numpy.ndarray
        First-moment estimate per parameter, keyed by (id(layer), name).
    s : dict of tuple (int, str) to numpy.ndarray
        Second-moment estimate per parameter, same keys.
    t : dict of tuple (int, str) to int
        Number of updates applied to each parameter, used for bias
        correction.

    Notes
    -----
    Update equations:
        m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
        s_t = beta2 * s_{t-1} + (1 - beta2) * (g_t - m_t)^2

        m_hat = m_t / (1 - beta1^t)
        s_hat = s_t / (1 - beta2^t)

        W = W - learning_rate * m_hat / (sqrt(s_hat) + epsilon)

    Idea:

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

        Returns
        -------
        None
            Sets the hyperparameters and empty state dicts self.m, self.s
            and self.t.

        Notes
        -----
        Processing:

        1. Store the hyperparameters.
        2. Create empty per-parameter state dicts; the state for each
           parameter is created lazily on its first update.

        The project uses beta1 = 0.9, beta2 = 0.999 and epsilon = 1e-8
        (see get_optimizer).
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # All state below is stored per parameter, keyed by
        # (id(layer), parameter name), so weights and bias are tracked
        # separately.

        # First moment estimate for each parameter.
        # This stores the running average of gradients.
        self.m = {}

        # Second moment estimate for each parameter.
        # In AdaBelief, this stores the running average of
        # squared belief errors: (gradient - first_moment)^2
        self.s = {}

        # Time step for each parameter.
        # Needed for bias correction because m and s start at zero.
        self.t = {}

    def update(self, layer):
        """
        Update every trainable parameter of a layer using AdaBelief.

        Parameters
        ----------
        layer : Dense or BatchNorm
            A trainable layer exposing get_params() and get_grads(), which
            return dicts of str to numpy.ndarray with matching keys and
            shapes (float64).

        Returns
        -------
        None
            Each parameter array returned by layer.get_params() is
            modified in place, and self.m, self.s and self.t are updated.

        Notes
        -----
        Processing:

        These steps are applied to each parameter (e.g. weights and bias)
        separately, using the key (id(layer), name).

        1. Read the current gradient g. On the first update of this
           parameter, create zero arrays for m and s and set t = 0.
        2. Increase the step counter t by 1.
        3. Update the first moment:
           m = beta1 * m + (1 - beta1) * g
        4. Compute the belief error g - m, using the m just updated.
        5. Update the second moment:
           s = beta2 * s + (1 - beta2) * belief_error^2
        6. Apply bias correction: m_hat = m / (1 - beta1^t) and
           s_hat = s / (1 - beta2^t).
        7. Update the parameter in place:
           param -= learning_rate * m_hat / (sqrt(s_hat) + epsilon)

        The original AdaBelief paper also adds epsilon inside the s
        update; this implementation adds it only in the denominator.
        """
        grads = layer.get_grads()
        for name, param in layer.get_params().items():
            key = (id(layer), name)

            # Current gradient for this parameter
            g = grads[name]

            # If this is the first time the optimizer sees this parameter,
            # initialize all internal state for it.
            if key not in self.m:
                self.m[key] = np.zeros_like(param)
                self.s[key] = np.zeros_like(param)
                self.t[key] = 0

            # Increase the time step for this parameter
            self.t[key] += 1
            t = self.t[key]

            # Update first moment estimate
            # This is the exponential moving average of gradients
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * g

            # Belief error:
            # difference between current gradient and expected gradient
            belief_error = g - self.m[key]

            # Update second moment estimate using the squared belief error
            self.s[key] = self.beta2 * self.s[key] + (1 - self.beta2) * (belief_error**2)

            # Bias correction for the first moment
            # Needed because the running average starts at zero
            m_hat = self.m[key] / (1 - self.beta1**t)

            # Bias correction for the second moment
            s_hat = self.s[key] / (1 - self.beta2**t)

            # Final AdaBelief update, in place
            param -= self.learning_rate * m_hat / (np.sqrt(s_hat) + self.epsilon)


def get_optimizer(name, learning_rate):
    """
    Factory function that returns an optimizer object by name.

    Parameters
    ----------
    name : str
        Name of the optimizer, case-insensitive. Supported names:

        - SGD: "sgd"
        - MomentumSGD: "momentum", "momentumsgd", "momentum_sgd"
        - AdaBelief: "adabelief"
    learning_rate : float
        Learning rate used by the optimizer.

    Returns
    -------
    SGD or MomentumSGD or AdaBelief
        A new instance of the requested optimizer.

    Raises
    ------
    ValueError
        If the optimizer name is not supported.

    Notes
    -----
    Processing:

    1. Convert name to lowercase.
    2. Create the matching optimizer with the given learning rate and
       the project's fixed hyperparameters: beta = 0.9 for momentum;
       beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8 for AdaBelief.
    3. Raise ValueError for any other name.

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
        return AdaBelief(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)

    else:
        raise ValueError(f"Unsupported optimizer: {name}")
