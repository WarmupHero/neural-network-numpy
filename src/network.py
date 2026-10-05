import numpy as np

from src.layers import Dense
from src.activations import get_activation
from src.utils import RANDOM_SEED

class NeuralNetwork:
    """
    A simple feed-forward neural network built as an ordered sequence
    of dense layers and activation functions.

    This class is responsible for:
    - storing the network layers
    - running the full forward pass
    - running the full backward pass
    - returning trainable layers for optimization
    - building the architecture from a configuration

    Notes
    -----
    - The loss function is handled separately in losses.py
    - Parameter updates are handled separately in optimizers.py
    - The network owns a single random number generator so that
      all layer initializations are reproducible but not identical
    """

    def __init__(self, random_seed=RANDOM_SEED):
        """
        Create an empty neural network.

        Parameters
        ----------
        random_seed : int, default=RANDOM_SEED
            Seed used to initialize the network's random number generator.

        This gives us:
        - reproducibility across runs
        - different weight values across different layers
        """
        # Store all layers in the exact order they are added.
        # Example:
        # [Dense, ReLU, Dense, Sigmoid]
        self.layers = []

        # Create one seeded random number generator for the whole network.
        # This is passed to each Dense layer so weight initialization
        # is reproducible.
        self.random = np.random.RandomState(random_seed)

    def add_dense(self, input_dim, output_dim):
        """
        Add a dense (fully connected) layer to the network.

        Parameters
        ----------
        input_dim : int
            Number of input features to the layer.
        output_dim : int
            Number of neurons in the layer.

        Explanation
        -----------
        A Dense layer performs the linear transformation:

            Z = X @ W

        where:
        - X has shape (batch_size, input_dim)
        - W has shape (input_dim, output_dim)
        - Z has shape (batch_size, output_dim)
        """
        # Create a Dense layer and pass in the shared random generator.
        # This keeps initialization reproducible across runs.
        self.layers.append(
            Dense(input_dim, output_dim, random_state=self.random)
        )

    def add_activation(self, activation_name):
        """
        Add an activation layer to the network.

        Parameters
        ----------
        activation_name : str
            Name of the activation function.

        Supported examples
        ------------------
        - "relu"
        - "sigmoid"
        - "tanh"
        - "linear"

        Explanation
        -----------
        Activation functions are placed after Dense layers to introduce
        nonlinearity. Without them, multiple Dense layers would collapse
        into one overall linear transformation.
        """
        self.layers.append(get_activation(activation_name))

    def forward(self, X):
        """
        Run a full forward pass through the network.

        Parameters
        ----------
        X : numpy.ndarray
            Input data of shape (batch_size, input_dim)

        Returns
        -------
        numpy.ndarray
            Final network output after passing through all layers in order.

        Explanation
        -----------
        If the layers are:

            Dense -> ReLU -> Dense -> Sigmoid

        then this method computes:

            X
            -> Dense.forward(X)
            -> ReLU.forward(...)
            -> Dense.forward(...)
            -> Sigmoid.forward(...)

        This is how the network produces predictions.
        """
        # Start with the raw input data.
        output = X

        # Pass the current output through each layer in sequence.
        # Each layer's output becomes the next layer's input.
        for layer in self.layers:
            output = layer.forward(output)

        return output

    def backward(self, grad_loss):
        """
        Run a full backward pass through the network.

        Parameters
        ----------
        grad_loss : numpy.ndarray
            Gradient of the loss with respect to the network output.

        Returns
        -------
        numpy.ndarray
            Gradient with respect to the network input.
            This is returned mainly for completeness.

        Explanation
        -----------
        Backpropagation works in reverse order.

        If the forward path was:

            Dense -> ReLU -> Dense -> Sigmoid

        then the backward path is:

            Sigmoid.backward(...)
            Dense.backward(...)
            ReLU.backward(...)
            Dense.backward(...)

        Each layer receives a gradient from the layer after it,
        computes its own contribution using the chain rule,
        and passes a new gradient to the layer before it.
        """
        # Start with the gradient coming from the loss function.
        grad = grad_loss

        # Move backward through the layers in reverse order.
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

        return grad

    def get_trainable_layers(self):
        """
        Return the layers that contain trainable parameters.

        Returns
        -------
        list
            List of trainable layers.

        Explanation
        -----------
        In this project, the trainable layers are the Dense layers,
        because they have:
        - weights
        - dweights

        Activation layers do not have trainable parameters, so they
        are excluded.
        """
        trainable_layers = [
            layer for layer in self.layers
            if hasattr(layer, "weights") and hasattr(layer, "dweights")
        ]
        return trainable_layers

    def build_from_config(self, config):
        """
        Build the network automatically from a configuration dictionary.

        Parameters
        ----------
        config : dict
            Dictionary describing the network architecture.

            Expected format:
            {
                "input_dimension": 10,
                "layers": [
                    {"type": "dense", "units": 32, "activation": "relu"},
                    {"type": "dense", "units": 16, "activation": "tanh"},
                    {"type": "dense", "units": 1, "activation": "linear"}
                ]
            }

        Notes
        -----
        For each layer entry:
        - verify the layer type is "dense"
        - add a Dense layer
        - add the specified activation layer

        Explanation
        -----------
        This method lets the architecture be defined outside the code,
        usually in a config file. That makes experiments easier to run
        and compare.
        """
        # Read the input dimension of the first layer.
        current_input_dim = config["input_dimension"]

        # Read the list of layer specifications.
        layer_configs = config["layers"]

        # Build the network one layer at a time.
        for layer_config in layer_configs:
            # Read and normalize the layer type.
            layer_type = layer_config["type"].lower()

            # This project only supports Dense layers.
            if layer_type != "dense":
                raise ValueError(f"Unsupported layer type: {layer_type}")

            # Number of output units in this Dense layer.
            units = layer_config["units"]

            # Activation function to apply after this Dense layer.
            activation = layer_config["activation"]

            # Add the Dense layer.
            self.add_dense(current_input_dim, units)

            # Add the matching activation layer.
            self.add_activation(activation)

            # The output size of this layer becomes the input size
            # of the next layer.
            current_input_dim = units

    def summary(self):
        """
        Print a simple summary of the network architecture.

        Example output
        --------------
        Layer 1: Dense
        Layer 2: ReLU
        Layer 3: Dense
        Layer 4: Sigmoid

        This is only a lightweight helper for inspection.
        """
        print("\n--- Network Architecture ---")
        for i, layer in enumerate(self.layers, start=1):
            print(f"Layer {i}: {layer.__class__.__name__}")