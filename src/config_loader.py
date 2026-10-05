import json
import os

from src.utils import ROOT_DIR


class ConfigLoader:
    """
    Utility class for loading and validating high-level experiment configs.

    This class is responsible for:
    - locating config files inside the project's configs folder
    - loading JSON files into Python dictionaries
    - validating that the config structure is correct
    - checking that important values are sensible before experiments run

    Why this is useful
    ------------------
    Validation helps catch mistakes early, such as:
    - missing keys
    - unsupported layer types
    - invalid activation names
    - impossible values like negative units or epochs
    - using the wrong loss for the task type
    """

    # Supported activation names used by src.activations.get_activation(...)
    SUPPORTED_ACTIVATIONS = {"relu", "sigmoid", "tanh", "linear"}

    # Supported loss names used by src.losses.get_loss(...)
    SUPPORTED_CLASSIFICATION_LOSSES = {
        "bce",
        "binary_crossentropy",
        "binary_cross_entropy"
    }
    SUPPORTED_REGRESSION_LOSSES = {"mse"}

    def __init__(self):
        """
        Create a ConfigLoader and point it to the configs directory.
        """
        self.configs_dir = os.path.join(ROOT_DIR, "configs")

    def load(self, filename):
        """
        Load a JSON configuration file.

        Parameters
        ----------
        filename : str
            Name of the config file, for example:
            "classification_experiments.json"

        Returns
        -------
        dict
            Parsed JSON config as a Python dictionary.

        Raises
        ------
        FileNotFoundError
            If the requested config file does not exist.
        """
        # Build the full path to the target config file
        config_path = os.path.join(self.configs_dir, filename)

        # Stop early if the file does not exist
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        # Open and parse the JSON file
        with open(config_path, "r") as f:
            config = json.load(f)

        return config

    def validate(self, config):
        """
        Validate a high-level experiment config.

        Parameters
        ----------
        config : dict
            Configuration dictionary loaded from JSON.

        Returns
        -------
        bool
            True if validation succeeds.

        Raises
        ------
        ValueError
            If any required key is missing or any value is invalid.
        """

        # ----------------------------
        # Validate top-level structure
        # ----------------------------

        required_top_level_keys = [
            "task_type",
            "input_dimension",
            "loss",
            "architectures",
            "experiments",
            "preprocessing"
        ]

        # Ensure all required top-level keys exist
        for key in required_top_level_keys:
            if key not in config:
                raise ValueError(f"Missing required config key: '{key}'")

        # task_type must be one of the two project modes
        if config["task_type"] not in ["classification", "regression"]:
            raise ValueError("task_type must be 'classification' or 'regression'")

        # input_dimension must be a positive integer
        if not isinstance(config["input_dimension"], int) or config["input_dimension"] < 1:
            raise ValueError("input_dimension must be a positive integer")

        # architectures must be a non-empty dictionary
        if not isinstance(config["architectures"], dict) or len(config["architectures"]) == 0:
            raise ValueError("'architectures' must be a non-empty dictionary")

        # ----------------------------
        # Validate preprocessing block
        # ----------------------------
        
        preprocessing = config["preprocessing"]
        
        # preprocessing must be a dictionary
        if not isinstance(preprocessing, dict):
            raise ValueError("'preprocessing' must be a dictionary")
        
        required_preprocessing_keys = [
            "enabled",
            "scale_features"
        ]
        
        # Ensure all required preprocessing keys exist
        for key in required_preprocessing_keys:
            if key not in preprocessing:
                raise ValueError(f"Missing required preprocessing key: '{key}'")
        
        # enabled must be a boolean
        if not isinstance(preprocessing["enabled"], bool):
            raise ValueError("'preprocessing.enabled' must be true or false")
        
        # scale_features must be a boolean
        if not isinstance(preprocessing["scale_features"], bool):
            raise ValueError("'preprocessing.scale_features' must be true or false")

        # ----------------------------
        # Validate loss compatibility
        # ----------------------------

        # Normalize the loss name to lowercase so that values like "MSE" still work
        loss_name = str(config["loss"]).lower()

        # Classification configs should use BCE-style losses
        if config["task_type"] == "classification":
            if loss_name not in self.SUPPORTED_CLASSIFICATION_LOSSES:
                raise ValueError(
                    "For classification, loss must be one of: "
                    "'bce', 'binary_crossentropy', 'binary_cross_entropy'"
                )

        # Regression configs should use MSE
        elif config["task_type"] == "regression":
            if loss_name not in self.SUPPORTED_REGRESSION_LOSSES:
                raise ValueError("For regression, loss must be 'mse'")

        # ----------------------------
        # Validate architectures
        # ----------------------------

        for arch_name, layers in config["architectures"].items():
            # Each architecture should be a non-empty list of layer definitions
            if not isinstance(layers, list) or len(layers) == 0:
                raise ValueError(f"Architecture '{arch_name}' must be a non-empty list")

            # Validate each layer entry
            for i, layer in enumerate(layers):
                if "type" not in layer:
                    raise ValueError(f"Architecture '{arch_name}', layer {i} is missing 'type'")
                if "units" not in layer:
                    raise ValueError(f"Architecture '{arch_name}', layer {i} is missing 'units'")
                if "activation" not in layer:
                    raise ValueError(f"Architecture '{arch_name}', layer {i} is missing 'activation'")

                # This project only supports dense layers
                if layer["type"].lower() != "dense":
                    raise ValueError(
                        f"Architecture '{arch_name}', layer {i} has unsupported type: {layer['type']}"
                    )

                # Validate that units is a positive integer
                if not isinstance(layer["units"], int) or layer["units"] < 1:
                    raise ValueError(
                        f"Architecture '{arch_name}', layer {i} must have "
                        f"'units' as a positive integer"
                    )

                # Validate that activation is supported
                activation_name = str(layer["activation"]).lower()
                if activation_name not in self.SUPPORTED_ACTIVATIONS:
                    raise ValueError(
                        f"Architecture '{arch_name}', layer {i} has unsupported activation: "
                        f"{layer['activation']}. Supported activations are: "
                        f"{sorted(self.SUPPORTED_ACTIVATIONS)}"
                    )

        # ----------------------------
        # Validate experiments block
        # ----------------------------

        experiments = config["experiments"]

        required_experiment_keys = [
            "optimizers",
            "learning_rates",
            "batch_sizes",
            "epochs",
            "early_stopping",
            "patience",
            "min_delta",
            "min_epochs_before_early_stop"
        ]

        # Ensure all required experiment keys exist
        for key in required_experiment_keys:
            if key not in experiments:
                raise ValueError(f"Missing required experiments key: '{key}'")

        # optimizers must be a non-empty list
        if not isinstance(experiments["optimizers"], list) or len(experiments["optimizers"]) == 0:
            raise ValueError("'optimizers' must be a non-empty list")

        # learning_rates must be a non-empty list
        if not isinstance(experiments["learning_rates"], list) or len(experiments["learning_rates"]) == 0:
            raise ValueError("'learning_rates' must be a non-empty list")

        # batch_sizes must be a non-empty list
        if not isinstance(experiments["batch_sizes"], list) or len(experiments["batch_sizes"]) == 0:
            raise ValueError("'batch_sizes' must be a non-empty list")

        # epochs must be a positive integer
        if not isinstance(experiments["epochs"], int) or experiments["epochs"] < 1:
            raise ValueError("'epochs' must be a positive integer")

        # early_stopping must be a boolean
        if not isinstance(experiments["early_stopping"], bool):
            raise ValueError("'early_stopping' must be true or false")

        # patience must be a positive integer
        if not isinstance(experiments["patience"], int) or experiments["patience"] < 1:
            raise ValueError("'patience' must be a positive integer")

        # min_delta must be numeric and non-negative
        if not isinstance(experiments["min_delta"], (int, float)) or experiments["min_delta"] < 0:
            raise ValueError("'min_delta' must be a non-negative number")

        # min_epochs_before_early_stop must be a non-negative integer
        min_epochs_before_early_stop = experiments["min_epochs_before_early_stop"]
        if (
            not isinstance(min_epochs_before_early_stop, int)
            or isinstance(min_epochs_before_early_stop, bool)
            or min_epochs_before_early_stop < 0
        ):
            raise ValueError("'min_epochs_before_early_stop' must be a non-negative integer")

        # The minimum-epoch guard cannot exceed the configured maximum epochs
        if min_epochs_before_early_stop > experiments["epochs"]:
            raise ValueError(
                "'min_epochs_before_early_stop' cannot be greater than 'epochs'"
            )

        # If all checks pass, the config is valid
        return True

    def load_and_validate(self, filename):
        """
        Load and validate a config file.

        Parameters
        ----------
        filename : str
            Name of the JSON config file.

        Returns
        -------
        dict
            Loaded and validated config dictionary.
        """
        # First load the config from disk
        config = self.load(filename)

        # Then validate its structure and contents
        self.validate(config)

        return config