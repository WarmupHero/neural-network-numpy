import csv
import json
import os
import time

from src.fetch_data import Fetch
from src.preprocessing import PreprocessBanknote, PreprocessEnergy
from src.config_loader import ConfigLoader
from src.network import NeuralNetwork
from src.losses import get_loss
from src.optimizers import get_optimizer
from src.train import Trainer
from src.utils import ROOT_DIR

# Toggle this to True if you want preprocessing.py to generate EDA plots.
# Leaving it False makes the full experiment sweep faster and quieter.
SHOW_EDA = False

# Mapping from problem type to the config file that defines its experiments.
CONFIG_FILES = {
    "classification": "classification_experiments.json",
    "regression": "regression_experiments.json"
}

# Folder where experiment outputs will be saved.
# This includes the summary CSV and the full JSON results.
REPORT_DIR = os.path.join(ROOT_DIR, "report")

# Create the report directory if it does not already exist.
os.makedirs(REPORT_DIR, exist_ok=True)


def build_model_config(experiment_config, architecture_name):
    """
    Build the small model-config dictionary needed by NeuralNetwork.build_from_config().

    Parameters
    ----------
    experiment_config : dict
        The validated contents of the current problem's JSON config file.
    architecture_name : str
        The architecture key chosen for the current run, such as "A1" or "A2".

    Returns
    -------
    dict
        A smaller dictionary containing only the fields required by
        NeuralNetwork.build_from_config():
        - "input_dimension"
        - "layers"
    """
    return {
        "input_dimension": experiment_config["input_dimension"],
        "layers": experiment_config["architectures"][architecture_name]
    }


def run_single_experiment(
    problem_name,
    experiment_config,
    architecture_name,
    optimizer_name,
    learning_rate,
    batch_size,
    dataset_splits
):
    """
    Run one experiment.

    Parameters
    ----------
    problem_name : str
        Either "classification" or "regression".
    experiment_config : dict
        Validated config for the current problem.
    architecture_name : str
        Name of the selected architecture from the config.
    optimizer_name : str
        Name of the optimizer to use.
    learning_rate : float
        Learning rate for the optimizer.
    batch_size : int
        Mini-batch size for training.
    dataset_splits : tuple
        (X_train, y_train, X_val, y_val, X_test, y_test)

    Returns
    -------
    dict
        Full experiment summary including settings, metrics,
        early-stopping info, and training histories.
    """
    print("\n" + "=" * 70)
    print(
        f"Running: problem={problem_name} | "
        f"architecture={architecture_name} | "
        f"optimizer={optimizer_name} | "
        f"lr={learning_rate} | "
        f"batch={batch_size}"
    )
    print("=" * 70)

    # Unpack the dataset splits for this problem
    X_train, y_train, X_val, y_val, X_test, y_test = dataset_splits

    # Safety check: the dataset feature dimension must match what the config says
    if X_train.shape[1] != experiment_config["input_dimension"]:
        raise ValueError(
            f"Config input_dimension={experiment_config['input_dimension']} "
            f"does not match dataset dimension={X_train.shape[1]}")

    # Build the specific model configuration for this architecture
    model_config = build_model_config(experiment_config, architecture_name)

    # Create the neural network and build its layer structure from config
    network = NeuralNetwork()
    network.build_from_config(model_config)

    # Create the loss function and optimizer from config choices
    loss_fn = get_loss(experiment_config["loss"])
    optimizer = get_optimizer(
        name=optimizer_name,
        learning_rate=learning_rate)

    # Create the Trainer object, including early stopping settings
    trainer = Trainer(
        network=network,
        loss_fn=loss_fn,
        optimizer=optimizer,
        task_type=experiment_config["task_type"],
        early_stopping=experiment_config["experiments"]["early_stopping"],
        patience=experiment_config["experiments"]["patience"],
        min_delta=experiment_config["experiments"]["min_delta"],
        min_epochs_before_early_stop=experiment_config["experiments"]["min_epochs_before_early_stop"]
    )

    # Train the model and collect history
    history = trainer.fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=experiment_config["experiments"]["epochs"],
        batch_size=batch_size,
        verbose=False
    )

    # Evaluate on the held-out test set
    results = trainer.evaluate(X_test, y_test)

    # Build a summary record for this run.
    # Keep the full JSON informative for later plotting/analysis,
    # even though the CSV summary is intentionally minimal.
    summary = {
        "problem_name": problem_name,
        "architecture": architecture_name,
        "optimizer": optimizer_name,
        "batch": batch_size,
        "learning_rate": learning_rate,

        "epochs": experiment_config["experiments"]["epochs"],
        "early_stopping": experiment_config["experiments"]["early_stopping"],
        "patience": experiment_config["experiments"]["patience"],
        "min_delta": experiment_config["experiments"]["min_delta"],
        "min_epochs_before_early_stop": experiment_config["experiments"]["min_epochs_before_early_stop"],
        "epochs_ran": history["epochs_ran"],
        "stopped_early": history["stopped_early"],
        "best_epoch": history["best_epoch"],
        "best_val_loss": history["best_val_loss"],

        "test_loss": float(results["test_loss"]),
        "test_metric": float(results["test_metric"]),
        "train_loss_history": [float(x) for x in history["train_loss"]],
        "val_loss_history": [float(x) for x in history["val_loss"]],
        "val_metric_history": [float(x) for x in history["val_metric"]],

        "preprocessing_enabled": experiment_config["preprocessing"]["enabled"],
        "scale_features": experiment_config["preprocessing"]["scale_features"]
    }

    return summary


def save_summary_csv(results, filename="main_summary.csv"):
    """
    Save compact experiment summary as CSV.

    Parameters
    ----------
    results : list of dict
        List of experiment summaries.
    filename : str, default="main_summary.csv"
        Output CSV filename.

    Notes
    -----
    This CSV intentionally keeps only the fields needed for a compact
    summary table.
    """
    output_path = os.path.join(REPORT_DIR, filename)

    # Minimal CSV fields requested for the experiment summary
    fieldnames = [
        "problem_name",
        "optimizer",
        "batch",
        "learning_rate",
        "architecture",
        "epochs_ran",
        "test_metric"
    ]

    # Write the CSV file
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # Keep only the selected CSV fields from each row
        for row in results:
            filtered_row = {key: row.get(key, "") for key in fieldnames}
            writer.writerow(filtered_row)

    print(f"\nSaved summary CSV to: {output_path}")


def save_full_results_json(results, filename="main_results_full.json"):
    """
    Save full experiment results as JSON.

    Parameters
    ----------
    results : list of dict
        List of experiment summaries, including history arrays.
    filename : str, default="main_results_full.json"
        Output JSON filename.

    Why this is useful
    ------------------
    The JSON preserves the full histories:
    - train_loss_history
    - val_loss_history
    - val_metric_history

    This makes it useful for later analysis and plotting.
    """
    output_path = os.path.join(REPORT_DIR, filename)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved full results JSON to: {output_path}")


def print_summary_table(results):
    """
    Print a compact summary table to the console.

    Parameters
    ----------
    results : list of dict
        List of experiment summaries.

    Notes
    -----
    The "Epochs" column here displays epochs_ran,
    meaning the actual number of epochs completed.
    """
    print("\n" + "=" * 110)
    print("FINAL EXPERIMENT SUMMARY")
    print("=" * 110)
    print(
        f"{'Problem':<15}"
        f"{'Optimizer':<15}"
        f"{'Batch':<10}"
        f"{'LR':<12}"
        f"{'Arch':<10}"
        f"{'Epochs':<10}"
        f"{'Test Metric':<15}"
    )
    print("-" * 110)

    for row in results:
        print(
            f"{row['problem_name']:<15}"
            f"{row['optimizer']:<15}"
            f"{row['batch']:<10}"
            f"{row['learning_rate']:<12}"
            f"{row['architecture']:<10}"
            f"{row['epochs_ran']:<10}"
            f"{row['test_metric']:<15.6f}"
        )


def main():
    """
    Main experiment driver.

    Workflow
    --------
    1. Download datasets if needed
    2. Load and validate experiment configs
    3. Preprocess classification and regression datasets once
    4. Run the full experiment sweep
    5. Print/save results
    """
    # Ensure all required datasets are available locally
    fetch = Fetch()
    fetch.download_all()

    # Load and validate the experiment configs first.
    # We do this before preprocessing so preprocessing behavior can be
    # controlled by values stored in the JSON configuration files.
    config_loader = ConfigLoader()
    experiment_configs = {
        problem_name: config_loader.load_and_validate(filename)
        for problem_name, filename in CONFIG_FILES.items()
    }

    # Read preprocessing settings for each problem
    classification_preprocessing = experiment_configs["classification"]["preprocessing"]
    regression_preprocessing = experiment_configs["regression"]["preprocessing"]

    # Preprocess each dataset only once.
    # This is good because all experiments on the same problem then use
    # the exact same train/validation/test split.
    print("\nPreprocessing classification dataset...")
    classification_data = PreprocessBanknote().get_data(
        show_eda=SHOW_EDA,
        preprocessing_enabled=classification_preprocessing["enabled"],
        scale_features=classification_preprocessing["scale_features"]
    )

    print("\nPreprocessing regression dataset...")
    regression_data = PreprocessEnergy().get_data(
        show_eda=SHOW_EDA,
        preprocessing_enabled=regression_preprocessing["enabled"],
        scale_features=regression_preprocessing["scale_features"]
    )

    # Store the ready-to-use dataset splits by problem name
    datasets = {
        "classification": classification_data,
        "regression": regression_data
    }

    # This will hold one summary dictionary per run
    all_results = []

    # Count total number of runs for progress reporting
    total_runs = 0
    for problem_name, config in experiment_configs.items():
        total_runs += (
            len(config["architectures"])
            * len(config["experiments"]["optimizers"])
            * len(config["experiments"]["learning_rates"])
            * len(config["experiments"]["batch_sizes"])
        )

    run_counter = 0

    # Full nested sweep over:
    # - problem
    # - architecture
    # - optimizer
    # - learning rate
    # - batch size
    for problem_name, config in experiment_configs.items():
        for architecture_name in config["architectures"].keys():
            for optimizer_name in config["experiments"]["optimizers"]:
                for learning_rate in config["experiments"]["learning_rates"]:
                    for batch_size in config["experiments"]["batch_sizes"]:
                        run_counter += 1
                        print(f"\nStarting run {run_counter}/{total_runs}...")

                        result = run_single_experiment(
                            problem_name=problem_name,
                            experiment_config=config,
                            architecture_name=architecture_name,
                            optimizer_name=optimizer_name,
                            learning_rate=learning_rate,
                            batch_size=batch_size,
                            dataset_splits=datasets[problem_name])

                        all_results.append(result)

    # Print a compact console summary
    print_summary_table(all_results)

    # Save the compact CSV summary
    save_summary_csv(all_results)

    # Save the full JSON results including histories
    save_full_results_json(all_results)


if __name__ == "__main__":
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()

    elapsed = end_time - start_time
    print(f"\nTotal execution time: {elapsed:.2f} seconds")