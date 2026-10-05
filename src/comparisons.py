import json
import os
import time

# Standard Matplotlib plotting interface.
import matplotlib.pyplot as plt

# ROOT_DIR points to the root folder of the project/repository.
# We use it to build paths to the results file and the output folder.
from src.utils import ROOT_DIR

# Path to the full results JSON produced by the main experiment runner.
# This file contains all train/validation loss histories needed for plotting.
RESULTS_PATH = os.path.join(ROOT_DIR, "report", "main_results_full.json")

# Folder where all comparison plots and text analyses will be saved.
PLOTS_DIR = os.path.join(ROOT_DIR, "report", "comparisons")

# Create the output folder if it does not already exist.
os.makedirs(PLOTS_DIR, exist_ok=True)

def load_results(path):
    """
    Load the full experiment results JSON.

    Parameters
    ----------
    path : str
        Path to the JSON results file.

    Returns
    -------
    list of dict
        One dictionary per experiment run.
    """
    # Open the JSON file in read mode.
    with open(path, "r") as f:
        # Parse the JSON contents into Python objects and return them.
        return json.load(f)


def _build_epoch_ticks(runs):
    """
    Build x-axis tick marks for one comparison figure.

    ----------------------
    Different runs may end at different epochs because of early stopping.
    We want:
    - readable tick spacing
    - explicit inclusion of each run's final epoch
    """
    # Compute the final epoch of each run from the length of its train-loss history.
    end_epochs = [len(run["train_loss_history"]) for run in runs]

    # Find the longest run in this figure.
    max_epoch = max(end_epochs)

    # Choose a readable x-axis tick spacing based on the longest run.
    # Short runs get denser ticks; longer runs get more spread-out ticks.
    if max_epoch <= 20:
        step = 1
    elif max_epoch <= 50:
        step = 5
    else:
        step = 10

    # Create regularly spaced epoch ticks.
    regular_ticks = list(range(1, max_epoch + 1, step))

    # Ensure the maximum epoch is explicitly included.
    if regular_ticks[-1] != max_epoch:
        regular_ticks.append(max_epoch)

    # Also include the final epoch of every run, so early-stopped runs
    # have their stopping point clearly visible on the x-axis.
    ticks = sorted(set(regular_ticks + end_epochs))

    # Return both the tick list and the maximum epoch used for x-axis limits.
    return ticks, max_epoch


def _compute_convergence_metrics(run, relative_tolerance=0.01, absolute_floor=1e-4):
    """
    Compute simple convergence metrics for one run.

    ---------------------
    Convergence rule used
    ---------------------
    1. Take the final training loss:
       final_train_loss = last value in train_loss_history

    2. Define a tolerance around that final value:
       tolerance = max(absolute_floor, relative_tolerance * abs(final_train_loss))
       absolute_floor is the minimum allowed tolerance, used so the convergence
       rule does not become unrealistically strict when the final training loss
       is very small.

    3. Find the first epoch after which ALL remaining training-loss values
       stay within that tolerance of the final training loss.
    """

    # Extract the training-loss history for this run.
    train_history = run["train_loss_history"]

    # Extract the validation-loss history for this run.
    val_history = run["val_loss_history"]

    # Final training loss = last recorded training-loss value.
    final_train_loss = float(train_history[-1])

    # Final validation loss = last recorded validation-loss value.
    final_val_loss = float(val_history[-1])

    # Number of epochs actually run (may be less than max epochs due to early stopping).
    epochs_ran = len(train_history)

    # Define the tolerance used for the convergence test.
    # Usually this is 1% of the final training loss, but we never allow it
    # to become smaller than the absolute floor.
    tolerance = max(absolute_floor, relative_tolerance * abs(final_train_loss))

    # Default convergence epoch to the last epoch, in case the run only settles
    # right at the end.
    convergence_epoch = epochs_ran

    # Scan from the beginning of the run.
    # We are looking for the first epoch after which the rest of the training-loss
    # values stay close to the final training loss.
    for start_idx in range(epochs_ran):
        # Take the "tail" of the training-loss curve from this epoch to the end.
        tail = train_history[start_idx:]

        # Check whether all values in that remaining tail are within the allowed
        # tolerance of the final training loss.
        if all(abs(loss - final_train_loss) <= tolerance for loss in tail):
            # Convert the 0-based list index to a 1-based epoch number.
            convergence_epoch = start_idx + 1
            # Stop as soon as we find the first such epoch.
            break

    # Return all computed metrics in one dictionary so they can be reused
    # in plots and written analyses.
    return {
        "epochs_ran": epochs_ran,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "tolerance": tolerance,
        "convergence_epoch": convergence_epoch,
    }


def _format_metrics_text(run):
    """
    Format a small block of convergence-related metrics for display inside a plot.
    """
    # Compute convergence metrics for this run.
    metrics = _compute_convergence_metrics(run)

    # Build a small multi-line string that will be shown inside the figure.
    return (
        f"Final train loss: {metrics['final_train_loss']:.6f}\n"
        f"Final val loss:   {metrics['final_val_loss']:.6f}\n"
        f"Epochs ran:       {metrics['epochs_ran']}\n"
        f"Convergence epoch:{metrics['convergence_epoch']}"
    )


def _plot_loss_curves(ax, run, subplot_title, include_metrics_box=False):
    """
    Plot one run's train-loss and validation-loss curves on a given subplot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Subplot axis to draw on.
    run : dict
        One experiment record from the JSON results.
    subplot_title : str
        Title shown above the subplot.
    include_metrics_box : bool, default=False
        Whether to show a small textbox with convergence metrics.
        We enable this for the depth and learning-rate comparisons, where the
        written analysis discusses convergence.
    """

    # Build the epoch numbers for the x-axis: 1, 2, 3, ..., final epoch.
    epochs = range(1, len(run["train_loss_history"]) + 1)

    # The final epoch is just the length of the stored training-loss history.
    end_epoch = len(run["train_loss_history"])

    # Plot the training-loss curve as a solid line.
    ax.plot(epochs, run["train_loss_history"], label="Train Loss")

    # Plot the validation-loss curve as a dashed line so it is visually distinct.
    ax.plot(epochs, run["val_loss_history"], linestyle="--", label="Validation Loss")

    # Draw a vertical dotted line at the final epoch.
    # This makes early stopping visible on the plot.
    ax.axvline(
        x=end_epoch,
        linestyle=":",
        linewidth=1.5,
        alpha=0.8,
        label=f"Ended at epoch {end_epoch}",
    )

    # Add the subplot title, including the end epoch for readability.
    ax.set_title(f"{subplot_title} | End Epoch = {end_epoch}", fontsize=11)

    # Label the y-axis.
    ax.set_ylabel("Loss")

    # Add a light background grid to make the curves easier to read.
    ax.grid(True, alpha=0.3)

    # Show the legend for train loss, validation loss, and final-epoch line.
    ax.legend()

    # If requested, place a small metrics summary box inside this subplot.
    # This is especially useful for the depth and learning-rate comparisons.
    if include_metrics_box:
        # Build the metrics text.
        metrics_text = _format_metrics_text(run)

        # Place the text box in the upper-right corner of the subplot.
        ax.text(
            0.98,
            0.98,
            metrics_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )


def _apply_epoch_ticks(axes, runs):
    """
    Apply consistent x-axis limits and ticks across all subplots in one figure.
    """
    # Build the tick marks and determine the largest epoch shown in this figure.
    ticks, max_epoch = _build_epoch_ticks(runs)

    # Apply the same x-axis configuration to every subplot in the figure.
    for ax in axes:
        # Force all subplots to share the same epoch range.
        ax.set_xlim(1, max_epoch)

        # Apply the chosen tick positions.
        ax.set_xticks(ticks)

        # Rotate tick labels slightly for readability.
        ax.tick_params(axis="x", rotation=45)


def _save_and_show(fig, filename):
    """
    Save a figure to disk, print the location, then show it interactively.
    """
    # Build the full output path for the figure file.
    output_path = os.path.join(PLOTS_DIR, filename)

    # Save the figure with high resolution and tight layout.
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    # Print the file path so the user can see where it was saved.
    print(f"Saved plot to: {output_path}")

    # Show the plot in a window when running from CMD.
    plt.show(block=True)

    # Close the figure afterward to free memory.
    plt.close(fig)


def _write_text_file(filename, content):
    """
    Save a short text analysis file to the plots folder.
    """
    # Build the full output path for the text file.
    output_path = os.path.join(PLOTS_DIR, filename)

    # Open the file for writing using UTF-8 encoding.
    with open(output_path, "w", encoding="utf-8") as f:
        # Write the text content to disk.
        f.write(content)

    # Print the file path so the user knows where the analysis was saved.
    print(f"Saved analysis text to: {output_path}")


def _depth_label(architecture_name):
    """
    Convert architecture code names into human-readable depth labels.
    """
    # Map architecture A1 to a more descriptive label.
    if architecture_name == "A1":
        return "A1 (1 hidden layer)"

    # Map architecture A2 to a more descriptive label.
    if architecture_name == "A2":
        return "A2 (3 hidden layers)"

    # Fallback: return the original architecture name if it is neither A1 nor A2.
    return architecture_name


def _build_depth_analysis_text(selected_runs, problem_name, optimizer, learning_rate, batch_size):
    """
    We expect exactly two runs:
    - A1
    - A2

    The text explains:
    - which architecture converged faster
    - which ended with lower final training loss
    - which ended with lower final validation loss
    """
    # Unpack the two matched runs.
    run_a1, run_a2 = selected_runs

    # Compute convergence metrics for A1.
    metrics_a1 = _compute_convergence_metrics(run_a1)

    # Compute convergence metrics for A2.
    metrics_a2 = _compute_convergence_metrics(run_a2)

    # Build human-readable labels for the two architectures.
    label_a1 = _depth_label(run_a1["architecture"])
    label_a2 = _depth_label(run_a2["architecture"])

    # Determine which architecture converges faster.
    if metrics_a1["convergence_epoch"] < metrics_a2["convergence_epoch"]:
        faster_sentence = (
            f"{label_a1} converges faster by this criterion, "
            f"because it reaches and stays near its final training loss earlier."
        )
    elif metrics_a2["convergence_epoch"] < metrics_a1["convergence_epoch"]:
        faster_sentence = (
            f"{label_a2} converges faster by this criterion, "
            f"because it reaches and stays near its final training loss earlier."
        )
    else:
        faster_sentence = "Both architectures reach their final training-loss region at the same epoch."

    # Compare final training losses.
    if metrics_a1["final_train_loss"] < metrics_a2["final_train_loss"]:
        train_sentence = (
            f"{label_a1} ends with the lower final training loss, "
            f"so it achieves the better final training fit in this matched experiment."
        )
    elif metrics_a2["final_train_loss"] < metrics_a1["final_train_loss"]:
        train_sentence = (
            f"{label_a2} ends with the lower final training loss, "
            f"so it achieves the better final training fit in this matched experiment."
        )
    else:
        train_sentence = "Both architectures end with the same final training loss."

    # Compare final validation losses.
    if metrics_a1["final_val_loss"] < metrics_a2["final_val_loss"]:
        val_sentence = (
            f"{label_a1} ends with the lower final validation loss, "
            f"which suggests better validation performance in this matched experiment."
        )
    elif metrics_a2["final_val_loss"] < metrics_a1["final_val_loss"]:
        val_sentence = (
            f"{label_a2} ends with the lower final validation loss, "
            f"which suggests better validation performance in this matched experiment."
        )
    else:
        val_sentence = "Both architectures end with the same final validation loss."

    # Build and return the final text block that will be saved as a .txt file.
    return (
        "Network Depth Experiment\n"
        "========================\n\n"
        "Selected matched experiment\n"
        "---------------------------\n"
        f"Problem: {problem_name}\n"
        f"Optimizer: {optimizer}\n"
        f"Learning rate: {learning_rate}\n"
        f"Batch size: {batch_size}\n"
        "Only the number of hidden layers changes: A1 vs A2.\n\n"
        "---------------------\n"
        "For each run, convergence epoch is the first epoch after which all remaining\n"
        "training-loss values stay within a small tolerance of the final training loss.\n\n"
        "Measured values\n"
        "---------------\n"
        f"{label_a1}:\n"
        f"  - epochs ran: {metrics_a1['epochs_ran']}\n"
        f"  - final train loss: {metrics_a1['final_train_loss']:.6f}\n"
        f"  - final val loss: {metrics_a1['final_val_loss']:.6f}\n"
        f"  - convergence epoch: {metrics_a1['convergence_epoch']}\n\n"
        f"{label_a2}:\n"
        f"  - epochs ran: {metrics_a2['epochs_ran']}\n"
        f"  - final train loss: {metrics_a2['final_train_loss']:.6f}\n"
        f"  - final val loss: {metrics_a2['final_val_loss']:.6f}\n"
        f"  - convergence epoch: {metrics_a2['convergence_epoch']}\n\n"
        "Interpretation\n"
        "--------------\n"
        f"{faster_sentence}\n"
        f"{train_sentence}\n"
        f"{val_sentence}\n"
    )


def _build_learning_rate_analysis_text(selected_runs, problem_name, architecture, optimizer, batch_size):
    """
    We expect exactly two runs:
    - learning rate 0.1
    - learning rate 0.001
    """
    # Unpack the two matched learning-rate runs.
    run_lr_high, run_lr_low = selected_runs

    # Compute convergence metrics for the first learning-rate run.
    metrics_high = _compute_convergence_metrics(run_lr_high)

    # Compute convergence metrics for the second learning-rate run.
    metrics_low = _compute_convergence_metrics(run_lr_low)

    # Store the actual learning-rate values for labeling.
    lr_high = run_lr_high["learning_rate"]
    lr_low = run_lr_low["learning_rate"]

    # Build human-readable labels for the two learning rates.
    high_label = f"LR = {lr_high}"
    low_label = f"LR = {lr_low}"

    # Determine which learning rate converges faster.
    if metrics_high["convergence_epoch"] < metrics_low["convergence_epoch"]:
        faster_sentence = (
            f"{high_label} converges faster by this criterion, "
            f"because it reaches and stays near its final training loss earlier."
        )
    elif metrics_low["convergence_epoch"] < metrics_high["convergence_epoch"]:
        faster_sentence = (
            f"{low_label} converges faster by this criterion, "
            f"because it reaches and stays near its final training loss earlier."
        )
    else:
        faster_sentence = "Both learning rates reach their final training-loss region at the same epoch."

    # Compare final training losses.
    if metrics_high["final_train_loss"] < metrics_low["final_train_loss"]:
        train_sentence = (
            f"{high_label} ends with the lower final training loss "
            f"in this matched experiment."
        )
    elif metrics_low["final_train_loss"] < metrics_high["final_train_loss"]:
        train_sentence = (
            f"{low_label} ends with the lower final training loss "
            f"in this matched experiment."
        )
    else:
        train_sentence = "Both learning rates end with the same final training loss."

    # Compare final validation losses.
    if metrics_high["final_val_loss"] < metrics_low["final_val_loss"]:
        val_sentence = (
            f"{high_label} ends with the lower final validation loss "
            f"in this matched experiment."
        )
    elif metrics_low["final_val_loss"] < metrics_high["final_val_loss"]:
        val_sentence = (
            f"{low_label} ends with the lower final validation loss "
            f"in this matched experiment."
        )
    else:
        val_sentence = "Both learning rates end with the same final validation loss."

    # Build and return the final text block that will be saved as a .txt file.
    return (
        "Learning Rate Sensitivity\n"
        "=========================\n\n"
        "Selected matched experiment\n"
        "---------------------------\n"
        f"Problem: {problem_name}\n"
        f"Architecture: {architecture}\n"
        f"Optimizer: {optimizer}\n"
        f"Batch size: {batch_size}\n"
        "Only the learning rate changes: 0.1 vs 0.001.\n\n"
        "Convergence rule used\n"
        "---------------------\n"
        "Convergence is defined using TRAINING loss.\n"
        "For each run, convergence epoch is the first epoch after which all remaining\n"
        "training-loss values stay within a small tolerance of the final training loss.\n\n"
        "Measured values\n"
        "---------------\n"
        f"{high_label}:\n"
        f"  - epochs ran: {metrics_high['epochs_ran']}\n"
        f"  - final train loss: {metrics_high['final_train_loss']:.6f}\n"
        f"  - final val loss: {metrics_high['final_val_loss']:.6f}\n"
        f"  - convergence epoch: {metrics_high['convergence_epoch']}\n\n"
        f"{low_label}:\n"
        f"  - epochs ran: {metrics_low['epochs_ran']}\n"
        f"  - final train loss: {metrics_low['final_train_loss']:.6f}\n"
        f"  - final val loss: {metrics_low['final_val_loss']:.6f}\n"
        f"  - convergence epoch: {metrics_low['convergence_epoch']}\n\n"
        "Interpretation\n"
        "--------------\n"
        f"{faster_sentence}\n"
        f"{train_sentence}\n"
        f"{val_sentence}\n"
    )


# --------------------------------------------------
# 1. Optimizer comparison
# --------------------------------------------------
def filter_optimizer_runs(results, problem_name, architecture, learning_rate, batch_size):
    """
    Select the three optimizer runs for one matched experiment.

    What stays fixed
    ----------------
    - problem_name
    - architecture
    - learning_rate
    - batch_size

    What changes
    ------------
    - optimizer only
    """
    # Keep only runs that match the fixed parameters for this comparison.
    filtered = [
        run for run in results
        if run["problem_name"] == problem_name
        and run["architecture"] == architecture
        and run["learning_rate"] == learning_rate
        and run["batch"] == batch_size
    ]

    # Sort optimizers into a consistent top-to-bottom order for plotting.
    optimizer_order = {"sgd": 0, "momentum": 1, "adabelief": 2}
    filtered.sort(key=lambda x: optimizer_order[x["optimizer"]])

    # Return the three matched optimizer runs.
    return filtered


def plot_optimizer_comparison(results, problem_name, architecture, learning_rate, batch_size, filename):
    """
    Create one figure comparing SGD, Momentum, and AdaBelief
    under one matched parameter setting.
    """
    # Select the three runs for this optimizer comparison.
    selected_runs = filter_optimizer_runs(
        results=results,
        problem_name=problem_name,
        architecture=architecture,
        learning_rate=learning_rate,
        batch_size=batch_size
    )

    # Sanity check: the optimizer comparison expects exactly 3 optimizer runs.
    if len(selected_runs) != 3:
        raise ValueError(
            f"Expected 3 optimizer runs, found {len(selected_runs)} "
            f"for {problem_name}, {architecture}, lr={learning_rate}, batch={batch_size}"
        )

    # Create a figure with 3 vertical subplots, one per optimizer.
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    # Plot each optimizer run on its own subplot.
    for ax, run in zip(axes, selected_runs):
        optimizer_name = run["optimizer"].upper()
        subplot_title = (
            f"{optimizer_name} Optimizer | "
            f"{problem_name.capitalize()} | {architecture} | "
            f"LR={learning_rate} | Batch={batch_size}"
        )
        _plot_loss_curves(ax, run, subplot_title, include_metrics_box=True)

    # Apply consistent epoch ticks across all subplots.
    _apply_epoch_ticks(axes, selected_runs)

    # Label the x-axis only on the last subplot.
    axes[-1].set_xlabel("Epoch")

    # Add an overall title for the full figure.
    fig.suptitle(
        f"Optimizer Comparison: {problem_name.capitalize()} Problem | "
        f"Architecture {architecture} | LR={learning_rate} | Batch Size={batch_size}",
        fontsize=14,
        fontweight="bold"
    )

    # Adjust spacing so the suptitle and subplots fit cleanly.
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Save and display the figure.
    _save_and_show(fig, filename)


# --------------------------------------------------
# 2. Depth comparison (A1 vs A2)
# --------------------------------------------------
def filter_depth_runs(results, problem_name, optimizer, learning_rate, batch_size):
    """
    Select the two runs needed for the depth experiment.

    What stays fixed
    ----------------
    - problem_name
    - optimizer
    - learning_rate
    - batch_size

    What changes
    ------------
    - architecture only (A1 vs A2)
    """
    # Keep only runs that match the fixed settings for the depth comparison.
    filtered = [
        run for run in results
        if run["problem_name"] == problem_name
        and run["optimizer"] == optimizer
        and run["learning_rate"] == learning_rate
        and run["batch"] == batch_size
        and run["architecture"] in ["A1", "A2"]
    ]

    # Sort A1 before A2 so the order is stable and easy to read.
    architecture_order = {"A1": 0, "A2": 1}
    filtered.sort(key=lambda x: architecture_order[x["architecture"]])

    # Return the two matched architecture runs.
    return filtered


def plot_depth_comparison(results, problem_name, optimizer, learning_rate, batch_size, filename, analysis_filename):
    """
    Output
    ------
    1. A figure comparing A1 and A2 with:
       - training loss
       - validation loss
       - convergence metrics box

    2. A small text file summarizing:
       - which architecture converged faster
       - which achieved lower final training loss
       - which achieved lower final validation loss
    """
    # Select the two runs for the depth comparison.
    selected_runs = filter_depth_runs(
        results=results,
        problem_name=problem_name,
        optimizer=optimizer,
        learning_rate=learning_rate,
        batch_size=batch_size
    )

    # Sanity check: the depth comparison expects exactly 2 architectures.
    if len(selected_runs) != 2:
        raise ValueError(
            f"Expected 2 architecture runs, found {len(selected_runs)} "
            f"for {problem_name}, optimizer={optimizer}, lr={learning_rate}, batch={batch_size}"
        )

    # Create a figure with 2 vertical subplots: one for A1 and one for A2.
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    # Plot each architecture run on its own subplot.
    for ax, run in zip(axes, selected_runs):
        architecture = run["architecture"]
        subplot_title = (
            f"Architecture {architecture} | "
            f"{problem_name.capitalize()} | {optimizer.upper()} | "
            f"LR={learning_rate} | Batch={batch_size}"
        )
        _plot_loss_curves(ax, run, subplot_title, include_metrics_box=True)

    # Apply consistent epoch ticks across both subplots.
    _apply_epoch_ticks(axes, selected_runs)

    # Label the shared x-axis.
    axes[-1].set_xlabel("Epoch")

    # Add an overall figure title.
    fig.suptitle(
        f"Network Depth Comparison: A1 vs A2 | "
        f"{problem_name.capitalize()} Problem | Optimizer={optimizer.upper()} | "
        f"LR={learning_rate} | Batch Size={batch_size}",
        fontsize=14,
        fontweight="bold"
    )

    # Adjust spacing so labels and title fit properly.
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save and display the figure.
    _save_and_show(fig, filename)

    # Build the written analysis text for network depth analysis.
    analysis_text = _build_depth_analysis_text(
        selected_runs=selected_runs,
        problem_name=problem_name,
        optimizer=optimizer,
        learning_rate=learning_rate,
        batch_size=batch_size
    )

    # Save the written analysis to a text file.
    _write_text_file(analysis_filename, analysis_text)


# --------------------------------------------------
# 3. Learning-rate comparison (0.1 vs 0.001)
# --------------------------------------------------
def filter_learning_rate_runs(results, problem_name, architecture, optimizer, batch_size):
    """
    Select the two runs needed for the learning-rate sensitivity experiment.

    What stays fixed
    ----------------
    - problem_name
    - architecture
    - optimizer
    - batch_size

    What changes
    ------------
    - learning_rate only (0.1 vs 0.001)
    """
    # Keep only runs that match the fixed settings for the learning-rate comparison.
    filtered = [
        run for run in results
        if run["problem_name"] == problem_name
        and run["architecture"] == architecture
        and run["optimizer"] == optimizer
        and run["batch"] == batch_size
        and run["learning_rate"] in [0.1, 0.001]
    ]

    # Sort the two learning rates into a stable order for plotting.
    lr_order = {0.1: 0, 0.001: 1}
    filtered.sort(key=lambda x: lr_order[x["learning_rate"]])

    # Return the two matched learning-rate runs.
    return filtered


def plot_learning_rate_comparison(results, problem_name, architecture, optimizer, batch_size, filename, analysis_filename):
    """
    Create the learning-rate comparison figure and write a short analysis text file.

    Output
    ------
    1. A figure comparing learning rates 0.1 and 0.001 with:
       - training loss
       - validation loss
       - convergence metrics box

    2. A small text file summarizing:
       - which learning rate converged faster
       - which achieved lower final training loss
       - which achieved lower final validation loss
    """
    # Select the two runs for the learning-rate comparison.
    selected_runs = filter_learning_rate_runs(
        results=results,
        problem_name=problem_name,
        architecture=architecture,
        optimizer=optimizer,
        batch_size=batch_size
    )

    # Sanity check: We expect exactly 2 learning-rate runs.
    if len(selected_runs) != 2:
        raise ValueError(
            f"Expected 2 learning-rate runs, found {len(selected_runs)} "
            f"for {problem_name}, {architecture}, optimizer={optimizer}, batch={batch_size}"
        )

    # Create a figure with 2 vertical subplots: one for each learning rate.
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    # Plot each learning-rate run on its own subplot.
    for ax, run in zip(axes, selected_runs):
        lr = run["learning_rate"]
        subplot_title = (
            f"Learning Rate = {lr} | "
            f"{problem_name.capitalize()} | {architecture} | "
            f"{optimizer.upper()} | Batch={batch_size}"
        )

        _plot_loss_curves(ax, run, subplot_title, include_metrics_box=True)

    # Apply consistent epoch ticks across both subplots.
    _apply_epoch_ticks(axes, selected_runs)

    # Label the shared x-axis.
    axes[-1].set_xlabel("Epoch")

    # Add an overall figure title.
    fig.suptitle(
        f"Learning-Rate Sensitivity Comparison: LR 0.1 vs 0.001 | "
        f"{problem_name.capitalize()} Problem | Architecture {architecture} | "
        f"Optimizer={optimizer.upper()} | Batch Size={batch_size}",
        fontsize=14,
        fontweight="bold"
    )

    # Adjust spacing so labels and title fit properly.
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save and display the figure.
    _save_and_show(fig, filename)

    # Build the written analysis text for learning rate comparison.
    analysis_text = _build_learning_rate_analysis_text(
        selected_runs=selected_runs,
        problem_name=problem_name,
        architecture=architecture,
        optimizer=optimizer,
        batch_size=batch_size
    )

    # Save the written analysis to a text file.
    _write_text_file(analysis_filename, analysis_text)


def main():
    """
    Generate all required plots and short text analyses.

   -------------------------
    1. Optimizer comparison:
       - provide at least 3 different plots
       - same parameters inside each plot, optimizer varies

    2. Network depth experiment:
       - compare A1 vs A2 for one matched experiment
       - also write a short text analysis

    3. Learning-rate sensitivity:
       - compare LR 0.1 vs 0.001 for one matched experiment
       - also write a short text analysis
    """
    # Load the full experiment results from disk.
    results = load_results(RESULTS_PATH)

    # Randomly selected runs. To adjust, change the arguments of the plot to any acceptable value. 
    # Below the acceptable values as up to the latest repo version:
    # - problem_name: classification or regression
    # - architecture: A1 or A2
    # - learning_rate: 0.1 or 0.001
    # - batch_size: 16 or 64
    # - optimizer: sgd, momentum, or adabelief
    # --------------------------------------------------
    # Optimizer comparison
    # --------------------------------------------------

    # Plot 1:
    # Compare optimizers for the classification problem using architecture A1,
    # learning rate 0.1, and batch size 16.
    plot_optimizer_comparison(
        results=results,
        problem_name="classification",
        architecture="A1",
        learning_rate=0.1,
        batch_size=16,
        filename="optimizers_classification_A1_lr01_bs16.png"
    )

    # Plot 2:
    # Compare optimizers for the classification problem using architecture A2,
    # learning rate 0.1, and batch size 64.
    plot_optimizer_comparison(
        results=results,
        problem_name="classification",
        architecture="A2",
        learning_rate=0.1,
        batch_size=64,
        filename="optimizers_classification_A2_lr01_bs64.png"
    )

    # Plot 3:
    # Compare optimizers for the regression problem using architecture A2,
    # learning rate 0.001, and batch size 64.
    plot_optimizer_comparison(
        results=results,
        problem_name="regression",
        architecture="A2",
        learning_rate=0.001,
        batch_size=64,
        filename="optimizers_regression_A2_lr0001_bs64.png"
    )

    # --------------------------------------------------
    # Network depth experiment
    # --------------------------------------------------

    # Here we keep:
    # - problem fixed
    # - optimizer fixed
    # - learning rate fixed
    # - batch size fixed
    #
    # And vary only:
    # - architecture (A1 vs A2)
    #
    # This produces:
    # - one depth-comparison plot
    # - one short text analysis file
    plot_depth_comparison(
        results=results,
        problem_name="classification",
        optimizer="sgd",
        learning_rate=0.1,
        batch_size=16,
        filename="depth_classification_sgd_lr01_bs16.png",
        analysis_filename="depth_analysis.txt"
    )

    # --------------------------------------------------
    # Learning-rate sensitivity
    # --------------------------------------------------

    # Here we keep:
    # - problem fixed
    # - architecture fixed
    # - optimizer fixed
    # - batch size fixed
    #
    # And vary only:
    # - learning rate (0.1 vs 0.001)
    #
    # This produces:
    # - one learning-rate comparison plot
    # - one short text analysis file
    plot_learning_rate_comparison(
        results=results,
        problem_name="classification",
        architecture="A1",
        optimizer="sgd",
        batch_size=16,
        filename="learning_rate_classification_A1_sgd_bs16.png",
        analysis_filename="learning_rate_analysis.txt")

if __name__ == "__main__":
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()

    elapsed = end_time - start_time
    print(f"\nTotal execution time: {elapsed:.2f} seconds")