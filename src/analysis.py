import json
import os
from statistics import mean
import time

from src.utils import ROOT_DIR

# Load JSON that contains all recorded training/validation histories
# for every experiment run in full.
RESULTS_PATH = os.path.join(ROOT_DIR, "report", "main_results_full.json")

# Path where this script will write the analysis report.
OUTPUT_PATH = os.path.join(ROOT_DIR, "report", "analysis.txt")

# Preferred display order for optimizer summaries in tables.
OPTIMIZER_ORDER = ["sgd", "momentum", "adabelief"]

# Preferred display order for architecture summaries in tables.
ARCHITECTURE_ORDER = ["A1", "A2"]


def load_results(path):
    """
    Load the full experiment results JSON from disk.

    Parameters
    ----------
    path : str
        Path to the JSON file.

    Returns
    -------
    list of dict
        One dictionary per experiment run.
    """
    # Open the JSON file and parse it into Python objects.
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def convergence_epoch(train_loss_history, relative_tolerance=0.01, absolute_floor=1e-4):
    """
    Define convergence using TRAINING loss, not validation loss.

    Convergence epoch = first epoch after which all remaining training-loss
    values stay within a small tolerance of the final training loss.

    Tolerance used:
        max(absolute_floor, relative_tolerance * abs(final_train_loss))

    ---------------------------
    Exercise asks which optimizer converges fastest on average.
    To match that wording, we define convergence from the training-loss curve,
    not from early stopping or validation loss.

    Parameters
    ----------
    train_loss_history : list[float]
        Training loss recorded at each epoch for one run.
    relative_tolerance : float, default=0.01
        Relative closeness threshold to the final training loss.
        Here, 0.01 means "within 1% of the final training loss."
    absolute_floor : float, default=1e-4
        Minimum tolerance allowed, so the rule does not become unrealistically
        strict when the final training loss is extremely small.

    Returns
    -------
    int
        The first epoch after which the remaining training-loss values
        stay close to the final training loss.
    """
    # The final training loss is the last value in the recorded history.
    final_train_loss = train_loss_history[-1]

    # Build the tolerance band around the final training loss.
    # We use either:
    # - 1% of the final loss, or
    # - an absolute minimum tolerance of 0.0001
    # whichever is larger.
    tolerance = max(absolute_floor, relative_tolerance * abs(final_train_loss))

    # Total number of epochs actually run for this experiment.
    epochs_ran = len(train_loss_history)

    # Scan from the first epoch forward.
    # We are looking for the earliest point after which the remaining
    # training-loss values all stay within the tolerance band.
    for start_idx in range(epochs_ran):
        # Tail of the curve from the current epoch to the end.
        tail = train_loss_history[start_idx:]

        # If every remaining value is sufficiently close to the final loss,
        # then this epoch is our convergence point.
        if all(abs(loss - final_train_loss) <= tolerance for loss in tail):
            # Convert 0-based index to 1-based epoch number.
            return start_idx + 1

    # Fallback: if no earlier stable point is found, treat the final epoch
    # as the convergence epoch.
    return epochs_ran


def add_derived_metrics(results):
    """
    Add the derived values needed for analysis.

    For loss quality, we use best validation loss within each run.
    For convergence speed, we use training-loss convergence epoch.

    Parameters
    ----------
    results : list of dict
        Raw experiment results loaded from the JSON file.

    Returns
    -------
    list of dict
        The same results list, with extra derived fields added to each run.
    """
    for run in results:
        # Best validation loss achieved by this run.
        run["best_val_loss"] = min(run["val_loss_history"])

        # Final training loss reached by this run.
        run["final_train_loss"] = run["train_loss_history"][-1]

        # Convergence epoch computed from the training-loss history.
        run["convergence_epoch"] = convergence_epoch(run["train_loss_history"])

    return results


def add_normalized_best_val_loss(results):
    """
    Normalize best validation loss within each problem separately.

    This is necessary for the combined overall comparison because:
    - classification uses BCE
    - regression uses MSE

    These losses are not on the same scale, so they should not be averaged
    directly across tasks.

    Parameters
    ----------
    results : list of dict
        Experiment results with derived metrics already added.

    Returns
    -------
    list of dict
        The same results list, with normalized best validation loss added.
    """
    # Group runs by problem so normalization happens inside each task only.
    problem_groups = {}
    for run in results:
        problem_groups.setdefault(run["problem_name"], []).append(run)

    # Normalize best validation loss separately inside each problem group.
    for _, runs in problem_groups.items():
        values = [r["best_val_loss"] for r in runs]
        min_v = min(values)
        max_v = max(values)

        for r in runs:
            # If all values are identical, assign 0.0 to avoid division by zero.
            if max_v == min_v:
                r["normalized_best_val_loss"] = 0.0
            else:
                # Standard min-max normalization.
                r["normalized_best_val_loss"] = (r["best_val_loss"] - min_v) / (max_v - min_v)

    return results


def filter_by_problem(results, problem_name=None):
    """
    Filter runs by problem name.

    Parameters
    ----------
    results : list of dict
        Full results list.
    problem_name : str or None
        Problem name to keep. If None, return all runs.

    Returns
    -------
    list of dict
        Filtered runs.
    """
    if problem_name is None:
        return results
    return [r for r in results if r["problem_name"] == problem_name]


def group_by_key(results, key):
    """
    Group a list of run dictionaries by a chosen key.

    Parameters
    ----------
    results : list of dict
        Runs to group.
    key : str
        Dictionary key to group by, such as 'optimizer' or 'architecture'.

    Returns
    -------
    dict[str, list[dict]]
        Dictionary mapping each group name to the list of runs in that group.
    """
    groups = {}
    for run in results:
        groups.setdefault(run[key], []).append(run)
    return groups


def summarize_task_group(group_runs):
    """
    Build a task-level summary for one group of runs.

    This is used for:
    - classification tables
    - regression tables

    At the task level, we can report average best validation loss directly,
    because all runs in the table use the same loss scale.

    Parameters
    ----------
    group_runs : list of dict
        Runs belonging to one group (same optimizer or same architecture).

    Returns
    -------
    dict
        Summary statistics for that group.
    """
    return {
        # Number of runs contributing to this group average.
        "num_runs": len(group_runs),

        # Average convergence epoch based on training-loss convergence.
        "avg_convergence_epoch": mean(r["convergence_epoch"] for r in group_runs),

        # Average best validation loss inside this task.
        "avg_best_val_loss": mean(r["best_val_loss"] for r in group_runs),
    }


def summarize_combined_group(group_runs):
    """
    Build a combined overall summary for one group of runs.

    This is used for the final overall answers across all experiments.
    Because classification and regression use different loss scales,
    we use normalized best validation loss here instead of raw loss.

    Parameters
    ----------
    group_runs : list of dict
        Runs belonging to one group (same optimizer or same architecture).

    Returns
    -------
    dict
        Summary statistics for that group.
    """
    return {
        # Number of runs contributing to this group average.
        "num_runs": len(group_runs),

        # Average convergence epoch across all runs in the group.
        "avg_convergence_epoch": mean(r["convergence_epoch"] for r in group_runs),

        # Average normalized best validation loss across all runs in the group.
        "avg_normalized_best_val_loss": mean(r["normalized_best_val_loss"] for r in group_runs),
    }


def ordered_keys(summary_dict, preferred_order):
    """
    Return keys from a summary dictionary in a preferred display order.

    Parameters
    ----------
    summary_dict : dict
        Summary dictionary whose keys should be ordered.
    preferred_order : list[str]
        Desired order, such as OPTIMIZER_ORDER or ARCHITECTURE_ORDER.

    Returns
    -------
    list[str]
        Keys that exist in summary_dict, ordered according to preferred_order.
    """
    return [k for k in preferred_order if k in summary_dict]


def format_task_table(title, summary_dict, preferred_order):
    """
    Format a plain-text table for one task-specific summary section.

    Parameters
    ----------
    title : str
        Section title.
    summary_dict : dict
        Summary statistics by optimizer or architecture.
    preferred_order : list[str]
        Display order for the rows.

    Returns
    -------
    str
        A formatted multi-line string representing the table.
    """
    # Start the section with a title and underline.
    lines = [title, "-" * len(title)]

    # Add the table header row.
    lines.append(
        f"{'Group':<15}"
        f"{'Runs':<8}"
        f"{'Avg Conv Epoch':<18}"
        f"{'Avg Best Val Loss':<20}"
    )

    # Add one row per group in the requested order.
    for group_name in ordered_keys(summary_dict, preferred_order):
        stats = summary_dict[group_name]
        lines.append(
            f"{group_name:<15}"
            f"{stats['num_runs']:<8}"
            f"{stats['avg_convergence_epoch']:<18.2f}"
            f"{stats['avg_best_val_loss']:<20.6f}"
        )

    # Add a blank line after the table for readability.
    lines.append("")
    return "\n".join(lines)


def format_combined_table(title, summary_dict, preferred_order):
    """
    Format a plain-text table for one combined-overall summary section.

    Parameters
    ----------
    title : str
        Section title.
    summary_dict : dict
        Summary statistics by optimizer or architecture.
    preferred_order : list[str]
        Display order for the rows.

    Returns
    -------
    str
        A formatted multi-line string representing the table.
    """
    # Start the section with a title and underline.
    lines = [title, "-" * len(title)]

    # Add the table header row.
    lines.append(
        f"{'Group':<15}"
        f"{'Runs':<8}"
        f"{'Avg Conv Epoch':<18}"
        f"{'Avg Norm Best Val Loss':<24}"
    )

    # Add one row per group in the requested order.
    for group_name in ordered_keys(summary_dict, preferred_order):
        stats = summary_dict[group_name]
        lines.append(
            f"{group_name:<15}"
            f"{stats['num_runs']:<8}"
            f"{stats['avg_convergence_epoch']:<18.2f}"
            f"{stats['avg_normalized_best_val_loss']:<24.6f}"
        )

    # Add a blank line after the table for readability.
    lines.append("")
    return "\n".join(lines)


def best_group(summary_dict, field_name):
    """
    Return the group name with the smallest value for a chosen summary field.

    This is used because:
    - lower convergence epoch means faster convergence
    - lower loss means better loss quality

    Parameters
    ----------
    summary_dict : dict
        Summary statistics by group.
    field_name : str
        Name of the field to minimize.

    Returns
    -------
    str
        Name of the best group under that criterion.
    """
    return min(summary_dict.items(), key=lambda x: x[1][field_name])[0]


def architecture_name_to_depth(architecture_name):
    """
    Convert architecture code names into more readable depth labels.

    Parameters
    ----------
    architecture_name : str
        Architecture identifier, such as A1 or A2.

    Returns
    -------
    str
        Human-readable description of the architecture depth.
    """
    if architecture_name == "A1":
        return "A1 (1 hidden layer)"
    if architecture_name == "A2":
        return "A2 (3 hidden layers)"
    return architecture_name


def build_depth_effect_sentence(combined_architecture_summary):
    """
    Build a short verbal answer for how network depth affects optimization overall.

    The answer compares:
    - which architecture converges faster on average
    - which architecture achieves better average normalized loss

    Parameters
    ----------
    combined_architecture_summary : dict
        Combined summary for architectures across all experiments.

    Returns
    -------
    str
        One sentence (or short pair of clauses) describing the depth effect.
    """
    # Find the architecture with the lower average convergence epoch.
    faster_arch = best_group(combined_architecture_summary, "avg_convergence_epoch")

    # Find the architecture with the lower average normalized best validation loss.
    better_arch = best_group(combined_architecture_summary, "avg_normalized_best_val_loss")

    # Convert shorthand architecture names into readable labels.
    faster_label = architecture_name_to_depth(faster_arch)
    better_label = architecture_name_to_depth(better_arch)

    # If the same architecture is best on both criteria, say so directly.
    if faster_arch == better_arch:
        return (
            f"Overall, {faster_label} both converges faster on average "
            f"and achieves the better average normalized loss.")

    # Otherwise explain the trade-off.
    return (
        f"Overall, {faster_label} converges faster on average, "
        f"while {better_label} achieves the better average normalized loss. ")


def main():
    # ------------------------------------------------------------
    # 1. Load experiment results from local repo.
    # ------------------------------------------------------------
    results = load_results(RESULTS_PATH)

    # ------------------------------------------------------------
    # 2. Add derived metrics needed for analysis.
    # ------------------------------------------------------------
    # This adds:
    # - best_val_loss
    # - final_train_loss
    # - convergence_epoch
    results = add_derived_metrics(results)

    # ------------------------------------------------------------
    # 3. Add normalized loss values for combined overall comparisons.
    # ------------------------------------------------------------
    results = add_normalized_best_val_loss(results)

    # ------------------------------------------------------------
    # 4. Split runs by task for supporting per-task tables.
    # ------------------------------------------------------------
    classification_runs = filter_by_problem(results, "classification")
    regression_runs = filter_by_problem(results, "regression")

    # ------------------------------------------------------------
    # 5. Build classification summaries by optimizer and architecture.
    # ------------------------------------------------------------
    cls_optimizer_summary = {
        name: summarize_task_group(runs)
        for name, runs in group_by_key(classification_runs, "optimizer").items()
    }
    cls_architecture_summary = {
        name: summarize_task_group(runs)
        for name, runs in group_by_key(classification_runs, "architecture").items()
    }

    # ------------------------------------------------------------
    # 6. Build regression summaries by optimizer and architecture.
    # ------------------------------------------------------------
    reg_optimizer_summary = {
        name: summarize_task_group(runs)
        for name, runs in group_by_key(regression_runs, "optimizer").items()
    }
    reg_architecture_summary = {
        name: summarize_task_group(runs)
        for name, runs in group_by_key(regression_runs, "architecture").items()
    }

    # ------------------------------------------------------------
    # 7. Build combined overall summaries across all experiments.
    # ------------------------------------------------------------
    combined_optimizer_summary = {
        name: summarize_combined_group(runs)
        for name, runs in group_by_key(results, "optimizer").items()
    }
    combined_architecture_summary = {
        name: summarize_combined_group(runs)
        for name, runs in group_by_key(results, "architecture").items()
    }

    # ------------------------------------------------------------
    # 8. Compute the final overall answers.
    # ------------------------------------------------------------
    # Which optimizer converges fastest?
    overall_fastest_optimizer = best_group(combined_optimizer_summary, "avg_convergence_epoch")

    # Which optimizer gives the best loss value?
    # For the combined comparison, this uses normalized best validation loss.
    overall_best_optimizer = best_group(combined_optimizer_summary, "avg_normalized_best_val_loss")

    # How does depth affect optimization?
    depth_effect_sentence = build_depth_effect_sentence(combined_architecture_summary)

    # ------------------------------------------------------------
    # 9. Build the final report text as a list of lines.
    # ------------------------------------------------------------
    lines = []

    # Report title.
    lines.append("ANALYSIS")
    lines.append("================")
    lines.append("")

    # Explain the analysis method used.
    lines.append("Method")
    lines.append("------")
    lines.append(
        "For each run, convergence epoch is the first epoch after which all remaining "
        "training-loss values stay within a small tolerance of the final training loss.")
    
    lines.append("")
    lines.append(
        "For overall loss comparison across classification and regression, combined results use "
        "normalized best validation loss within each problem separately.")
    
    lines.append("")

    # Add supporting classification tables.
    lines.append("Supporting Data — Classification")
    lines.append("--------------------------------")
    lines.append("Loss : BCE")
    lines.append("")
    lines.append(
        format_task_table(
            "Optimizer Summary — Classification",
            cls_optimizer_summary,
            OPTIMIZER_ORDER,))
    
    lines.append(
        format_task_table(
            "Architecture Summary — Classification",
            cls_architecture_summary,
            ARCHITECTURE_ORDER,))
    

    # Add supporting regression tables.
    lines.append("Supporting Data — Regression")
    lines.append("----------------------------")
    lines.append("Loss: MSE")
    lines.append("")
    lines.append(
        format_task_table(
            "Optimizer Summary — Regression",
            reg_optimizer_summary,
            OPTIMIZER_ORDER,))
    
    lines.append(
        format_task_table(
            "Architecture Summary — Regression",
            reg_architecture_summary,
            ARCHITECTURE_ORDER,))

    # Add combined overall tables used for the final answers.
    lines.append("Supporting Data — Combined Overall")
    lines.append("---------------------------------")
    lines.append("Loss values below are normalized within each problem before averaging.")
    lines.append("")
    lines.append(
        format_combined_table(
            "Optimizer Summary — Combined Overall",
            combined_optimizer_summary,
            OPTIMIZER_ORDER,))
    
    lines.append(
        format_combined_table(
            "Architecture Summary — Combined Overall",
            combined_architecture_summary,
            ARCHITECTURE_ORDER,))

    # Add the final direct answers to the three questions.
    lines.append("Answers")
    lines.append("-------")
    lines.append(f"1. Fastest optimizer on average over all experiments: {overall_fastest_optimizer}")
    lines.append(
        f"2. Best optimizer by average loss-function value over all experiments using average normalized best validation loss: {overall_best_optimizer} "
    )
    lines.append(f"3. How network depth affects optimization: {depth_effect_sentence}")
    lines.append("")

    # Join all report lines into one final text block.
    report_text = "\n".join(lines)

    # ------------------------------------------------------------
    # 10. Write the final analysis report to disk.
    # ------------------------------------------------------------
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)

if __name__ == "__main__":
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()

    elapsed = end_time - start_time
    print(f"\nTotal execution time: {elapsed:.2f} seconds")