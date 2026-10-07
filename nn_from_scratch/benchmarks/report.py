"""
Compare the from-scratch NumPy network with the library models.

Input:
- the newest ``reports/main_results_full_<stamp>.json`` (the NumPy network)
- the newest ``reports/benchmark_<library>_results_<stamp>.json`` for each
  library that has results (currently scikit-learn)

Output:
- ``reports/benchmark_report_<stamp>.txt``: comparison tables per task
- ``reports/figures/benchmarks/benchmark_<task>_<stamp>.png``: one bar
  chart per task

Every model is treated the same way: on each seed, the configuration with
the lowest validation metric is selected, and its test metric is reported.
Results are averaged over seeds (mean ± sample standard deviation).

Run it with ``python -m nn_from_scratch.benchmarks.report`` or
``make benchmark-report``.
"""

from collections import Counter
import json
import math
import os
from statistics import mean, stdev
import sys
import time

import matplotlib.pyplot as plt

from nn_from_scratch.config import (
    BENCHMARK_FIGURES_DIR,
    REPORT_DIR,
    RUN_STAMP,
    latest_stamped_file,
    resolve_results_path,
    stamped_filename,
)

# Libraries whose results the report looks for, in display order.
LIBRARIES = ["sklearn"]

# Display names for libraries and models.
LIBRARY_LABELS = {"sklearn": "scikit-learn"}
MODEL_LABELS = {
    "logistic_regression": "logistic regression",
    "ridge": "ridge regression",
    "svm": "SVM (RBF)",
    "svr": "SVR (RBF)",
    "random_forest": "random forest",
    "gradient_boosting": "gradient boosting",
    "mlp": "MLP",
}

# The NumPy network's baseline architectures, as in the main analysis.
BASELINE_ARCHITECTURES = ["A1", "A2"]

# Metric names per problem: (main metric, secondary metric key, secondary label).
PROBLEM_METRICS = {
    "classification": ("BCE", "test_accuracy", "Accuracy"),
    "regression": ("MSE", "test_r2", "R²"),
}


def load_json(path):
    """
    Read a JSON results file.

    Parameters
    ----------
    path : str
        Path of the file.

    Returns
    -------
    list of dict
        The records stored in the file.

    Notes
    -----
    Processing:
    1. Open the file as UTF-8 and parse it.
    """
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def newest_library_results(library):
    """
    Find the newest results file for one library.

    Parameters
    ----------
    library : str
        Library name, e.g. "sklearn".

    Returns
    -------
    str or None
        Path of the newest reports/benchmark_<library>_results_<stamp>.json,
        or None if the library has not been run yet.

    Notes
    -----
    Processing:
    1. Ask latest_stamped_file for the newest matching file.
    2. Return None instead of raising when there is none, so the report
       can include whichever libraries have results.
    """
    try:
        return latest_stamped_file(REPORT_DIR, f"benchmark_{library}_results", ".json")
    except FileNotFoundError:
        return None


def is_diverged(run):
    """
    Return True if a NumPy network run diverged.

    Parameters
    ----------
    run : dict
        One record from the main results file.

    Returns
    -------
    bool
        True if the run's "diverged" flag is set, or its best validation
        loss is not a finite number.

    Notes
    -----
    Processing:
    1. Same rule as nn_from_scratch.analysis.is_diverged: diverged runs are
       never selected.
    """
    return bool(run.get("diverged")) or not math.isfinite(run["best_val_loss"])


def select_per_seed(runs, val_key):
    """
    For each seed, select the run with the lowest validation metric.

    Parameters
    ----------
    runs : list of dict
        Candidate runs for one model (or group of models), across seeds.
        Each has a "seed" and a validation metric under val_key.
    val_key : str
        "best_val_loss" for NumPy network runs, "val_metric" for library
        runs. Both hold the validation BCE or MSE.

    Returns
    -------
    list of dict
        One selected run per seed, sorted by seed. Seeds whose runs all
        have a non-finite validation metric are skipped.

    Notes
    -----
    Processing:
    1. Group the runs by seed.
    2. In each group, drop runs with a non-finite validation metric.
    3. Keep the run with the smallest validation metric.

    The test set is never used for selection.
    """
    by_seed = {}
    for run in runs:
        by_seed.setdefault(run["seed"], []).append(run)

    selected = []
    for seed in sorted(by_seed):
        candidates = [r for r in by_seed[seed] if math.isfinite(r[val_key])]
        if candidates:
            selected.append(min(candidates, key=lambda r: r[val_key]))
    return selected


def mean_pm_std(values, digits):
    """
    Format a list of numbers as "mean ± std".

    Parameters
    ----------
    values : list of float
        One value per seed. None values are ignored.
    digits : int
        Decimal places.

    Returns
    -------
    str
        For example "0.5200 ± 0.1876", or "-" when there are no values.

    Notes
    -----
    Processing:
    1. Drop None values (e.g. metrics missing from older results files).
    2. Format the mean and the sample standard deviation (0 for one value).
    """
    values = [v for v in values if v is not None]
    if not values:
        return "-"
    spread = stdev(values) if len(values) > 1 else 0.0
    return f"{mean(values):.{digits}f} ± {spread:.{digits}f}"


def describe_config(run):
    """
    Short text describing a selected configuration.

    Parameters
    ----------
    run : dict
        A NumPy network run or a library run.

    Returns
    -------
    str
        For a NumPy run, e.g. "A1 · adabelief · LR 0.1 · bs 16". For a
        library run, its hyperparameters, e.g. "C=10, gamma=scale".

    Notes
    -----
    Processing:
    1. NumPy runs are recognised by their "architecture" key.
    2. Library runs list their hyperparameters, leaving out fixed settings
       that are the same for every configuration (max_iter, probability,
       kernel, n_estimators).
    """
    if "architecture" in run:
        return (
            f"{run['architecture']} · {run['optimizer']} · "
            f"LR {run['learning_rate']} · bs {run['batch']}"
        )
    fixed = {"max_iter", "probability", "kernel", "n_estimators"}
    return ", ".join(f"{k}={v}" for k, v in run["params"].items() if k not in fixed) or "default"


def summarize(label, selected, secondary_key):
    """
    Summarize one model's selected runs across seeds.

    Parameters
    ----------
    label : str
        Row label for the report, e.g. "scikit-learn · random forest".
    selected : list of dict
        One selected run per seed.
    secondary_key : str
        "test_accuracy" or "test_r2".

    Returns
    -------
    dict
        Keys: "label", "test" (list of float), "secondary" (list of float
        or None), "error_removed" (list of float), "train_seconds" (list of
        float or None) and "config" (str: the configuration chosen most
        often, with how many seeds chose it).

    Notes
    -----
    Processing:
    1. Collect the test metric, secondary metric, training time and
       "error removed" (1 - test / constant-prediction baseline) of each
       selected run.
    2. Find the most frequently selected configuration.
    """
    configs = Counter(describe_config(r) for r in selected)
    config, count = configs.most_common(1)[0]
    return {
        "label": label,
        "test": [r["test_metric"] for r in selected],
        "secondary": [r.get(secondary_key) for r in selected],
        "error_removed": [1 - r["test_metric"] / r["baseline_test_metric"] for r in selected],
        "train_seconds": [r.get("train_seconds") for r in selected],
        "config": f"{config} ({count}/{len(selected)} seeds)",
    }


def build_rows(problem_name, numpy_runs, library_runs):
    """
    Build the comparison rows for one problem.

    Parameters
    ----------
    problem_name : str
        "classification" or "regression".
    numpy_runs : list of dict
        All NumPy network runs for this problem.
    library_runs : dict of str to list of dict
        Library name -> that library's runs for this problem.

    Returns
    -------
    rows : list of dict
        Summaries (see summarize) in display order: the NumPy network
        (baseline architectures, then all architectures), each library
        model, then each library's best model overall.
    selections : dict of str to list of dict
        The selected runs per seed for the two head-to-head entries:
        "numpy" (NumPy network, all architectures) and each library name
        (that library's best model overall).

    Notes
    -----
    Processing:
    1. NumPy network: drop diverged runs, then select per seed among A1 /
       A2, and separately among all architectures.
    2. Each library model: select per seed among its configurations.
    3. Each library overall: select per seed among all its models'
       configurations, i.e. what a practitioner would pick from that
       library using the validation set.
    """
    _, secondary_key, _ = PROBLEM_METRICS[problem_name]
    numpy_ok = [r for r in numpy_runs if not is_diverged(r)]

    numpy_baseline = select_per_seed(
        [r for r in numpy_ok if r["architecture"] in BASELINE_ARCHITECTURES], "best_val_loss"
    )
    numpy_all = select_per_seed(numpy_ok, "best_val_loss")
    rows = [
        summarize("NumPy NN (from scratch) · A1 / A2", numpy_baseline, secondary_key),
        summarize("NumPy NN (from scratch) · all 8 architectures", numpy_all, secondary_key),
    ]
    selections = {"numpy": numpy_all}

    for library, runs in library_runs.items():
        name = LIBRARY_LABELS.get(library, library)
        models = list(dict.fromkeys(r["model"] for r in runs))
        for model in models:
            selected = select_per_seed([r for r in runs if r["model"] == model], "val_metric")
            rows.append(
                summarize(f"{name} · {MODEL_LABELS.get(model, model)}", selected, secondary_key)
            )
        best = select_per_seed(runs, "val_metric")
        best_row = summarize(f"{name} · best model (selected on validation)", best, secondary_key)
        best_row["config"] = ", ".join(
            f"{MODEL_LABELS.get(m, m)} ×{n}"
            for m, n in Counter(r["model"] for r in best).most_common()
        )
        rows.append(best_row)
        selections[library] = best

    return rows, selections


def format_table(problem_name, rows):
    """
    Format one problem's comparison rows as a text table.

    Parameters
    ----------
    problem_name : str
        "classification" or "regression".
    rows : list of dict
        Summaries from build_rows.

    Returns
    -------
    str
        A titled, fixed-width table, followed by the configuration chosen
        most often for each row.

    Notes
    -----
    Processing:
    1. Write one line per row: label, test metric, secondary metric,
       error removed (%) and training time of the selected configuration.
    2. List the most often selected configuration per row underneath.
    """
    metric, _, secondary = PROBLEM_METRICS[problem_name]
    digits = 5 if problem_name == "classification" else 4
    title = f"{problem_name.capitalize()} — test {metric} (lower is better), mean ± std over seeds"
    lines = [title, "-" * len(title)]
    lines.append(
        f"{'Model':<52}{'Test ' + metric:<24}{secondary:<20}{'Error removed %':<20}{'Fit time (s)':<14}"
    )
    for row in rows:
        lines.append(
            f"{row['label']:<52}"
            f"{mean_pm_std(row['test'], digits):<24}"
            f"{mean_pm_std(row['secondary'], 4):<20}"
            f"{mean_pm_std([100 * v for v in row['error_removed']], 2):<20}"
            f"{mean_pm_std(row['train_seconds'], 2):<14}"
        )
    lines.append("")
    lines.append("Most often selected configuration:")
    for row in rows:
        lines.append(f"- {row['label']}: {row['config']}")
    lines.append("")
    return "\n".join(lines)


def format_head_to_head(problem_name, selections):
    """
    Compare the NumPy network with each library's best model, seed by seed.

    Parameters
    ----------
    problem_name : str
        "classification" or "regression".
    selections : dict of str to list of dict
        From build_rows: "numpy" and one entry per library.

    Returns
    -------
    str
        One line per library: on how many seeds the NumPy network's test
        metric was lower (better), and the per-seed test metrics.

    Notes
    -----
    Processing:
    1. Pair the NumPy network's selected run with the library's selected
       run on the same seed.
    2. Count the seeds where the NumPy network's test metric is lower.
    """
    metric, _, _ = PROBLEM_METRICS[problem_name]
    numpy_by_seed = {r["seed"]: r for r in selections["numpy"]}
    lines = [f"Head-to-head per seed — {problem_name} (test {metric}):"]
    for library, selected in selections.items():
        if library == "numpy":
            continue
        pairs = [(numpy_by_seed[r["seed"]], r) for r in selected if r["seed"] in numpy_by_seed]
        wins = sum(n["test_metric"] < lib["test_metric"] for n, lib in pairs)
        detail = "; ".join(
            f"seed {n['seed']}: {n['test_metric']:.4g} vs {lib['test_metric']:.4g}"
            for n, lib in pairs
        )
        lines.append(
            f"- NumPy NN (all architectures) vs {LIBRARY_LABELS.get(library, library)} best: "
            f"NumPy better on {wins}/{len(pairs)} seeds ({detail})"
        )
    lines.append("")
    return "\n".join(lines)


def plot_problem(problem_name, rows):
    """
    Draw a horizontal bar chart of the test metric for one problem.

    Parameters
    ----------
    problem_name : str
        "classification" or "regression".
    rows : list of dict
        Summaries from build_rows.

    Returns
    -------
    str
        Path of the saved PNG in reports/figures/benchmarks/.

    Notes
    -----
    Processing:
    1. One bar per row: the mean test metric across seeds, with the
       standard deviation as an error bar. NumPy rows are highlighted.
    2. Use a log scale: the values span orders of magnitude (BCE from
       about 1e-4 to 0.04; MSE from about 0.2 for the best models to 9
       for ridge regression), which a linear axis would flatten.
    3. Save the figure with the run stamp and close it.
    """
    metric, _, _ = PROBLEM_METRICS[problem_name]
    labels = [row["label"] for row in rows]
    means = [mean(row["test"]) for row in rows]
    stds = [stdev(row["test"]) if len(row["test"]) > 1 else 0.0 for row in rows]
    colors = ["tab:orange" if label.startswith("NumPy") else "tab:blue" for label in labels]

    fig, ax = plt.subplots(figsize=(10, 0.55 * len(rows) + 1.5))
    positions = list(range(len(rows)))[::-1]
    ax.barh(positions, means, xerr=stds, color=colors, capsize=3)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.set_xscale("log")
    ax.set_xlabel(f"Test {metric} (mean ± std over seeds; lower is better)")
    ax.set_title(f"{problem_name.capitalize()}: from-scratch NumPy network vs. library models")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    os.makedirs(BENCHMARK_FIGURES_DIR, exist_ok=True)
    path = os.path.join(BENCHMARK_FIGURES_DIR, stamped_filename(f"benchmark_{problem_name}.png"))
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved figure to: {path}")
    return path


def main(numpy_results_path=None):
    """
    Build the comparison report and figures.

    Parameters
    ----------
    numpy_results_path : str or None, default=None
        A specific main_results_full_<stamp>.json. None uses the newest.

    Returns
    -------
    None
        Writes reports/benchmark_report_<stamp>.txt and one figure per
        problem, and prints their paths.

    Raises
    ------
    FileNotFoundError
        If no library has results yet, or no NumPy results file exists.

    Notes
    -----
    Processing:
    1. Load the NumPy network's results and each library's newest results.
    2. For each problem, build the rows, the table, the head-to-head lines
       and the figure.
    3. Write the report with a header explaining the method and listing
       the input files.
    """
    numpy_path = resolve_results_path(numpy_results_path)
    numpy_results = load_json(numpy_path)
    library_paths = {lib: newest_library_results(lib) for lib in LIBRARIES}
    library_paths = {lib: path for lib, path in library_paths.items() if path}
    if not library_paths:
        raise FileNotFoundError(
            "No library results found. Run python -m nn_from_scratch.benchmarks.run first."
        )
    library_results = {lib: load_json(path) for lib, path in library_paths.items()}

    print(f"Run stamp: {RUN_STAMP}")
    lines = ["LIBRARY COMPARISON", "==================", ""]
    lines.append(
        "The from-scratch NumPy network is compared with library models trained on exactly "
        "the same train / validation / test splits (same seeds), scored with the same NumPy "
        "metric functions. For every model, the configuration with the lowest validation "
        "metric is selected on each seed and its test metric is reported; the test set is "
        "never used for selection. 'Error removed' is 1 - test / constant-prediction "
        "baseline. 'Fit time' is the training time of the selected configuration on one "
        "seed (one CPU run each; the NumPy network trains up to 100 epochs with early "
        "stopping)."
    )
    lines.append("")
    lines.append(f"NumPy network results: {os.path.basename(numpy_path)}")
    for lib, path in library_paths.items():
        lines.append(f"{LIBRARY_LABELS.get(lib, lib)} results: {os.path.basename(path)}")
    lines.append("")

    for problem_name in PROBLEM_METRICS:
        numpy_runs = [r for r in numpy_results if r["problem_name"] == problem_name]
        library_runs = {
            lib: [r for r in records if r["problem_name"] == problem_name]
            for lib, records in library_results.items()
        }
        rows, selections = build_rows(problem_name, numpy_runs, library_runs)
        lines.append(format_table(problem_name, rows))
        lines.append(format_head_to_head(problem_name, selections))
        plot_problem(problem_name, rows)

    path = os.path.join(REPORT_DIR, stamped_filename("benchmark_report.txt"))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved benchmark report to: {path}")


# Running this module writes the comparison report and figures from the
# newest results (optionally a specific NumPy results file given as the
# first argument).
if __name__ == "__main__":
    start_time = time.perf_counter()
    main(sys.argv[1] if len(sys.argv) > 1 else None)
    print(f"\nTotal execution time: {time.perf_counter() - start_time:.2f} seconds")
