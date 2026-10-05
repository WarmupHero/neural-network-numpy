# Neural Network from Scratch in NumPy: Detailed Documentation

This is the in-depth reference for the project. For a quick overview, results, and plots, see the [repository README](../README.md).

## Contents

1. [Project overview](#1-project-overview)
   - [Pipeline map](#pipeline-map)
   - [Dependencies](#dependencies)
   - [Running the pipeline](#running-the-pipeline)
   - [Tests](#tests)
2. [Experiment configuration](#2-experiment-configuration)
3. [Analysis details](#3-analysis-details)
   - [comparisons.py](#comparisonspy)
   - [analysis.py](#analysispy)

---

## 1. Project overview

This project implements a feed-forward neural network from scratch in NumPy for two tasks: binary classification and regression.

1. The pipeline downloads the datasets.
2. It preprocesses them: cleaning, optional scaling, and a train / validation / test split (60 / 20 / 20).
3. It loads the experiment settings (architectures, optimizers, learning rates, batch sizes, early stopping) from JSON configuration files.
4. It trains and evaluates every configured network and saves the results.
5. Separate scripts read the saved results to generate comparison plots and a written analysis.

| Task | Dataset | Target | Loss | Evaluation metric |
|---|---|---|---|---|
| Classification | Banknote Authentication (UCI #267) | `class` (0/1) | Binary cross-entropy (BCE) | BCE |
| Regression | Energy Efficiency (UCI #242) | `Heating_Load` (`Cooling_Load` is dropped) | Mean squared error (MSE) | MSE |

The same quantity is used for training and for evaluation, so the reported metric is directly comparable with the loss curves. The `test_metric` column in `report/main_summary.csv` holds the **test-set BCE** for classification rows and the **test-set MSE** for regression rows. Lower is better in both cases. The metric is chosen in `src/train.py` (`Trainer._compute_metric`) and implemented in `src/metrics.py`, which reuses the loss classes from `src/losses.py`.

### Pipeline map

| Module | Role | Reads | Writes |
|---|---|---|---|
| `src/utils.py` | Shared constants: project paths and `RANDOM_SEED` | – | – |
| `src/fetch_data.py` | Downloads the raw UCI datasets with `ucimlrepo` if they are not cached | – | `datasets/banknote_auth.csv`, `datasets/energy_efficiency.csv` |
| `src/preprocessing.py` | Removes duplicates, splits train/val/test with a NumPy-only `train_val_test_split` (stratified by class for classification), optionally standardizes features (fit on train only), runs EDA plots | `datasets/*.csv` | `report/preprocessing_graphs/*.png` |
| `src/scalers.py` | Custom standard scaler, (x − μ) / σ | – | – |
| `src/visualizations.py` | Exploratory and preprocessing plots used by `preprocessing.py` | – | – |
| `src/config_loader.py` | Loads and validates the JSON experiment configs | `configs/*.json` | – |
| `src/layers.py` | Bias-free dense layer, Z = X·W, with its backward pass | – | – |
| `src/activations.py` | ReLU, Sigmoid, Tanh, Linear (forward and backward) | – | – |
| `src/network.py` | Builds the network from a config; full forward and backward passes | – | – |
| `src/losses.py` | BCE and MSE (forward and gradient) | – | – |
| `src/optimizers.py` | SGD, Momentum SGD, AdaBelief | – | – |
| `src/metrics.py` | Evaluation metrics: BCE (classification) and MSE (regression) | – | – |
| `src/train.py` | Mini-batch training loop, validation, early stopping, best-weight restore, test evaluation | – | – |
| `main.py` | Orchestrates the full experiment grid | everything above | `report/main_results_full.json`, `report/main_summary.csv` |
| `src/comparisons.py` | Optimizer, depth, and learning-rate comparison plots with short text analyses | `report/main_results_full.json` | `report/comparisons/*` |
| `src/analysis.py` | Aggregate analysis across all runs | `report/main_results_full.json` | `report/analysis.txt` |

`main.py` has a `SHOW_EDA` flag (default `False`). Set it to `True` to display the preprocessing plots interactively while the pipeline runs. The plots are saved to `report/preprocessing_graphs/` either way.

### Dependencies

Pinned versions are in [`requirements.txt`](../requirements.txt). The project was developed with Python 3.12.

The model, training, metrics, scaling, and data splitting use **NumPy only**. The other packages handle loading the CSVs, plotting, and downloading the datasets.

```
numpy==1.26.4        # all model, training, and preprocessing math
pandas==2.2.2        # loading and cleaning the CSV files
matplotlib==3.9.2    # plots
seaborn==0.13.2      # EDA plots
ucimlrepo==0.0.7     # one-time dataset download
```

### Running the pipeline

Run every command from the repository root:

```bash
pip install -r requirements.txt

# 1. Load or download the data, preprocess, train all configured experiments.
#    Writes report/main_results_full.json and report/main_summary.csv
python main.py

# 2. Optimizer / network-depth / learning-rate comparisons.
#    Writes plots and short analyses to report/comparisons/
python -m src.comparisons

# 3. Aggregate analysis.
#    Writes report/analysis.txt
python -m src.analysis
```

Steps 2 and 3 read `report/main_results_full.json` and fail if `main.py` hasn't been run first. The repository already includes a results file, so you can run them immediately.

### Tests

```bash
pip install pytest
python -m pytest
```

- `tests/test_gradients.py` compares every analytic gradient against a central finite-difference estimate: each activation, both losses, and the weight gradients of whole A1- and A2-style networks.
- `tests/test_preprocessing.py` checks that the NumPy split has the right sizes, has no overlapping rows, preserves class balance when stratified, and is reproducible from the seed.
- `tests/test_training.py` checks that each optimizer reduces the loss on a toy problem, and that `Trainer.fit` / `Trainer.evaluate` run end to end and report BCE as the classification metric.

---

## 2. Experiment configuration

Two JSON files define the experiment setup:

- `configs/classification_experiments.json`
- `configs/regression_experiments.json`

They control a large part of the pipeline, within the limits of the current implementation. They are configurable, but they don't define a fully general neural-network framework.

### What the JSON controls

**Preprocessing**
- `preprocessing.enabled`: if `true`, the dataset is cleaned and can optionally be scaled. If `false`, cleaning, EDA, and scaling are skipped, but the data is still split into train/validation/test because training requires those splits.
- `preprocessing.scale_features` (`true`/`false`): standardizes input features with the custom `StandardScaler`, using statistics fit on the training split only. If preprocessing is disabled, scaling is skipped too.

**Architectures**
- Architecture names (for example `A1`, `A2`): any JSON key is allowed, so you can add new architectures.
- Number of layers: at least 1.
- `units` per layer: any integer ≥ 1. This is the layer's output size. There's no upper bound beyond compute and memory.
- `activation` per layer: one of `relu`, `sigmoid`, `tanh`, `linear`.

**Experiment sweep values**
- `optimizers`: `sgd`, `momentum` (aliases `momentumsgd`, `momentum_sgd`), `adabelief`.
- `learning_rates`: positive numbers, typically 0.1, 0.01, or 0.001.
- `batch_sizes`: positive integers, bounded by the size of the training split.

**Training settings**
- `epochs`: integer ≥ 1.
- `early_stopping`: `true` or `false`.
- `patience`: the number of consecutive epochs without a meaningful validation-loss improvement allowed before stopping.
- `min_delta`: the minimum decrease in validation loss that counts as an improvement. It should be small relative to the scale of the chosen loss.
- `min_epochs_before_early_stop`: the number of epochs that must finish before early stopping can trigger.

**Problem-level settings**
- `task_type`: `classification` or `regression`.
- `loss`: classification accepts `bce`, `binary_crossentropy`, `binary_cross_entropy`. Regression accepts `mse`.
- `input_dimension`: must match the number of dataset features, or execution fails.

### What the JSON does not control

1. **Layer types.** Each layer has a `"type"` field, but only `dense` is implemented. Convolutional, dropout, batch-norm, or recurrent layers would need code changes first.
2. **Optimizer internals.** Only the learning rate comes from JSON. Momentum uses β = 0.9, and AdaBelief uses β₁ = 0.9, β₂ = 0.999, ε = 1e-8, all fixed in `src/optimizers.py` (`get_optimizer`).
3. **Evaluation metrics.** The task type determines them: BCE for classification and MSE for regression, matching the training losses.
4. **Random seed.** It's set by `RANDOM_SEED` in `src/utils.py`.
5. **EDA display.** It's controlled by `SHOW_EDA` in `main.py` (see above).

### Summary

The JSON files make the experiment setup mostly configurable: architectures, optimizer selection, learning rates, batch sizes, training length, early stopping, and the supported task and loss choices.

The implementation itself supports only:
- feed-forward dense networks
- a fixed set of activations, losses, and optimizers
- fixed optimizer hyperparameters beyond the learning rate

---

## 3. Analysis details

### comparisons.py

`comparisons.py` compares optimizers, network depth, and learning rates. It loads the full JSON results file, selects matched runs, plots their training and validation loss curves, and saves the figures. For the depth and learning-rate comparisons, it also writes a short text analysis saying which setting converged faster and which ended with the lower loss.

Each comparison holds every setting fixed except one:

| Comparison | Fixed | Varies | Output |
|---|---|---|---|
| Optimizers | problem, architecture, learning rate, batch size | optimizer (SGD → Momentum → AdaBelief, one subplot each) | `optimizers_*.png` |
| Network depth | problem, optimizer, learning rate, batch size | architecture (A1 vs A2) | `depth_*.png`, `depth_analysis.txt` |
| Learning rate | problem, architecture, optimizer, batch size | learning rate (0.1 vs 0.001) | `learning_rate_*.png`, `learning_rate_analysis.txt` |

#### Convergence epoch

Convergence is measured from the training-loss curve:

1. `final_train_loss` is the last value in `train_loss_history`.
2. `tolerance = max(absolute_floor, relative_tolerance * abs(final_train_loss))`, with `relative_tolerance = 0.01` and `absolute_floor = 1e-4`. In practice this is a 1% band around the final loss that never gets narrower than 0.0001.
3. Starting from epoch 1, scan forward to find the first epoch after which all remaining training-loss values stay within that band.

That epoch is the **convergence epoch**, a practical definition of when training has essentially settled. The same helper also records `epochs_ran`, `final_train_loss`, and `final_val_loss`. These values appear in the metrics box on the depth and learning-rate plots.

#### Helper functions

- `load_results(path)`: loads the JSON file
- `filter_optimizer_runs(...)`: selects the 3 matched optimizer runs
- `filter_depth_runs(...)`: selects the 2 matched architecture runs
- `filter_learning_rate_runs(...)`: selects the 2 matched learning-rate runs
- `_compute_convergence_metrics(run)`: computes final losses and the convergence epoch
- `_plot_loss_curves(...)`: draws train/validation curves for one run
- `_save_and_show(...)`: saves the figure and displays it
- `_write_text_file(...)`: saves the short written analysis

`main()` chooses which experiments to visualize by calling the three plot functions with fixed arguments. To visualize different runs, change those arguments. The comparison logic stays the same.

### analysis.py

`analysis.py` answers three questions:

1. Which optimizer converges fastest on average?
2. Which optimizer reaches the best loss?
3. How does network depth affect optimization?

#### Derived metrics

For each run, it computes:
- `best_val_loss`: the minimum of `val_loss_history`
- `final_train_loss`: the last value of `train_loss_history`
- `convergence_epoch`: the same definition as in `comparisons.py`

"Fastest convergence" therefore means the lowest average convergence epoch.

#### Loss normalization

Classification uses BCE and regression uses MSE. These are on different scales, so they can't be averaged together directly. `analysis.py` min-max normalizes the best validation loss **within each problem**:

```
normalized_best_val_loss = (best_val_loss - min_loss_in_problem) / (max_loss_in_problem - min_loss_in_problem)
```

This puts each task's losses on a 0–1 scale before combining them. If every loss within a problem is identical, the script assigns 0.0 to avoid dividing by zero.

#### Grouping

- **Per task** (classification, regression), grouped by optimizer and by architecture: `num_runs`, `avg_convergence_epoch`, `avg_best_val_loss`.
- **Combined across tasks**, grouped by optimizer and by architecture: `num_runs`, `avg_convergence_epoch`, `avg_normalized_best_val_loss`. Normalized loss is used here because the groups mix classification and regression runs.

The "best" group is the one with the smallest value in the relevant field. For the depth question, the script compares A1 and A2 in the combined architecture summary, on both convergence epoch and normalized loss.

#### Helper functions

- `load_results(path)`: loads the JSON results
- `convergence_epoch(...)`: computes the convergence epoch from the training loss
- `add_derived_metrics(results)`: adds `best_val_loss`, `final_train_loss`, and `convergence_epoch`
- `add_normalized_best_val_loss(results)`: normalizes the best validation loss within each task
- `filter_by_problem(...)`: separates classification and regression runs
- `group_by_key(...)`: groups runs by optimizer or architecture
- `summarize_task_group(...)`: computes averages within one task
- `summarize_combined_group(...)`: computes averages across all runs using normalized loss
- `best_group(summary_dict, field_name)`: selects the group with the smallest value in the given field

The report (`report/analysis.txt`) contains a method section, supporting tables per task and combined, and direct answers to the three questions.

### In short

- `comparisons.py` explains individual matched experiments with plots and short text analyses.
- `analysis.py` answers the overall questions using averages across all runs.
