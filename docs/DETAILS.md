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

The same quantity is used for training and for evaluation, so the reported metric is directly comparable with the loss curves. The `test_metric` column in `report/main_summary_<stamp>.csv` holds the **test-set BCE** for classification rows and the **test-set MSE** for regression rows. Lower is better in both cases. The metric is chosen in `src/train.py` (`Trainer._compute_metric`) and implemented in `src/metrics.py`, which reuses the loss classes from `src/losses.py`.

### Pipeline map

| Module | Role | Reads | Writes |
|---|---|---|---|
| `src/utils.py` | Shared constants: project paths and `RANDOM_SEED`; the per-process `RUN_STAMP` and the `stamped_filename` / `latest_stamped_file` / `resolve_results_path` helpers | – | – |
| `src/fetch_data.py` | Downloads the raw UCI datasets with `ucimlrepo` if they are not cached | – | `datasets/banknote_auth.csv`, `datasets/energy_efficiency.csv` |
| `src/preprocessing.py` | Removes duplicates, splits train/val/test with a NumPy-only `train_val_test_split` (stratified by class for classification), optionally standardizes features (fit on train only), runs EDA plots | `datasets/*.csv` | `report/preprocessing_graphs/*.png` |
| `src/scalers.py` | Custom standard scaler, (x − μ) / σ | – | – |
| `src/visualizations.py` | Exploratory and preprocessing plots used by `preprocessing.py` | – | – |
| `src/config_loader.py` | Loads and validates the JSON experiment configs | `configs/*.json` | – |
| `src/layers.py` | Dense layer, Z = X·W (+ b), with its backward pass; optional bias and `normal` / `he` / `xavier` weight initialization. BatchNorm layer with learned scale and shift, running statistics, and a training / evaluation mode. Inverted Dropout layer | – | – |
| `src/activations.py` | ReLU, Sigmoid, Tanh, Linear (forward and backward) | – | – |
| `src/network.py` | Builds the network from a config; full forward and backward passes | – | – |
| `src/losses.py` | BCE and MSE (forward and gradient) | – | – |
| `src/optimizers.py` | SGD, Momentum SGD, AdaBelief | – | – |
| `src/metrics.py` | Evaluation metrics: BCE (classification) and MSE (regression) | – | – |
| `src/train.py` | Mini-batch training loop, validation, early stopping, best-model restore, divergence detection, test evaluation; switches the network between training and evaluation mode | – | – |
| `main.py` | Orchestrates the full experiment grid | everything above | `report/main_results_full_<stamp>.json`, `report/main_summary_<stamp>.csv` |
| `src/comparisons.py` | Optimizer, depth, and learning-rate comparison plots with short text analyses | newest `report/main_results_full_*.json` (or a path given as the first argument) | `report/comparisons/*_<stamp>.*` |
| `src/analysis.py` | Aggregate analysis across all runs | newest `report/main_results_full_*.json` (or a path given as the first argument) | `report/analysis_<stamp>.txt` |

`main.py` has a `SHOW_EDA` flag (default `False`). Set it to `True` to display the preprocessing plots interactively while the pipeline runs. The plots are saved to `report/preprocessing_graphs/` either way.

#### Run stamps

Every file a run writes carries a stamp of the form `name_YYYYMMDD-HHMMSS.ext` (local time), so nothing from an earlier run is ever overwritten. The stamp is `RUN_STAMP` in `src/utils.py`, computed once when the module is first imported, so all files written by one process share it. `main.py`, `src.comparisons` and `src.analysis` are separate processes and therefore get their own stamps. `src.comparisons` and `src.analysis` print which results file they used.

Only the newest version of each output is committed. `.gitignore` ignores all stamped files under `report/`, and the pre-commit hook in `.githooks/` runs `tools/stage_latest_outputs.py`, which:

- force-adds the newest stamped file of each output (using `latest_stamped_outputs` in `src/utils.py`)
- removes older stamped files from the index, leaving them on disk
- rewrites stamped output names in `README.md` and `docs/DETAILS.md` to the newest stamps, and stops the commit if either file has unstaged edits, so those edits are never committed by accident

Each output is handled on its own, so the newest analysis report may come from a different run than the newest plots. Enable the hook once per clone with `git config core.hooksPath .githooks`. It runs with `.venv`'s Python if present, otherwise `python`.

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
#    Writes report/main_results_full_<stamp>.json and report/main_summary_<stamp>.csv
python main.py

# 2. Optimizer / network-depth / learning-rate comparisons.
#    Writes stamped plots and short analyses to report/comparisons/
python -m src.comparisons

# 3. Aggregate analysis.
#    Writes report/analysis_<stamp>.txt
python -m src.analysis
```

Steps 2 and 3 read the newest `report/main_results_full_*.json` and fail if none exists. To analyse a specific run instead, pass its path as the first argument, e.g. `python -m src.analysis report/main_results_full_20261007-112904.json`. The repository already includes a results file, so you can run them immediately. Set `MPLBACKEND=Agg` to run step 2 without plot windows opening.

### Tests

```bash
pip install pytest
python -m pytest
```

- `tests/test_gradients.py` compares every analytic gradient against a central finite-difference estimate: each activation, both losses, and the weight and bias gradients of whole A1- and A2-style networks, with and without bias terms and with each initialization scheme.
- `tests/test_layers.py` checks the Dense layer's bias (forward shift, batch-summed gradient), that each initialization scheme has the right standard deviation and that the default reproduces the original weights exactly, that every optimizer updates the bias as well as the weights, that early-stopping checkpoints restore the bias, and that the config loader rejects invalid `use_bias` / `init` values.
- `tests/test_batchnorm.py` checks batch normalization: training mode normalizes over the batch and updates the running statistics; evaluation mode uses them and gives a single sample the same output alone as inside a batch; its gradients (input, gamma, beta) match finite differences in both modes. It also checks the network's `train()` / `eval()` switch, that checkpoints restore the running statistics, and that the trainer stops and flags a diverging run.
- `tests/test_dropout.py` checks inverted dropout: in training mode it drops about `rate` of the values with a new mask per batch and scales the survivors so the expected value is unchanged; backward reuses the same mask; evaluation mode is the identity. It also checks that dropout sits after the activation, doesn't change the initial weights, is reproducible from the seed, and that the config loader rejects invalid rates and dropout on the output layer.
- `tests/test_utils.py` checks the run-stamp helpers used for output file names.
- `tests/test_seeds.py` checks that the split and the weight initialization follow the seed, the constant-prediction baselines, the `seeds` config validation, and the multi-seed analysis helpers (model selection by validation loss, error removed, mean ± std).
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
- `use_bias` per layer (optional, default `false`): adds a trainable bias vector, so the layer computes X·W + b. The bias starts at zero.
- `init` per layer (optional, default `normal`): the weight initialization scheme.
  - `normal`: N(0, 0.1²), the original fixed scale
  - `he`: N(0, 2 / fan_in), designed for ReLU
  - `xavier`: N(0, 2 / (fan_in + fan_out)), designed for sigmoid and tanh

  The defaults reproduce the original bias-free layer exactly, including the random weights drawn, so adding these keys to a new architecture never changes the results of the existing ones.

- `batch_norm` per layer (optional, default `false`): inserts a batch-normalization layer between the Dense layer and its activation (Dense → BatchNorm → activation). It draws no random numbers, so it doesn't change any Dense layer's initial weights.

- `dropout` per layer (optional, default `0`): adds inverted dropout with this rate after the layer's activation (Dense → (BatchNorm) → activation → Dropout). It must be in [0, 1) and isn't allowed on the output layer. Dropout masks come from a separate seeded generator, so adding dropout doesn't change any layer's initial weights.

The configs define eight architectures:
- the baselines `A1` (1 hidden sigmoid layer) and `A2` (3 hidden ReLU layers)
- four variants of A2: `A2-bias` adds bias terms, `A2-he` uses He initialization, `A2-bias-he` does both, and `A2-bn` adds batch normalization to the three hidden layers. `A2-bn` has no Dense biases, because batch norm's β takes over their role.
- two dropout versions: `A1-dropout` and `A2-bn-dropout`, which add dropout 0.2 after every hidden layer of `A1` and `A2-bn`

#### Training and evaluation mode

Batch norm behaves differently while learning and while being measured, so the network has `train()` and `eval()` methods that set a `training` flag on every layer. The trainer uses training mode for the mini-batch updates and evaluation mode for validation, `evaluate()` and `predict()`, and leaves the network in evaluation mode after `fit()`. In training mode, batch norm normalizes with the current batch's statistics and updates running averages of them. In evaluation mode it uses the running averages, so a prediction depends only on its own input. Dense and activation layers ignore the flag.

#### Per-run seed and constant-prediction baseline

Each run records its `seed` and a `baseline_test_metric`: the test metric of a model that ignores the features. For classification that model always predicts the training class proportion; for regression, the training mean. Because it is computed from that seed's split, "error removed" (1 − test / baseline; R² for regression, McFadden's pseudo-R² for classification) can be reported for every seed.

#### Training metric for the generalization gap

Each run also records `train_metric`: the BCE or MSE on the training set, measured in evaluation mode on the restored best model (`Trainer.score`). The training loss recorded during `fit()` can't be used for this, because it's measured in training mode and, with dropout, includes the dropout noise. `test_metric − train_metric` is the generalization gap: a large positive gap means overfitting.

#### Divergence

If the training or validation loss of an epoch is NaN or infinite, the trainer stops immediately and sets `diverged` in the run's results. If an earlier epoch was finite, early stopping still restores that checkpoint, so a diverged run can report a finite but meaningless test metric. The analysis therefore uses the `diverged` flag rather than the metric.

**Experiment sweep values**
- `optimizers`: `sgd`, `momentum` (aliases `momentumsgd`, `momentum_sgd`), `adabelief`.
- `learning_rates`: positive numbers, typically 0.1, 0.01, or 0.001.
- `batch_sizes`: positive integers, bounded by the size of the training split.
- `seeds` (optional, default `[42]`): the whole sweep is repeated once per seed. Each seed sets the train / validation / test split, the weight initialization, the dropout masks and the per-epoch shuffling. Both configs use `[42, 43, 44, 45, 46]`. Seed 42 reproduces the original single-seed results exactly, and EDA plots are only made for the first seed.

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

1. **Layer types.** Each layer has a `"type"` field, but only `dense` is implemented; batch norm is attached to a dense layer with `batch_norm`. Batch norm and dropout are attached to a dense layer with `batch_norm` and `dropout`. Convolutional or recurrent layers would need code changes first. (Optimizers and checkpoints already handle any number of parameters per layer, through `get_params()` / `get_grads()`, and non-trainable state through `get_buffers()`.)
2. **Optimizer internals.** Only the learning rate comes from JSON. Momentum uses β = 0.9, and AdaBelief uses β₁ = 0.9, β₂ = 0.999, ε = 1e-8, all fixed in `src/optimizers.py` (`get_optimizer`).
3. **Evaluation metrics.** The task type determines them: BCE for classification and MSE for regression, matching the training losses.
4. **The default random seed.** `RANDOM_SEED` in `src/utils.py` (42) is used when a config lists no `seeds`, and it's the seed that the single-seed analysis sections and the comparison plots use.
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
| Optimizers | problem, architecture, learning rate, batch size | optimizer (SGD → Momentum → AdaBelief, one subplot each) | `optimizers_*_<stamp>.png` |
| Network depth | problem, optimizer, learning rate, batch size | architecture (A1 vs A2) | `depth_*_<stamp>.png`, `depth_analysis_<stamp>.txt` |
| Learning rate | problem, architecture, optimizer, batch size | learning rate (0.1 vs 0.001) | `learning_rate_*_<stamp>.png`, `learning_rate_analysis_<stamp>.txt` |
| A2 variants | problem, optimizer, learning rate, batch size | A2 → A2-bias → A2-he → A2-bias-he → A2-bn, one subplot each, log-scale loss | `a2_variants_*_<stamp>.png` |

There are two A2 variant plots, both regression at LR 0.1, where the baseline A2 collapses:
- SGD, batch 16: shows all three failure modes (collapse to 0, collapse to the mean, divergence)
- momentum, batch 64: no variant diverges, so each fix's outcome is visible side by side

Both are skipped for results files that contain no variant runs.

The **dropout plot** (`dropout_regression_adabelief_lr01_bs16_<stamp>.png`) compares A1 with A1-dropout and A2-bn with A2-bn-dropout for regression, AdaBelief, LR 0.1 and batch 16, the setting of the best baseline regression run. With dropout, the plotted training loss is measured with dropout active, so it sits above the validation loss. That is dropout's expected signature, not a sign of a problem. The plot is skipped for results files without dropout runs.

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

All of the sections above use the baseline architectures (A1, A2) only, including the min-max normalization. Adding new architectures to the configs therefore leaves those numbers unchanged.

#### A2 variants

A separate section compares A2 with its bias, initialization and batch-norm variants:
- one summary table per task (runs, average convergence epoch, average best validation loss)
- a **collapse check**: the regression test MSE of the four runs that collapse in the baseline A2 (SGD and momentum, LR 0.1, batch 16 and 64), for each variant

Diverged runs (see "Divergence" above) are listed separately and left out of the averages. A test MSE of 602.17 means the network outputs a constant 0. One near 101.4 means it outputs a constant close to the mean target, which a dead network can still do through its output bias.

#### Dropout

A final section compares each architecture with its dropout version (A1 vs A1-dropout, A2-bn vs A2-bn-dropout), on runs matched by optimizer, learning rate and batch size; matches where either run diverged are left out. For each task it reports:
- the average training metric (`train_metric`, evaluation mode)
- the average test metric
- the average generalization gap (test − train)
- in how many matched runs dropout gave the lower test metric

The section is skipped for results files without dropout runs or without `train_metric`.

#### Multi-seed results

Every section above uses seed 42 only, so its numbers stay comparable with single-seed results files. When the results contain several seeds, a final **Multi-Seed Results** section reports mean ± sample standard deviation across seeds:

- **Selected model per seed**: for each task and seed, the run with the lowest validation loss (never the test score). It's reported once among the baseline architectures A1 / A2 and once among all architectures, with test metric, constant baseline and error removed.
- **Architectures across seeds**: each architecture's best run per seed (again chosen by validation loss), its average convergence epoch, and how many of its runs were no better than the constant baseline or diverged.
- **Optimizer × learning rate** (A1 / A2): the average test metric per seed for every optimizer and learning rate.
- **Collapse check across seeds**: for the regression settings where the baseline A2 collapses, how many runs of each A2 variant learned, were no better than predicting the mean, or diverged.
- **Dropout across seeds**: the dropout comparison pooled over all seeds, with runs matched on seed as well.

Diverged runs are excluded from model selection and averages throughout. The comparison plots also use seed 42 only, because each shows one run's loss curves.

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

The report (`report/analysis_<stamp>.txt`) contains a method section, supporting tables per task and combined, direct answers to the three questions, and, when the results include them, the A2 variant section.

### In short

- `comparisons.py` explains individual matched experiments with plots and short text analyses.
- `analysis.py` answers the overall questions using averages across all runs.
