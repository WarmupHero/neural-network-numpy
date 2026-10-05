# Neural Network from Scratch in NumPy

A feed-forward neural network built with **NumPy only**, with no TensorFlow, PyTorch or scikit-learn. It includes the forward pass, backpropagation, three optimizers, and early stopping. It's applied to two complete examples:

- **Binary classification:** detecting forged banknotes (UCI Banknote Authentication)
- **Regression:** predicting a building's heating load (UCI Energy Efficiency)

Both are benchmarked across a 48-run hyperparameter grid.

![Python](https://img.shields.io/badge/python-3.12-blue)
![NumPy only](https://img.shields.io/badge/built%20with-NumPy%20only-informational)
[![tests](https://github.com/WarmupHero/neural-network-from-scratch/actions/workflows/tests.yml/badge.svg)](https://github.com/WarmupHero/neural-network-from-scratch/actions/workflows/tests.yml)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

---

## Highlights

- **Backprop from first principles.** Dense layers, ReLU / sigmoid / tanh / linear activations, and BCE / MSE losses (also used as the evaluation metrics), each with a hand-derived backward pass.
- **Three optimizers.** SGD, SGD with momentum, and [AdaBelief](https://arxiv.org/abs/2010.07468) with bias correction.
- **Training loop.** Mini-batching with per-epoch shuffling, validation tracking, early stopping with patience and `min_delta`, and restoration of the best weights.
- **Config-driven experiments.** Architectures, optimizers, learning rates, batch sizes, and early-stopping settings are defined in JSON. One command runs the whole sweep.
- **Verified correctness.** Every analytic gradient (activations, losses, and whole networks) is checked against central finite differences with `pytest`.
- **Leak-free preprocessing, also in NumPy.** Duplicates are removed, the data gets a 60 / 20 / 20 train / validation / test split (stratified by class for classification), and the standard scaler is fit on the training split only. No scikit-learn, no framework.

## Results

Models are trained and evaluated with the same objective: **binary cross-entropy (BCE)** for classification and **mean squared error (MSE)** for regression. Lower is better for both.

For each task, the reported model is the one of 24 runs with the **lowest validation loss**. Its test score is reported once, and the test set is never used to choose it.

| Task | Selected run | Test metric | Baseline (ignores the features) | Error removed vs. baseline |
|---|---|---|---|---|
| Classification: Banknote Authentication (1,348 unique samples, 4 features) | A1 · AdaBelief · LR 0.1 · batch 16 | **BCE 0.0002** | 0.689 (always predict the class proportions) | **>99.9%** |
| Regression: Energy Efficiency → Heating Load (768 samples, 8 features) | A1 · AdaBelief · LR 0.1 · batch 16 | **MSE 0.37** (RMSE 0.61 kWh/m²) | 101.3 (always predict the mean) | **99.6%** (R² = 0.996) |

**Error removed** = 1 − test metric ÷ baseline metric. For regression this is R². For classification it's McFadden's pseudo-R². Raw BCE and MSE are on different scales: BCE is in nats, and MSE is in squared target units, with Heating_Load spanning 6–43. A small BCE next to a larger MSE therefore says nothing about which model is better. Relative to their baselines, both tasks are almost fully solved.

The typical run is strong too. The median run removes 96% of the baseline error on classification and 92% on regression, and 15 of 24 and 18 of 24 runs clear 90%. The weaker runs are the failure modes described below.

*A1* = 1 hidden layer (32 units, sigmoid). *A2* = 3 hidden layers (32 units each, ReLU). Test sets are small (270 and 154 samples) and come from one split and one seed, so small differences between top runs are within noise. Full per-run results are in [`main_summary.csv`](report/main_summary.csv).

### Key findings

- **AdaBelief is the most robust optimizer.** Averaged over each optimizer's 8 classification runs, AdaBelief reaches a test BCE of **0.024**, versus about **0.352** for SGD and momentum. It also produces the selected model on both tasks.
- **Learning rate matters most for plain gradient descent.** At LR 0.001, SGD and momentum barely learn: test BCE stays at 0.66–0.69, essentially the no-information baseline of 0.689. Dropping the LR from 0.1 to 0.001 raises their mean classification BCE from about 0.021 to about 0.683. AdaBelief stays low at both rates (0.034 → 0.014).
- **Depth speeds up convergence but adds instability.** The 3-layer ReLU network converges in about 49 epochs on average versus about 85 for the shallow one. On regression at LR 0.1, however, all four SGD and momentum runs of A2 collapse. The ReLU units die and the output becomes zero: each run scores exactly 602.17, the test MSE of always predicting 0, and early stopping ends it at epoch 30. Both AdaBelief runs survive (MSE 2.73 and 2.13).

<p align="center">
  <img src="report/comparisons/optimizers_classification_A1_lr01_bs16.png" width="85%" alt="Training and validation loss for SGD, Momentum and AdaBelief on banknote classification">
</p>
<p align="center">
  <img src="report/comparisons/depth_classification_sgd_lr01_bs16.png" width="48%" alt="Depth comparison: A1 vs A2">
  <img src="report/comparisons/learning_rate_classification_A1_sgd_bs16.png" width="48%" alt="Learning-rate comparison: 0.1 vs 0.001">
</p>

More plots, including EDA, scaling comparisons, and correlation heatmaps, are in [`report/`](report/). The written analysis is in [`analysis.txt`](report/analysis.txt).

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt

python main.py                    # download data, preprocess, run all 48 experiments
python -m src.comparisons         # optimizer / depth / learning-rate plots
python -m src.analysis            # summary report -> report/analysis.txt

pip install pytest
python -m pytest                  # gradient checks, split checks, training smoke tests
```

## Project structure

```
neural-network-from-scratch/
├── main.py                 # runs the full experiment grid
├── configs/                # JSON experiment definitions (one per task)
├── src/
│   ├── layers.py           # Dense layer
│   ├── activations.py      # ReLU, Sigmoid, Tanh, Linear
│   ├── losses.py           # BCE, MSE
│   ├── optimizers.py       # SGD, Momentum, AdaBelief
│   ├── network.py          # model assembly, forward / backward
│   ├── train.py            # training loop, early stopping, evaluation
│   ├── preprocessing.py    # cleaning, splitting, scaling, EDA
│   ├── comparisons.py      # optimizer / depth / learning-rate plots
│   └── analysis.py         # aggregate analysis report
├── tests/                  # pytest: gradient checks, split checks, training smoke tests
├── datasets/               # cached UCI CSVs
├── report/                 # generated results and figures
└── docs/DETAILS.md         # detailed documentation
```

See the [detailed project documentation](docs/DETAILS.md) for the module-by-module pipeline, everything the JSON configs can and cannot control, and how convergence is measured.

## Demo

Watch the full pipeline run on YouTube: `main.py` preprocesses the data and trains all 48 experiments, then `analysis.py` summarizes the results.

[![Demo video: Neural Network from Scratch in NumPy](https://img.youtube.com/vi/FIBYV5jaxqo/hqdefault.jpg)](https://youtu.be/FIBYV5jaxqo)

*The recording predates the latest changes, so its console output shows accuracy / MAE and the old folder layout. Current results are in [Results](#results).*

## Datasets & attribution

Both datasets come from the UCI Machine Learning Repository and are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The cached copies in `datasets/` are unmodified except that the Energy Efficiency columns are renamed to descriptive names.

- Lohweg, V. (2012). *Banknote Authentication* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C55P57
- Tsanas, A. & Xifara, A. (2012). *Energy Efficiency* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C51307

## License

The code is released under the [MIT License](LICENSE). The datasets remain under their original CC BY 4.0 licenses (see above).
