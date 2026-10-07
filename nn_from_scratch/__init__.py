"""
Neural network from scratch in NumPy.

The project's source package, laid out as in Cookiecutter Data Science:

- ``config``: project paths, the default random seed, and output-file stamping
- ``config_loader``: loads and validates the JSON experiment configs in ``configs/``
- ``dataset``: downloads the raw UCI datasets into ``data/raw/``
- ``features``: cleaning, train / validation / test splitting, and scaling
- ``scalers``: the NumPy standard scaler used by ``features``
- ``plots``: exploratory (EDA) and preprocessing figures
- ``nn``: the neural-network library (layers, activations, losses, metrics,
  network assembly, optimizers)
- ``modeling``: the Trainer class (``modeling.trainer``) and the experiment
  sweep entry point (``modeling.train``)
- ``comparisons``: loss-curve comparison figures and short text analyses
- ``analysis``: the aggregate analysis report
"""
