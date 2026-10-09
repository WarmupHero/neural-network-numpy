"""
Comparisons of the NumPy network with standard ML libraries.

The NumPy implementation in ``nn_numpy.nn`` and
``nn_numpy.modeling`` uses NumPy only. This subpackage trains models
from other libraries on exactly the same data splits and scores them with
the same metric functions, so the results can be compared directly.

- ``data``: the same train / validation / test splits as the main pipeline
- ``sklearn_models``: scikit-learn models and their hyperparameter grids
- ``run``: trains every library model and saves the results
- ``report``: compares the library results with the NumPy network's

The libraries are optional dependencies. Install them with
``pip install -e ".[benchmarks]"`` (or ``make benchmark-requirements``).
"""
