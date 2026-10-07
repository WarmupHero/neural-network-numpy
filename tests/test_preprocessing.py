"""
Tests for the NumPy-only train / validation / test split.
"""

import numpy as np
import pandas as pd

from nn_from_scratch.features import train_val_test_split


def make_frame(n=1000, positive_rate=0.3, seed=0):
    """
    Build a small synthetic dataframe with one feature and a binary class column.

    Parameters
    ----------
    n : int, default=1000
        Number of rows.
    positive_rate : float, default=0.3
        Probability that a row's "class" is 1.
    seed : int, default=0
        Seed for the random generator.

    Returns
    -------
    pandas.DataFrame of shape (n, 2)
        Columns "x" (float64, standard normal) and "class" (int, 0 or 1),
        with the default RangeIndex 0..n-1.

    Notes
    -----
    Processing:
    1. Draw "x" from a standard normal distribution.
    2. Set "class" to 1 where a uniform draw is below positive_rate.
    """
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "x": rng.normal(size=n),
        "class": (rng.uniform(size=n) < positive_rate).astype(int),
    })


def test_split_sizes_and_disjoint_cover():
    """
    The split has the requested sizes and every row lands in exactly one split.

    Notes
    -----
    Splitting 1000 rows with val_size=0.2 and test_size=0.2 must give
    600 / 200 / 200 rows. The three index sets must not overlap, and
    together they must equal the original index.
    """
    df = make_frame()
    train, val, test = train_val_test_split(df, val_size=0.2, test_size=0.2)

    assert (len(train), len(val), len(test)) == (600, 200, 200)

    train_idx, val_idx, test_idx = set(train.index), set(val.index), set(test.index)
    assert not (train_idx & val_idx) and not (train_idx & test_idx) and not (val_idx & test_idx)
    assert train_idx | val_idx | test_idx == set(df.index)


def test_stratified_split_preserves_class_balance():
    """
    A stratified split keeps the class proportion in every split.

    Notes
    -----
    With stratify_col="class", the fraction of class 1 in the train,
    validation and test splits must each be within 0.01 of the fraction
    in the full dataframe.
    """
    df = make_frame()
    overall = df["class"].mean()
    train, val, test = train_val_test_split(df, stratify_col="class")

    for split in (train, val, test):
        assert abs(split["class"].mean() - overall) < 0.01


def test_split_is_reproducible_and_seed_dependent():
    """
    The same seed gives the same split, and a different seed gives a different one.

    Notes
    -----
    Two splits with random_seed=1 must have identical indices (in the
    same order) for all three parts. The test split with random_seed=2
    must differ from the one with random_seed=1.
    """
    df = make_frame()
    first = train_val_test_split(df, random_seed=1)
    again = train_val_test_split(df, random_seed=1)
    other = train_val_test_split(df, random_seed=2)

    for a, b in zip(first, again):
        assert a.index.equals(b.index)
    assert not first[2].index.equals(other[2].index)
