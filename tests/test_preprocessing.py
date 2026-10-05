"""
Tests for the NumPy-only train / validation / test split.
"""

import numpy as np
import pandas as pd

from src.preprocessing import train_val_test_split


def make_frame(n=1000, positive_rate=0.3, seed=0):
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "x": rng.normal(size=n),
        "class": (rng.uniform(size=n) < positive_rate).astype(int),
    })


def test_split_sizes_and_disjoint_cover():
    df = make_frame()
    train, val, test = train_val_test_split(df, val_size=0.2, test_size=0.2)

    assert (len(train), len(val), len(test)) == (600, 200, 200)

    train_idx, val_idx, test_idx = set(train.index), set(val.index), set(test.index)
    assert not (train_idx & val_idx) and not (train_idx & test_idx) and not (val_idx & test_idx)
    assert train_idx | val_idx | test_idx == set(df.index)


def test_stratified_split_preserves_class_balance():
    df = make_frame()
    overall = df["class"].mean()
    train, val, test = train_val_test_split(df, stratify_col="class")

    for split in (train, val, test):
        assert abs(split["class"].mean() - overall) < 0.01


def test_split_is_reproducible_and_seed_dependent():
    df = make_frame()
    first = train_val_test_split(df, random_seed=1)
    again = train_val_test_split(df, random_seed=1)
    other = train_val_test_split(df, random_seed=2)

    for a, b in zip(first, again):
        assert a.index.equals(b.index)
    assert not first[2].index.equals(other[2].index)
