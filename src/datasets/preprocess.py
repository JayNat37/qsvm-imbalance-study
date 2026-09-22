""" Applies preprocessing steps such as train-test splitting, scaling, and optional dimensionality reduction. """
from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

from src.datasets.config import DatasetConfig

# Fixed test-set size/imbalance used whenever training-set imbalance subsampling
# is triggered, so all imbalance levels are evaluated against the same balanced test set.
_FIXED_TEST_SIZE = 200
_FIXED_TEST_IMBALANCE_FRACTION = 0.5


def subsample_fixed_imbalance(
    X: np.ndarray,
    y: np.ndarray,
    n: int,
    frac: float,
    seed: int,
    cap_to_available: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Draw a fixed-size subset of (X, y) with a specified class-0 fraction.

    Samples round(n * frac) points from class 0 and the remainder from
    class 1, concatenates them, and shuffles the result.

    Args:
        X: Feature matrix.
        y: Binary label vector (values 0 and 1).
        n: Total number of samples to draw.
        frac: Fraction of the n samples that should be class 0.
        seed: Random seed controlling the draw and shuffle.
        cap_to_available: If True, silently reduce n0/n1 to whatever is
            available instead of raising, so the caller can inspect the
            actual counts drawn (X_sub, y_sub may then total less than n).

    Returns:
        A tuple of (X_sub, y_sub) with n samples (or fewer, if capped).

    Raises:
        ValueError: If either class does not have enough samples to draw
            from and cap_to_available is False.
    """
    rng = np.random.RandomState(seed)

    idx0 = np.flatnonzero(y == 0)
    idx1 = np.flatnonzero(y == 1)

    n0 = int(round(n * frac))
    n1 = n - n0

    if cap_to_available:
        n0 = min(n0, idx0.size)
        n1 = min(n1, idx1.size)
    elif n0 > idx0.size or n1 > idx1.size:
        raise ValueError(
            f"Not enough samples to draw n0={n0} from class 0 ({idx0.size} available) "
            f"and n1={n1} from class 1 ({idx1.size} available)"
        )

    sel0 = rng.choice(idx0, size=n0, replace=False)
    sel1 = rng.choice(idx1, size=n1, replace=False)

    sel = np.concatenate([sel0, sel1])
    rng.shuffle(sel)

    return X[sel], y[sel]


def preprocess(
        X: np.ndarray,
        y: np.ndarray,
        *,
        cfg: DatasetConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split a dataset into training and test sets and apply preprocessing.

    The preprocessing pipeline optionally applies feature scaling and PCA,
    with transformations fitted on the training data only.

    Args:
        X: Feature matrix.
        y: Label vector.
        cfg: Dataset preprocessing configuration.

    Returns:
        A tuple of (X_train, X_test, y_train, y_test).

    Raises:
        ValueError: If the input arrays have invalid shapes or if the PCA
        configuration is invalid.
    """
    
    # Data Validation
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).ravel()

    if X.ndim != 2:
        raise ValueError(f"X must be 2D. Got {X.shape}")
    if y.ndim != 1:
        raise ValueError(f"y must be 1D. Got {y.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have same number of samples")
    
    # Split
    strat = y if _can_stratify(y) else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.split_seed,
        stratify=strat,
    )

    # Scaling
    if cfg.scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    # PCA
    if cfg.pca_components is not None:
        if cfg.pca_components <= 0:
            raise ValueError("pca_components must be positive or None")
        if cfg.pca_components > X_train.shape[1]:
            raise ValueError(
                f"pca_components={cfg.pca_components} exceeds number of features={X_train.shape[1]}"
            )

        pca = PCA(n_components=cfg.pca_components, random_state=cfg.split_seed)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

    # Optional MinMax scaling to [0, 1], fit on the training data only
    if cfg.minmax_scale:
        minmax = MinMaxScaler(feature_range=(0, 1))
        X_train = minmax.fit_transform(X_train)
        X_test = minmax.transform(X_test)

    # Optional fixed-size class-imbalance subsampling of the training set,
    # paired with a fixed balanced test set so imbalance levels stay comparable.
    params = cfg.params or {}
    fixed_size = params.get("fixed_size")
    imbalance_fraction = params.get("imbalance_fraction")

    if fixed_size is not None and imbalance_fraction is not None:
        train_seed = int(params.get("random_seed", cfg.split_seed))
        X_train, y_train = subsample_fixed_imbalance(
            X_train, y_train, n=int(fixed_size), frac=float(imbalance_fraction), seed=train_seed,
            cap_to_available=cfg.cap_imbalance_to_available,
        )
        X_test, y_test = subsample_fixed_imbalance(
            X_test, y_test, n=_FIXED_TEST_SIZE, frac=_FIXED_TEST_IMBALANCE_FRACTION, seed=int(cfg.split_seed),
            cap_to_available=cfg.cap_imbalance_to_available,
        )

    return (
        np.asarray(X_train, dtype=float),
        np.asarray(X_test, dtype=float),
        np.asarray(y_train),
        np.asarray(y_test),
    )

def _can_stratify(y: np.ndarray) -> bool:
    _, counts = np.unique(y, return_counts=True)
    return counts.size >= 2 and np.all(counts >= 2)