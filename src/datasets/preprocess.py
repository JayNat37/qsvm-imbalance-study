""" Applies preprocessing steps such as train-test splitting, scaling, and optional dimensionality reduction. """
from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from src.datasets.config import DatasetConfig


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

    return (
        np.asarray(X_train, dtype=float),
        np.asarray(X_test, dtype=float),
        np.asarray(y_train),
        np.asarray(y_test),
    )

def _can_stratify(y: np.ndarray) -> bool:
    _, counts = np.unique(y, return_counts=True)
    return counts.size >= 2 and np.all(counts >= 2)