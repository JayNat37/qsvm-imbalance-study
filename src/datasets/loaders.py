""" Loads datasets and prepares them in a format suitable for the experiment pipeline."""
from __future__ import annotations

from typing import Tuple
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

from src.datasets.config import DatasetConfig
from sklearn.datasets import make_moons


def _as_numpy(X, y) -> Tuple[np.ndarray, np.ndarray]:
    """Convert dataset features and labels to NumPy arrays."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    return X, y

def _load_banknote(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load the banknote dataset from file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Banknote file not found: {p.resolve()}")

    try:
        data = np.genfromtxt(p, delimiter=",", dtype=float)
        if data.ndim == 1:
            raise ValueError("Parsed 1D array")
    except Exception:
        data = np.genfromtxt(p, delimiter=None, dtype=float)

    if data.shape[1] != 5:
        raise ValueError(f"Expected 5 columns (4 features + label), got {data.shape[1]}")

    X = data[:, :4].astype(float)
    y = data[:, 4].astype(int)


    unique = set(np.unique(y).tolist())
    if not unique.issubset({0, 1}):
        vals = sorted(list(unique))
        mapping = {vals[0]: 0, vals[1]: 1}
        y = np.array([mapping[v] for v in y], dtype=int)

    return X, y


def load_dataset_raw(cfg: DatasetConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a dataset specified by the configuration and return 
    raw features and labels.
    """
    name = cfg.name.lower()
    p = cfg.params or {}

    # Load the dataset based on the name
    if name == "banknote":
        path = p.get("path")
        if not path:
            raise ValueError("banknote requires dataset.params.path")
        X, y = _load_banknote(path)
    
    elif name == "make_moons":
        n_samples = int(p.get("n_samples", 300))
        noise = float(p.get("noise", 0.2))
        random_seed = int(p.get("random_seed", 42))

        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_seed)
        X, y = _as_numpy(X, y)

    else:
        raise ValueError(f"Unknown dataset: {cfg.name}")

    # Optionally subsample the dataset
    if cfg.max_samples is not None and cfg.max_samples < len(y):
        X, _, y, _ = train_test_split(
            X,
            y,
            train_size=cfg.max_samples,
            stratify=y,
            random_state=cfg.split_seed,
        )

    return _as_numpy(X, y)

    
