"""
Hyperparameter grid search for the classical SVM and QSVM, used by the interactive
UI's "Tune hyperparameters" feature. Mirrors Josiah Chan's tuning methodology:
grid search against a held-out validation split, using the exact statevector
kernel for the QSVM search (fast), leaving the actual test set untouched.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from dataclasses import replace

from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from src.datasets.config import DatasetConfig
from src.datasets.preprocess import preprocess
from src.qsvm.config import NoiseConfig, QSVMConfig
from src.qsvm.kernel import QuantumKernel


def make_tuning_split(
    X_train: np.ndarray,
    y_train: np.ndarray,
    val_fraction: float = 0.25,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Carve a validation split out of the training pool, purely for tuning."""
    counts = np.bincount(y_train, minlength=2)
    stratify = y_train if counts.min() >= 2 else None
    return train_test_split(
        X_train, y_train, test_size=val_fraction, random_state=seed, stratify=stratify,
    )


def make_tuning_data(
    dataset_cfg: DatasetConfig,
    X_raw: np.ndarray,
    y_raw: np.ndarray,
    val_per_class: int = 30,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build (X_tr, X_val, y_tr, y_val) for tuning.

    With a fixed imbalance (fixed_size + imbalance_fraction) the training subsample can hold only
    one or two minority examples, so splitting a validation set out of it leaves one side without
    the minority class. Instead the training subsample is drawn exactly as the real run draws it,
    and a separate balanced validation set is drawn from the unused part of the training pool
    (disjoint from both the training subsample and the test set). Without a fixed imbalance this
    falls back to an ordinary validation split of the training pool.
    """
    params = dataset_cfg.params or {}
    fixed_size, frac = params.get("fixed_size"), params.get("imbalance_fraction")
    if fixed_size is None or frac is None:
        X_pool, _, y_pool, _ = preprocess(X_raw, y_raw, cfg=dataset_cfg)
        return make_tuning_split(X_pool, y_pool, seed=int(dataset_cfg.split_seed))

    pool_params = {k: v for k, v in params.items() if k not in ("fixed_size", "imbalance_fraction")}
    X_pool, _, y_pool, _ = preprocess(X_raw, y_raw, cfg=replace(dataset_cfg, params=pool_params))

    idx0, idx1 = np.flatnonzero(y_pool == 0), np.flatnonzero(y_pool == 1)
    n0 = int(round(int(fixed_size) * float(frac)))
    n1 = int(fixed_size) - n0
    if dataset_cfg.cap_imbalance_to_available:
        n0, n1 = min(n0, idx0.size), min(n1, idx1.size)
    rng = np.random.RandomState(int(params.get("random_seed", dataset_cfg.split_seed)))
    sel0, sel1 = rng.choice(idx0, size=n0, replace=False), rng.choice(idx1, size=n1, replace=False)
    train_idx = np.concatenate([sel0, sel1])

    left0, left1 = np.setdiff1d(idx0, sel0), np.setdiff1d(idx1, sel1)
    k = min(val_per_class, left0.size, left1.size)
    val_rng = np.random.RandomState(int(dataset_cfg.split_seed) + 10_000)
    val_idx = np.concatenate([val_rng.choice(left0, size=k, replace=False), val_rng.choice(left1, size=k, replace=False)])
    return X_pool[train_idx], X_pool[val_idx], y_pool[train_idx], y_pool[val_idx]


def tune_classical(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    c_grid: Sequence[float] = (0.1, 1.0, 10.0, 100.0),
    gamma_grid: Sequence[Any] = ("scale", 0.001, 0.01, 0.1, 1.0),
    class_weight_grid: Sequence[Optional[str]] = (None, "balanced"),
) -> Dict[str, Any]:
    """Grid search classical SVM (C, gamma, class_weight) against a validation split."""
    best: Optional[Dict[str, Any]] = None
    for C in c_grid:
        for gamma in gamma_grid:
            for class_weight in class_weight_grid:
                clf = SVC(kernel="rbf", C=C, gamma=gamma, class_weight=class_weight)
                clf.fit(X_tr, y_tr)
                val_accuracy = balanced_accuracy_score(y_val, clf.predict(X_val))
                if best is None or val_accuracy > best["val_accuracy"]:
                    best = {"C": C, "gamma": gamma, "class_weight": class_weight, "val_accuracy": val_accuracy}
    return best


def tune_qsvm(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_map: str = "zz",
    reps_grid: Sequence[int] = (1, 2, 3),
    entanglement_grid: Sequence[str] = ("linear", "full"),
    c_grid: Sequence[float] = (0.1, 1.0, 10.0, 100.0),
    class_weight_grid: Sequence[Optional[str]] = (None, "balanced"),
    bandwidth_grid: Sequence[float] = (1.0,),
    scale_to_pi: bool = False,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Grid search QSVM (bandwidth, reps, entanglement, C, class_weight) against a validation split.

    bandwidth multiplies the inputs before they enter the circuit (the quantum analogue of
    an RBF gamma). Small values make the ZZ kernel smooth and RBF-like; large values make it
    concentrate towards the identity matrix.

    Uses the exact statevector kernel for speed, matching how a production
    tuner would search before the slower noisy_aer kernel is used for the
    actual evaluation run. class_weight is applied on the precomputed-kernel
    SVC exactly as it would be on any other kernel - it's often a no-op on
    trivially separable tasks (the unconstrained solution never hits the C
    bound it would relax) but a real lever on harder ones.
    """
    best: Optional[Dict[str, Any]] = None
    for bandwidth in bandwidth_grid:
        X_tr_s, X_val_s = X_tr * bandwidth, X_val * bandwidth
        for reps in reps_grid:
            for entanglement in entanglement_grid:
                qcfg = QSVMConfig(
                    feature_map=feature_map, reps=reps, entanglement=entanglement, mode="statevector",
                    shots=None, seed=seed, scale_to_pi=scale_to_pi, C=1.0, noise=NoiseConfig(),
                )
                kernel = QuantumKernel(qcfg)
                K_tr = kernel.compute(X_tr_s, X_tr_s)
                K_val = kernel.compute(X_val_s, X_tr_s)
                for C in c_grid:
                    for class_weight in class_weight_grid:
                        clf = SVC(kernel="precomputed", C=C, class_weight=class_weight)
                        clf.fit(K_tr, y_tr)
                        val_accuracy = balanced_accuracy_score(y_val, clf.predict(K_val))
                        if best is None or val_accuracy > best["val_accuracy"]:
                            best = {
                                "bandwidth": bandwidth, "reps": reps, "entanglement": entanglement, "C": C,
                                "class_weight": class_weight, "val_accuracy": val_accuracy,
                            }
    return best
