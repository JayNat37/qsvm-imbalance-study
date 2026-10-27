"""
Hyperparameter grid search for the classical SVM and QSVM, used by the interactive
UI's "Tune hyperparameters" feature. Mirrors Josiah Chan's tuning methodology:
grid search against a held-out validation split, using the exact statevector
kernel for the QSVM search (fast), leaving the actual test set untouched.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

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
    scale_to_pi: bool = False,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Grid search QSVM (reps, entanglement, C, class_weight) against a validation split.

    Uses the exact statevector kernel for speed, matching how a production
    tuner would search before the slower noisy_aer kernel is used for the
    actual evaluation run. class_weight is applied on the precomputed-kernel
    SVC exactly as it would be on any other kernel - it's often a no-op on
    trivially separable tasks (the unconstrained solution never hits the C
    bound it would relax) but a real lever on harder ones.
    """
    best: Optional[Dict[str, Any]] = None
    for reps in reps_grid:
        for entanglement in entanglement_grid:
            qcfg = QSVMConfig(
                feature_map=feature_map, reps=reps, entanglement=entanglement, mode="statevector",
                shots=None, seed=seed, scale_to_pi=scale_to_pi, C=1.0, noise=NoiseConfig(),
            )
            kernel = QuantumKernel(qcfg)
            K_tr = kernel.compute(X_tr, X_tr)
            K_val = kernel.compute(X_val, X_tr)
            for C in c_grid:
                for class_weight in class_weight_grid:
                    clf = SVC(kernel="precomputed", C=C, class_weight=class_weight)
                    clf.fit(K_tr, y_tr)
                    val_accuracy = balanced_accuracy_score(y_val, clf.predict(K_val))
                    if best is None or val_accuracy > best["val_accuracy"]:
                        best = {
                            "reps": reps, "entanglement": entanglement, "C": C,
                            "class_weight": class_weight, "val_accuracy": val_accuracy,
                        }
    return best
