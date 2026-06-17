""" Computes quantum kernel matrices and handles backend selection."""
from __future__ import annotations

import numpy as np

from .config import QSVMConfig
from .backends import statevector, noisy_aer


def _scale_to_pi(X: np.ndarray, ref_max_abs: np.ndarray) -> np.ndarray:
    """
    Scale each feature to the interval [-pi, pi] using reference maxima.

    Args:
        X: Input feature matrix.
        ref_max_abs: Per-feature maximum absolute values used for scaling.

    Returns:
        The scaled feature matrix.
    """
    denom = np.where(ref_max_abs > 0, ref_max_abs, 1.0)
    Xn = X / denom
    return np.clip(Xn, -1.0, 1.0) * np.pi


class QuantumKernel:
    def __init__(self, cfg: QSVMConfig):
        """
        Initialise the quantum kernel with a validated QSVM configuration.
        """
        cfg.validate()
        self.cfg = cfg

    def compute(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Compute the quantum kernel matrix between two input datasets.

        Args:
            X1: First feature matrix.
            X2: Second feature matrix.

        Returns:
            Kernel matrix of shape (len(X1), len(X2)).

        Raises:
            ValueError: If the configured QSVM mode is unsupported.
        """
        X1 = np.asarray(X1, dtype=float)
        X2 = np.asarray(X2, dtype=float)

        if self.cfg.scale_to_pi:
            # Use X2 as reference - typically training data - to ensure consistent scaling
            ref_max = np.max(np.abs(X2), axis=0)
            X1s = _scale_to_pi(X1, ref_max)
            X2s = _scale_to_pi(X2, ref_max)
        else:
            X1s, X2s = X1, X2

        if self.cfg.mode == "statevector":
            return statevector.compute_kernel(X1s, X2s, self.cfg)
        if self.cfg.mode == "noisy_aer":
            return noisy_aer.compute_kernel(X1s, X2s, self.cfg)
        
        raise ValueError(f"Unknown QSVM mode: {self.cfg.mode}")
