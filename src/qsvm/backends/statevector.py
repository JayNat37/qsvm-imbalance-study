""" Implements noiseless quantum kernel evaluation using a statevector-based fidelity kernel."""
from __future__ import annotations
import numpy as np
from qiskit_machine_learning.kernels import FidelityStatevectorKernel
from src.qsvm.feature_maps import build_feature_map
from src.qsvm.config import QSVMConfig


def compute_kernel(X1: np.ndarray, X2: np.ndarray, cfg: QSVMConfig) -> np.ndarray:
    # Build the feature map using the input dimension
    feature_map = build_feature_map(X1.shape[1], cfg)

    # Create a noiseless statevector-based quantum kernel
    kernel = FidelityStatevectorKernel(feature_map=feature_map)

    # Evaluate the kernel matrix between X1 and X2
    return kernel.evaluate(X1, X2)