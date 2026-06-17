""" Builds the quantum feature map circuits used to encode classical input data into quantum states."""
from __future__ import annotations
from src.qsvm.config import QSVMConfig
from qiskit.circuit.library import ZZFeatureMap, PauliFeatureMap


def build_feature_map(n_features: int, cfg: QSVMConfig):  
    """
    Construct the quantum feature map specified in the QSVM configuration.

    Args:
        n_features: Number of classical input features to encode.
        cfg: QSVM configuration containing feature map settings.

    Returns:
        A Qiskit feature map circuit.

    Raises:
        ValueError: If the configured feature map type is unsupported.
    """
    if cfg.feature_map == "zz":
        return ZZFeatureMap(
            feature_dimension=n_features,
            reps=cfg.reps,
            entanglement=cfg.entanglement,
        )

    if cfg.feature_map == "pauli":
        return PauliFeatureMap(
            feature_dimension=n_features,
            reps=cfg.reps,
            entanglement=cfg.entanglement,
        )

    raise ValueError(f"Unknown feature_map={cfg.feature_map}")
