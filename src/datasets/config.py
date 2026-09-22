""" Configuration parameters for loading and preprocessing a dataset."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

@dataclass(frozen=True)
class DatasetConfig:
    """
    Configuration for dataset loading, train/test splitting, and preprocessing.

    Attributes:
        name: Dataset identifier.
        params: Optional dataset-specific loading parameters.
        test_size: Proportion of the dataset reserved for testing.
        split_seed: Random seed used for the train/test split.
        scale: Whether to apply feature scaling.
        pca_components: Number of PCA components, if PCA is used.
        max_samples: Optional limit on the number of samples loaded.
        minmax_scale: Whether to rescale features to [0, 1] after PCA.
        cap_imbalance_to_available: If True, the fixed-size imbalance
            subsampling (train and test) reduces to whatever class counts
            are actually available instead of raising when the pool is
            too small.
    """
    name: str
    params: Optional[Dict[str, Any]] = None
    test_size: float = 0.3
    split_seed: int = 42
    scale: bool = True
    pca_components: Optional[int] = None
    max_samples: Optional[int] = None
    minmax_scale: bool = False
    cap_imbalance_to_available: bool = False