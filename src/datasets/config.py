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
    """
    name: str
    params: Optional[Dict[str, Any]] = None
    test_size: float = 0.3
    split_seed: int = 42
    scale: bool = True
    pca_components: Optional[int] = None 
    max_samples: Optional[int] = None