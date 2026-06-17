"""
Utility functions for loading, merging, and updating YAML-based
configuration dictionaries.
"""
from __future__ import annotations

import copy
import yaml
from typing import Any, Dict, List


def load_yaml(path: str) -> Dict[str, Any]:
    """
    Load and parse a YAML configuration file.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed YAML content as a dictionary.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge an override configuration into a base configuration.

    Args:
        base: Base configuration dictionary.
        override: Configuration values to apply on top of the base.

    Returns:
        A new merged configuration dictionary.
    """
    result = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and k in result and isinstance(result[k], dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


def set_by_path(cfg: Dict[str, Any], path: str, value: Any) -> None:
    """
    Set a nested configuration value using a dot-separated path.

    For example, ``path="a.b.c"`` sets ``cfg["a"]["b"]["c"] = value``.

    Args:
        cfg: Configuration dictionary to update.
        path: Dot-separated key path.
        value: Value to assign.
    """
    keys = path.split(".")
    d = cfg
    for k in keys[:-1]:
        if k not in d or not isinstance(d[k], dict):
            d[k] = {}
        d = d[k]
    d[keys[-1]] = value
