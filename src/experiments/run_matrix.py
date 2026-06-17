"""
Run a batch of experiments listed in a YAML run matrix.

The run matrix file must contain an ``experiments`` field with a list of
experiment configuration file paths. Each configuration is executed in
sequence using ``run_experiment``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

from src.experiments.run_experiment import run_experiment


def main(matrix_path: str) -> None:
    p = Path(matrix_path)
    matrix = yaml.safe_load(p.read_text())
    experiments = matrix.get("experiments", [])

    if not experiments:
        raise SystemExit("No experiments found in run matrix.")

    for cfg_path in experiments:
        print(f"\n=== Running: {cfg_path} ===")
        run_experiment(cfg_path)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m src.experiments.run_matrix <run_matrix.yaml>")
    main(sys.argv[1])
