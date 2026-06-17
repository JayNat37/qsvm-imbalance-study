"""
Run classical SVM and QSVM experiments from YAML configuration files.

This module loads experiment settings, prepares datasets, executes the
selected model, and writes results to JSONL files.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .config_utils import load_yaml, deep_merge, set_by_path

from src.datasets.config import DatasetConfig
from src.datasets.loaders import load_dataset_raw
from src.datasets.preprocess import preprocess

from src.qsvm.config import QSVMConfig, NoiseConfig
from src.qsvm.kernel import QuantumKernel

from src.classical.classical_svm import fit_predict_svm
from .metrics import classification_metrics

from sklearn.svm import SVC

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def _build_dataset(run_cfg: Dict[str, Any]):
    """
    Build and preprocess the dataset defined in the run configuration.
    """
    d = run_cfg["dataset"]
    dcfg = DatasetConfig(
        name=d["name"],
        params=d.get("params", {}),
        test_size=d.get("test_size", 0.3),
        split_seed=d.get("split_seed", 42),
        scale=bool(d.get("scale", True)),
        pca_components=d.get("pca_components", None),
        max_samples=d.get("max_samples", None),
    )

    X, y = load_dataset_raw(dcfg)
    X_train, X_test, y_train, y_test = preprocess(X, y, cfg=dcfg)
    return X_train, X_test, y_train, y_test, dcfg

def _build_qsvm(run_cfg: Dict[str, Any]) -> QSVMConfig:
    """
    Construct a QSVM configuration object from the run configuration.
    """
    q = run_cfg["model"]["qsvm"]

    noise_cfg = q["noise"]

    noise = NoiseConfig(
        type=noise_cfg["type"],
        strength=float(noise_cfg["strength"]),
        params=noise_cfg.get("params", {}),
        apply_to=noise_cfg.get("apply_to", "all"),
    )

    qcfg = QSVMConfig(
        feature_map=q["feature_map"],
        reps=int(q["reps"]),
        entanglement=q["entanglement"],
        mode=q["mode"],
        shots=q.get("shots"),
        seed=int(q["seed"]),
        scale_to_pi=bool(q["scale_to_pi"]),
        C=float(q["C"]),
        noise=noise,
    )
    qcfg.validate()
    return qcfg


def _run_classical(run_cfg: Dict[str, Any], X_train, X_test, y_train, y_test):
    """
    Train and evaluate a classical SVM for a single run.
    """
    c = run_cfg["model"]["classical_svm"]

    t0 = time.time()
    y_pred = fit_predict_svm(
        X_train,
        y_train,
        X_test,
        kernel=c.get("kernel", "rbf"),
        C=float(c.get("C", 1.0)),
        gamma=c.get("gamma", "scale"),
        degree=int(c.get("degree", 3)),
    )
    fit_time = time.time() - t0

    metrics = classification_metrics(y_test, y_pred)
    return metrics, {"fit_time": fit_time}


def _run_qsvm(run_cfg: Dict[str, Any], X_train, X_test, y_train, y_test):
    """
    Train and evaluate a QSVM for a single run.
    """
    qcfg = _build_qsvm(run_cfg)
    kernel = QuantumKernel(qcfg)

    t0 = time.time()
    K_train = kernel.compute(X_train, X_train)
    K_test = kernel.compute(X_test, X_train)
    kernel_time = time.time() - t0

    clf = SVC(kernel="precomputed", C=qcfg.C)

    t1 = time.time()
    clf.fit(K_train, y_train)
    scores = clf.decision_function(K_test)
    y_pred = clf.predict(K_test)
    fit_time = time.time() - t1

    metrics = classification_metrics(y_test, y_pred)

    kernel_stats = {
        "mean": float(np.mean(K_train)),
        "std": float(np.std(K_train)),
        "diag_mean": float(np.mean(np.diag(K_train))),
    }

    return qcfg, metrics, kernel_stats, {"kernel_time": kernel_time, "fit_time": fit_time}


def run_experiment(config_path: str) -> None:
    """
    Run an experiment defined by a YAML configuration file.

    Supports optional parameter sweeps and repeated runs, then writes each
    result as a JSON record to the results directory.
    """
    cfg = load_yaml(config_path)

    # Optional base config support
    if "base" in cfg:
        base_cfg = load_yaml(Path(config_path).parent / cfg["base"])
        cfg = deep_merge(base_cfg, cfg)

    sweep = cfg.get("sweep")
    repeats = int(cfg["runs"]["repeats"])
    vary_seed = bool(cfg["runs"].get("vary_seed", False))
    seed_start = int(cfg["runs"].get("seed_start", 0))

    sweep_values = sweep["values"] if sweep else [None]
    sweep_path = sweep["path"] if sweep else None

    results_file = RESULTS_DIR / f"{cfg['name']}.jsonl"

    with results_file.open("a") as f_out:
        for sweep_value in sweep_values:
            for r in range(repeats):
                run_cfg = deep_merge(cfg, cfg.get("override", {}))

                if sweep_path is not None:
                    set_by_path(run_cfg, sweep_path, sweep_value)

                repeat_seed = seed_start + r
                
                # Only vary dataset generation/split across repeats for dataset-noise sweeps
                is_dataset_noise_experiment = sweep_path == "dataset.params.noise"

                # If you repeat, vary QSVM seed
                if vary_seed:
                    if is_dataset_noise_experiment:
                        # Vary dataset realisation and split across repeats
                        run_cfg["dataset"]["split_seed"] = repeat_seed
                        run_cfg.setdefault("dataset", {}).setdefault("params", {})
                        run_cfg["dataset"]["params"]["random_seed"] = repeat_seed

                    # Still vary QSVM simulator seed for QSVM experiments
                    if run_cfg["model"]["kind"] == "qsvm":
                        run_cfg["model"]["qsvm"]["seed"] = repeat_seed

                # Dataset is fixed across repeats by default (split_seed stays fixed)
                X_train, X_test, y_train, y_test, dcfg = _build_dataset(run_cfg)

                if run_cfg["model"]["kind"] == "classical_svm":
                    metrics, timing = _run_classical(run_cfg, X_train, X_test, y_train, y_test)

                    record: Dict[str, Any] = {
                        "experiment": cfg["name"],
                        "model_kind": "classical_svm",
                        "sweep_value": sweep_value,
                        "repeat": r,
                        "dataset": dcfg.__dict__,
                        "classical_svm": run_cfg["model"]["classical_svm"],
                        "metrics": metrics,
                        "timing": timing,
                    }

                    f_out.write(json.dumps(record) + "\n")
                    f_out.flush()

                    print(
                        f"[{cfg['name']}] classical sweep={sweep_value} "
                        f"run={r+1}/{repeats} acc={metrics['accuracy']:.3f}"
                    )

                elif run_cfg["model"]["kind"] == "qsvm":
                    qcfg, metrics, kernel_stats, timing = _run_qsvm(
                        run_cfg, X_train, X_test, y_train, y_test
                    )

                    record = {
                        "experiment": cfg["name"],
                        "model_kind": "qsvm",
                        "sweep_value": sweep_value,
                        "repeat": r,
                        "dataset": dcfg.__dict__,
                        "qsvm": qcfg.to_dict(),
                        "metrics": metrics,
                        "kernel_stats": kernel_stats,
                        "timing": timing,
                    }

                    f_out.write(json.dumps(record) + "\n")
                    f_out.flush()

                    print(
                        f"[{cfg['name']}] qsvm sweep={sweep_value} "
                        f"run={r+1}/{repeats} acc={metrics['accuracy']:.3f}"
                    )

                else:
                    raise ValueError(f"Unknown model.kind={run_cfg['model']['kind']}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python src/experiments/run_experiment.py <config.yaml>")
    run_experiment(sys.argv[1])
