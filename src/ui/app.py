"""
Interactive runner: single classical-SVM vs QSVM comparisons with a run history.

Launch with: streamlit run src/ui/app.py   (from the repo root)
"""
from __future__ import annotations

import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.svm import SVC

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.config import DatasetConfig
from src.datasets.loaders import load_dataset_raw
from src.datasets.preprocess import preprocess, _FIXED_TEST_SIZE, _FIXED_TEST_IMBALANCE_FRACTION
from src.classical.classical_svm import fit_predict_svm
from src.qsvm.config import QSVMConfig, NoiseConfig
from src.qsvm.kernel import QuantumKernel
from src.experiments.metrics import classification_metrics
from src.ui.tuning import make_tuning_split, tune_classical, tune_qsvm

HISTORY_FILE = REPO_ROOT / "results" / "ui_run_history.jsonl"

st.set_page_config(page_title="Classical vs Quantum SVM", layout="wide")


def load_history() -> list[dict]:
    if not HISTORY_FILE.exists():
        return []
    lines = HISTORY_FILE.read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def append_history(record: dict) -> None:
    HISTORY_FILE.parent.mkdir(exist_ok=True)
    with HISTORY_FILE.open("a") as f:
        f.write(json.dumps(record) + "\n")


def flatten_history(records: list[dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        d = r["dataset"]
        c = r["classical"]
        q = r["qsvm"]
        rows.append({
            "timestamp": r["timestamp"],
            "dataset": d["name"],
            "imbalance_frac": (d.get("params") or {}).get("imbalance_fraction"),
            "fixed_size": (d.get("params") or {}).get("fixed_size"),
            "c_acc": c["metrics"]["accuracy"],
            "q_acc": q["metrics"]["accuracy"],
            "c_f1": c["metrics"]["f1"],
            "q_f1": q["metrics"]["f1"],
            "c_precision": c["metrics"]["precision"],
            "q_precision": q["metrics"]["precision"],
            "c_recall": c["metrics"]["recall"],
            "q_recall": q["metrics"]["recall"],
            "pca": d.get("pca_components"),
            "split_seed": d.get("split_seed"),
            "c_kernel": c["params"]["kernel"],
            "c_C": c["params"]["C"],
            "c_gamma": str(c["params"]["gamma"]),  # mixes strings ("scale") and floats across rows otherwise
            "c_class_weight": c["params"].get("class_weight"),
            "c_time_s": round(c["time"], 3),
            "q_feature_map": q["params"]["feature_map"],
            "q_reps": q["params"]["reps"],
            "q_entanglement": q["params"]["entanglement"],
            "q_mode": q["params"]["mode"],
            "q_shots": q["params"]["shots"],
            "q_C": q["params"]["C"],
            "q_class_weight": q["params"].get("class_weight"),
            "q_time_s": round(q["time"], 3),
            "run_id": r["run_id"],
            "sweep_id": r.get("sweep_id"),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        # Exact minority/majority counts, derived from fixed_size/imbalance_frac (round-trips
        # exactly for absolute-count runs, since imbalance_frac was itself computed from counts).
        has_imb = df["fixed_size"].notna() & df["imbalance_frac"].notna()
        df["majority_count"] = np.where(has_imb, (df["fixed_size"] * df["imbalance_frac"]).round(), np.nan)
        df["minority_count"] = np.where(has_imb, df["fixed_size"] - df["majority_count"], np.nan)
    return df


def list_sweeps(history: list[dict]) -> list[dict]:
    """Group history records by sweep_id into a list of {id, label, count} for a sweep picker."""
    by_id: dict[str, list[dict]] = {}
    for r in history:
        sid = r.get("sweep_id")
        if sid:
            by_id.setdefault(sid, []).append(r)

    sweeps = []
    for sid, records in by_id.items():
        records = sorted(records, key=lambda r: r["timestamp"])
        first_ts = records[0]["timestamp"]
        dataset_name = records[0]["dataset"]["name"]
        sweeps.append({
            "id": sid,
            "label": f"{dataset_name} — {first_ts[:16].replace('T', ' ')} — {len(records)} points",
            "count": len(records),
            "timestamp": first_ts,
        })
    return sorted(sweeps, key=lambda s: s["timestamp"], reverse=True)


def run_pair(dataset_cfg: DatasetConfig, classical_params: dict, qsvm_params: dict) -> dict:
    X, y = load_dataset_raw(dataset_cfg)
    X_train, X_test, y_train, y_test = preprocess(X, y, cfg=dataset_cfg)

    t0 = time.time()
    y_pred_c = fit_predict_svm(
        X_train, y_train, X_test,
        kernel=classical_params["kernel"],
        C=classical_params["C"],
        gamma=classical_params["gamma"],
        degree=classical_params["degree"],
        class_weight=classical_params.get("class_weight"),
    )
    c_time = time.time() - t0
    c_metrics = classification_metrics(y_test, y_pred_c)

    qcfg = QSVMConfig(
        feature_map=qsvm_params["feature_map"],
        reps=qsvm_params["reps"],
        entanglement=qsvm_params["entanglement"],
        mode=qsvm_params["mode"],
        shots=qsvm_params["shots"],
        seed=qsvm_params["seed"],
        scale_to_pi=qsvm_params["scale_to_pi"],
        C=qsvm_params["C"],
        noise=NoiseConfig(
            type=qsvm_params["noise_type"],
            strength=qsvm_params["noise_strength"],
            params={},
            apply_to=qsvm_params["noise_apply_to"],
        ),
    )
    qcfg.validate()
    kernel = QuantumKernel(qcfg)

    t1 = time.time()
    K_train = kernel.compute(X_train, X_train)
    K_test = kernel.compute(X_test, X_train)
    clf = SVC(kernel="precomputed", C=qcfg.C, class_weight=qsvm_params.get("class_weight"))
    clf.fit(K_train, y_train)
    y_pred_q = clf.predict(K_test)
    q_time = time.time() - t1
    q_metrics = classification_metrics(y_test, y_pred_q)

    return {
        "run_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": dataset_cfg.__dict__,
        "train_shape": list(X_train.shape),
        "test_shape": list(X_test.shape),
        "train_class_counts": np.bincount(y_train, minlength=2).tolist(),
        "test_class_counts": np.bincount(y_test, minlength=2).tolist(),
        "classical": {"params": classical_params, "metrics": c_metrics, "time": c_time},
        "qsvm": {"params": qsvm_params, "metrics": q_metrics, "time": q_time, "config": qcfg.to_dict()},
    }


# ---------------------------------------------------------------------------
# Queue: state, timing model, helpers
# ---------------------------------------------------------------------------

_DEFAULT_RATES = {
    "classical": 0.02,                # seconds, flat (classical fits are always fast)
    "statevector_per_unique": 0.01,   # seconds per unique input (train+test count)
    "noisy_aer_per_pair": 0.0025,     # seconds per train/test pair, calibrated from observed runs
}

if "queue" not in st.session_state:
    st.session_state.queue = []
if "queue_running" not in st.session_state:
    st.session_state.queue_running = False
if "queue_run_start_time" not in st.session_state:
    st.session_state.queue_run_start_time = None
if "last_queue_total_time" not in st.session_state:
    st.session_state.last_queue_total_time = None
if "timing_rates" not in st.session_state:
    st.session_state.timing_rates = {k: [] for k in _DEFAULT_RATES}
if "tuning_result" not in st.session_state:
    st.session_state.tuning_result = None
if "_pending_apply" not in st.session_state:
    st.session_state._pending_apply = None

# Apply any staged widget values *before* the widgets below are instantiated —
# Streamlit forbids writing to session_state[key] after that key's widget has
# already rendered in the current script run.
if st.session_state._pending_apply:
    for _key, _value in st.session_state._pending_apply.items():
        st.session_state[_key] = _value
    st.session_state._pending_apply = None


def make_label(dataset_name: str, dataset_params: dict, classical_params: dict, qsvm_params: dict) -> str:
    frac = dataset_params.get("imbalance_fraction")
    size = dataset_params.get("fixed_size")
    imb = f"frac={frac}, n={size}" if frac is not None else "no fixed imbalance"
    q = f"{qsvm_params['feature_map']}/{qsvm_params['mode']}"
    if qsvm_params["mode"] == "noisy_aer":
        q += f"/{qsvm_params['shots']}sh"
    c_cw = classical_params.get("class_weight") or "none"
    q_cw = qsvm_params.get("class_weight") or "none"
    return (f"{dataset_name} [{imb}] · classical(C={classical_params['C']}, γ={classical_params['gamma']}, cw={c_cw}) "
            f"· qsvm({q}, C={qsvm_params['C']}, cw={q_cw})")


def build_queue_item(
    dataset_name: str, dataset_params: dict, classical_params: dict, qsvm_params: dict,
    test_size: float, split_seed: int, scale: bool, pca_components: int | None, max_samples: int | None,
    minmax_scale: bool, sweep_id: str | None = None, tune_before_run: bool = False,
    tune_grids: dict | None = None, label_prefix: str = "",
) -> dict:
    """Build a queue item dict with the given dataset/model settings."""
    label = make_label(dataset_name, dataset_params, classical_params, qsvm_params)
    return {
        "id": str(uuid.uuid4()),
        "status": "pending",
        "label": f"{label_prefix}{label}" if label_prefix else label,
        "dataset_name": dataset_name,
        "dataset_params": dict(dataset_params),
        "test_size": float(test_size),
        "split_seed": int(split_seed),
        "scale": bool(scale),
        "pca_components": int(pca_components) if pca_components else None,
        "max_samples": int(max_samples) if max_samples else None,
        "minmax_scale": bool(minmax_scale),
        "classical_params": dict(classical_params),
        "qsvm_params": dict(qsvm_params),
        "sweep_id": sweep_id,
        "tune_before_run": tune_before_run,
        "tune_grids": tune_grids,
        "result": None,
        "error": None,
        "elapsed": None,
    }


def _item_workload(item: dict) -> tuple[str, float]:
    """Estimate (rate_key, workload_units) for the qsvm half of a queued item, ahead of running it."""
    n_train = item["dataset_params"].get("fixed_size") or 100
    n_test = _FIXED_TEST_SIZE if item["dataset_params"].get("imbalance_fraction") is not None else max(
        int(round(n_train * item["test_size"] / (1 - item["test_size"]))), 1
    )
    if item["qsvm_params"]["mode"] == "statevector":
        return "statevector_per_unique", float(n_train + n_test)
    pairs = n_train * n_train + n_test * n_train
    return "noisy_aer_per_pair", float(pairs)


def _rate(key: str) -> float:
    samples = st.session_state.timing_rates.get(key) or []
    return (sum(samples) / len(samples)) if samples else _DEFAULT_RATES[key]


def predict_item_time(item: dict) -> float:
    rate_key, workload = _item_workload(item)
    qsvm_estimate = _rate(rate_key) * workload
    classical_estimate = _rate("classical")
    return max(qsvm_estimate + classical_estimate, 0.05)


def record_observed_time(item: dict, classical_time: float, qsvm_time: float) -> None:
    st.session_state.timing_rates["classical"].append(classical_time)
    rate_key, workload = _item_workload(item)
    if workload > 0:
        st.session_state.timing_rates[rate_key].append(qsvm_time / workload)


def _range_values(start: float, stop: float, step: float) -> list[float]:
    """Inclusive min..max range at the given step, tolerant of float drift."""
    if step <= 0 or stop < start:
        return []
    n = int(round((stop - start) / step)) + 1
    return [round(start + i * step, 10) for i in range(n)]


def tune_qsvm_over_feature_maps(X_tr, y_tr, X_val, y_val, feature_maps, **kwargs) -> dict:
    """Run tune_qsvm once per feature map, keeping whichever wins on validation accuracy."""
    best = None
    for fm in feature_maps:
        candidate = tune_qsvm(X_tr, y_tr, X_val, y_val, feature_map=fm, **kwargs)
        candidate["feature_map"] = fm
        if best is None or candidate["val_accuracy"] > best["val_accuracy"]:
            best = candidate
    return best


# ---------------------------------------------------------------------------
# Sidebar: parameter form
# ---------------------------------------------------------------------------

_FASHION_MNIST_LABELS = {
    0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
    5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle boot",
}

st.sidebar.header("Dataset")
dataset_name = st.sidebar.selectbox(
    "Dataset", ["mnist_01", "fashion_mnist", "breast_cancer", "banknote", "make_moons"], key="dataset_name",
)

dataset_params: dict = {}
if dataset_name in ("mnist_01", "fashion_mnist"):
    max_class = 9
    class_a = st.sidebar.number_input("Class A", value=0, min_value=0, max_value=max_class, step=1, key="class_a_input")
    class_b = st.sidebar.number_input("Class B", value=1, min_value=0, max_value=max_class, step=1, key="class_b_input")
    dataset_params["class_a"] = int(class_a)
    dataset_params["class_b"] = int(class_b)
    if dataset_name == "fashion_mnist":
        st.sidebar.caption(
            f"A = {_FASHION_MNIST_LABELS[int(class_a)]}, B = {_FASHION_MNIST_LABELS[int(class_b)]} "
            f"(0=T-shirt/top, 1=Trouser, 2=Pullover, 3=Dress, 4=Coat, 5=Sandal, 6=Shirt, 7=Sneaker, 8=Bag, 9=Ankle boot)"
        )
elif dataset_name == "banknote":
    dataset_params["path"] = st.sidebar.text_input("CSV path", value="data/data_banknote_authentication.txt")
elif dataset_name == "make_moons":
    dataset_params["n_samples"] = st.sidebar.number_input("n_samples", value=300, min_value=10)
    dataset_params["noise"] = st.sidebar.number_input("noise", value=0.2, min_value=0.0, max_value=1.0)
    dataset_params["random_seed"] = st.sidebar.number_input("random_seed", value=42)

test_size = st.sidebar.slider("Test size", 0.05, 0.5, 0.3)
split_seed = st.sidebar.number_input("Split seed", value=42, step=1)
scale = st.sidebar.checkbox("StandardScaler", value=True)

apply_pca = st.sidebar.checkbox("Apply PCA", value=True)
pca_components = st.sidebar.number_input("PCA components", value=5, min_value=1) if apply_pca else None

minmax_scale = st.sidebar.checkbox("MinMax scale [0,1] after PCA", value=True)

apply_cap = st.sidebar.checkbox("Cap pool size (max_samples)", value=True)
max_samples = st.sidebar.number_input("max_samples", value=1000, min_value=10) if apply_cap else None

use_imbalance = st.sidebar.checkbox("Fixed-size class imbalance", value=True)
if use_imbalance:
    imbalance_mode = st.sidebar.radio(
        "Specify imbalance as", ["Fraction + size", "Absolute counts"], key="imbalance_mode",
        help="'Absolute counts' lets you set exact class sizes, e.g. 2 minority examples against 100 majority.",
    )
    if imbalance_mode == "Fraction + size":
        fixed_size = st.sidebar.number_input("Fixed train size", value=100, min_value=2, step=2, key="fixed_size_input")
        imbalance_fraction = st.sidebar.slider("Class-0 fraction", 0.0, 1.0, 0.5, step=0.05, key="imbalance_fraction_input")
    else:
        n_class0 = st.sidebar.number_input("Class 0 count (majority)", value=97, min_value=1, step=1, key="n_class0_input")
        n_class1 = st.sidebar.number_input("Class 1 count (minority)", value=3, min_value=1, step=1, key="n_class1_input")
        fixed_size = int(n_class0) + int(n_class1)
        imbalance_fraction = n_class0 / fixed_size
        st.sidebar.caption(f"= fixed train size {fixed_size}, class-0 fraction {imbalance_fraction:.4f}")
    dataset_params["fixed_size"] = int(fixed_size)
    dataset_params["imbalance_fraction"] = float(imbalance_fraction)

st.sidebar.header("Classical SVM")
c_kernel = st.sidebar.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"], key="c_kernel")
c_C = st.sidebar.number_input("C (classical)", value=0.1, min_value=0.0001, format="%.4f", key="c_C")
c_gamma_mode = st.sidebar.radio("gamma", ["scale", "auto", "custom"], horizontal=True, key="c_gamma_mode")
c_gamma_custom = st.sidebar.number_input("gamma value", value=1.0, format="%.4f", key="c_gamma_custom") if c_gamma_mode == "custom" else None
c_gamma = c_gamma_custom if c_gamma_mode == "custom" else c_gamma_mode
c_degree = st.sidebar.number_input("degree (poly only)", value=3, min_value=1, key="c_degree") if c_kernel == "poly" else 3
c_class_weight_mode = st.sidebar.selectbox(
    "class_weight", ["None", "balanced"], key="c_class_weight_mode",
    help="'balanced' reweights the misclassification penalty inversely to class frequency — the standard fix for imbalanced training data.",
)
c_class_weight = None if c_class_weight_mode == "None" else "balanced"
compare_class_weight = st.sidebar.checkbox(
    "Also queue the class_weight-flipped variant", value=False, key="compare_class_weight",
    help="When adding to the queue, also queue an identical run with class_weight flipped "
         "(None <-> balanced) for both models, so you can compare them side by side in history.",
)

st.sidebar.header("QSVM")
feature_map = st.sidebar.selectbox("Feature map", ["zz", "pauli"], key="feature_map")
reps = st.sidebar.number_input("reps", value=1, min_value=1, key="reps")
entanglement = st.sidebar.selectbox("Entanglement", ["linear", "full", "circular"], key="entanglement")
mode = st.sidebar.selectbox("Mode", ["statevector", "noisy_aer"], key="mode")
q_C = st.sidebar.number_input("C (quantum)", value=10.0, min_value=0.0001, format="%.4f", key="q_C")
q_class_weight_mode = st.sidebar.selectbox(
    "class_weight (QSVM)", ["None", "balanced"], key="q_class_weight_mode",
    help="Same mechanism as classical's class_weight, applied to the precomputed-kernel SVC after "
         "the quantum kernel is computed. Often a no-op on easy tasks, a real lever on hard ones.",
)
q_class_weight = None if q_class_weight_mode == "None" else "balanced"
scale_to_pi = st.sidebar.checkbox("scale_to_pi", value=False, key="scale_to_pi")
q_seed = st.sidebar.number_input("QSVM seed", value=42, step=1, key="q_seed")

shots_input = st.sidebar.number_input(
    "shots", value=1024, min_value=1, step=1,
    disabled=(mode == "statevector"),
    help="Only used in noisy_aer mode. statevector mode is exact/analytic and always ignores shots.",
)
shots = int(shots_input) if mode == "noisy_aer" else None

noise_type = "none"
noise_strength = 0.0
noise_apply_to = "all"
if mode == "noisy_aer":
    noise_type = st.sidebar.selectbox("Noise type", ["none", "depolarizing", "amplitude_damping", "phase_damping", "combined"])
    if noise_type != "none":
        noise_strength = st.sidebar.number_input("Noise strength", value=0.01, min_value=0.0, max_value=1.0, format="%.4f")
        noise_apply_to = st.sidebar.selectbox("Noise applies to", ["all", "single_qubit", "two_qubit"])
else:
    st.sidebar.caption("Noise model is only available in noisy_aer mode.")

run_col1, run_col2 = st.sidebar.columns(2)
run_clicked = run_col1.button("Run now", type="primary")
add_to_queue_clicked = run_col2.button("Add to queue")
add_comparison_clicked = st.sidebar.button(
    "Add class_weight comparison (2 runs)", disabled=not compare_class_weight,
    help="Queues the current settings plus an identical run with class_weight flipped for both "
         "models. Enable 'Also queue the class_weight-flipped variant' above to use this.",
)

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("Classical SVM vs QSVM — Interactive Runner")

classical_params_current = {
    "kernel": c_kernel, "C": float(c_C), "gamma": c_gamma, "degree": int(c_degree),
    "class_weight": c_class_weight,
}
qsvm_params_current = {
    "feature_map": feature_map, "reps": int(reps), "entanglement": entanglement,
    "mode": mode, "shots": shots, "seed": int(q_seed), "scale_to_pi": bool(scale_to_pi),
    "C": float(q_C), "class_weight": q_class_weight, "noise_type": noise_type,
    "noise_strength": float(noise_strength), "noise_apply_to": noise_apply_to,
}

st.header("Tune hyperparameters")

_GRID_C_CHOICES = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
_GRID_GAMMA_CHOICES = ["scale", "auto", 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
_GRID_CLASS_WEIGHT_CHOICES = ["None", "balanced"]
_GRID_REPS_CHOICES = [1, 2, 3, 4, 5]
_GRID_ENTANGLEMENT_CHOICES = ["linear", "full", "circular"]

tune_col, settings_col = st.columns([1, 1])
with tune_col:
    tune_clicked = st.button("Tune for current dataset settings")
with settings_col:
    with st.popover("Grid settings", icon=":material/tune:"):
        st.caption("Classical SVM grid")
        grid_c_classical = st.multiselect(
            "C values", _GRID_C_CHOICES, default=[0.1, 1.0, 10.0, 100.0], key="grid_c_classical",
        )
        grid_gamma = st.multiselect(
            "gamma values", _GRID_GAMMA_CHOICES, default=["scale", 0.001, 0.01, 0.1, 1.0], key="grid_gamma",
        )
        grid_class_weight_labels = st.multiselect(
            "class_weight values", _GRID_CLASS_WEIGHT_CHOICES, default=["None", "balanced"], key="grid_class_weight",
        )

        st.caption("QSVM grid")
        grid_feature_maps = st.multiselect(
            "feature_map values", ["zz", "pauli"], default=["zz"], key="grid_feature_map",
            help="Search across feature maps too, not just reps/entanglement/C, matching what widened the QSVM re-tuning tests.",
        )
        grid_reps = st.multiselect("reps values", _GRID_REPS_CHOICES, default=[1, 2, 3], key="grid_reps")
        grid_entanglement = st.multiselect(
            "entanglement values", _GRID_ENTANGLEMENT_CHOICES, default=["linear", "full"], key="grid_entanglement",
        )
        grid_c_qsvm = st.multiselect(
            "C values (QSVM)", _GRID_C_CHOICES, default=[0.1, 1.0, 10.0, 100.0], key="grid_c_qsvm",
        )
        grid_class_weight_qsvm_labels = st.multiselect(
            "class_weight values (QSVM)", _GRID_CLASS_WEIGHT_CHOICES, default=["None", "balanced"],
            key="grid_class_weight_qsvm",
        )

if tune_clicked:
    empty_grids = [
        name for name, values in [
            ("classical C", grid_c_classical), ("gamma", grid_gamma), ("class_weight", grid_class_weight_labels),
            ("QSVM feature_map", grid_feature_maps), ("QSVM reps", grid_reps),
            ("QSVM entanglement", grid_entanglement), ("QSVM C", grid_c_qsvm),
            ("QSVM class_weight", grid_class_weight_qsvm_labels),
        ] if not values
    ]
    if empty_grids:
        st.error(f"Grid settings can't be empty: {', '.join(empty_grids)}. Pick at least one value each.")
    else:
        tuning_dataset_cfg = DatasetConfig(
            name=dataset_name,
            params=dataset_params,
            test_size=float(test_size),
            split_seed=int(split_seed),
            scale=bool(scale),
            pca_components=int(pca_components) if pca_components else None,
            max_samples=int(max_samples) if max_samples else None,
            minmax_scale=bool(minmax_scale),
            cap_imbalance_to_available=True,
        )
        class_weight_grid = tuple(None if v == "None" else v for v in grid_class_weight_labels)
        class_weight_grid_qsvm = tuple(None if v == "None" else v for v in grid_class_weight_qsvm_labels)
        try:
            with st.spinner("Loading data and running grid search..."):
                X_tune, y_tune = load_dataset_raw(tuning_dataset_cfg)
                X_train_pool, _, y_train_pool, _ = preprocess(X_tune, y_tune, cfg=tuning_dataset_cfg)
                X_tr, X_val, y_tr, y_val = make_tuning_split(X_train_pool, y_train_pool, seed=int(split_seed))
                best_classical = tune_classical(
                    X_tr, y_tr, X_val, y_val,
                    c_grid=grid_c_classical, gamma_grid=grid_gamma, class_weight_grid=class_weight_grid,
                )
                best_qsvm = tune_qsvm_over_feature_maps(
                    X_tr, y_tr, X_val, y_val, grid_feature_maps,
                    scale_to_pi=bool(scale_to_pi), seed=int(split_seed),
                    reps_grid=grid_reps, entanglement_grid=grid_entanglement, c_grid=grid_c_qsvm,
                    class_weight_grid=class_weight_grid_qsvm,
                )
            st.session_state.tuning_result = {"classical": best_classical, "qsvm": best_qsvm}
        except Exception as e:
            st.session_state.tuning_result = None
            st.error(f"Tuning failed: {e}")

if st.session_state.tuning_result:
    tr = st.session_state.tuning_result
    tcol1, tcol2 = st.columns(2)
    with tcol1:
        bc = tr["classical"]
        st.write(f"**Best classical**: C={bc['C']}, gamma={bc['gamma']}, class_weight={bc['class_weight'] or 'none'}")
        st.caption(f"Validation accuracy: {bc['val_accuracy']:.3f}")
    with tcol2:
        bq = tr["qsvm"]
        st.write(f"**Best QSVM**: feature_map={bq['feature_map']}, reps={bq['reps']}, "
                 f"entanglement={bq['entanglement']}, C={bq['C']}, class_weight={bq['class_weight'] or 'none'}")
        st.caption(f"Validation accuracy: {bq['val_accuracy']:.3f}")

    if st.button("Apply tuned values to sidebar"):
        bc, bq = tr["classical"], tr["qsvm"]
        pending = {
            "c_C": float(bc["C"]),
            "c_class_weight_mode": "None" if bc["class_weight"] is None else "balanced",
            "feature_map": bq["feature_map"],
            "reps": int(bq["reps"]),
            "entanglement": bq["entanglement"],
            "q_C": float(bq["C"]),
            "q_class_weight_mode": "None" if bq["class_weight"] is None else "balanced",
        }
        if bc["gamma"] in ("scale", "auto"):
            pending["c_gamma_mode"] = bc["gamma"]
        else:
            pending["c_gamma_mode"] = "custom"
            pending["c_gamma_custom"] = float(bc["gamma"])
        st.session_state._pending_apply = pending
        st.rerun()

st.divider()
st.header("3D imbalance sweep")
st.caption(
    "Sweep two axes of imbalance over a grid, running classical + QSVM at every "
    "point using the current sidebar settings, and add them all to the queue below."
)

sweep_mode = st.radio(
    "Sweep by", ["Size × Fraction", "Majority count × Minority count"], key="sweep_mode", horizontal=True,
    help=(
        "'Size × Fraction' sweeps total training size against class-0 fraction. "
        "'Majority × Minority' sweeps exact class counts directly — e.g. minority counts "
        "1, 2, 3 against a range of majority counts."
    ),
)

if sweep_mode == "Size × Fraction":
    sweep_col1, sweep_col2 = st.columns(2)
    with sweep_col1:
        st.write("Training set size")
        sweep_size_min = st.number_input("Size min", value=100, min_value=2, step=2, key="sweep_size_min")
        sweep_size_max = st.number_input("Size max", value=300, min_value=2, step=2, key="sweep_size_max")
        sweep_size_step = st.number_input("Size step", value=100, min_value=2, step=2, key="sweep_size_step")
    with sweep_col2:
        st.write("Class-0 fraction")
        sweep_frac_min = st.number_input("Fraction min", value=0.5, min_value=0.0, max_value=1.0, step=0.05, key="sweep_frac_min")
        sweep_frac_max = st.number_input("Fraction max", value=0.9, min_value=0.0, max_value=1.0, step=0.05, key="sweep_frac_max")
        sweep_frac_step = st.number_input("Fraction step", value=0.1, min_value=0.01, max_value=1.0, step=0.05, key="sweep_frac_step")

    sweep_sizes = [int(v) for v in _range_values(sweep_size_min, sweep_size_max, sweep_size_step)]
    sweep_fracs = _range_values(sweep_frac_min, sweep_frac_max, sweep_frac_step)
    sweep_grid = [(s + 0, f) for s in sweep_sizes for f in sweep_fracs]
    sweep_grid_desc = f"{len(sweep_sizes)} sizes × {len(sweep_fracs)} fractions"
else:
    sweep_col1, sweep_col2 = st.columns(2)
    with sweep_col1:
        st.write("Minority class count (exact)")
        sweep_minority_counts = st.multiselect(
            "Minority counts", [1, 2, 3, 4, 5, 7, 10, 15, 20], default=[1, 2, 3], key="sweep_minority_counts",
        )
    with sweep_col2:
        st.write("Majority class count")
        sweep_majority_min = st.number_input("Majority min", value=47, min_value=1, step=1, key="sweep_majority_min")
        sweep_majority_max = st.number_input("Majority max", value=197, min_value=1, step=1, key="sweep_majority_max")
        sweep_majority_step = st.number_input("Majority step", value=50, min_value=1, step=1, key="sweep_majority_step")

    sweep_majority_counts = [int(v) for v in _range_values(sweep_majority_min, sweep_majority_max, sweep_majority_step)]
    sweep_grid = [
        (n_min + n_maj, n_maj / (n_min + n_maj))
        for n_min in sweep_minority_counts for n_maj in sweep_majority_counts
    ]
    sweep_grid_desc = f"{len(sweep_minority_counts)} minority counts × {len(sweep_majority_counts)} majority counts"

tune_at_each_point = st.checkbox(
    "Tune hyperparameters at each point", value=False, key="sweep_tune_at_each_point",
    help=(
        "Off: every point uses the classical/QSVM hyperparameters currently set in the sidebar. "
        "On: each point gets its own quick grid search (using the 'Grid settings' above) before "
        "its evaluation run, so the manifold shows best-achievable accuracy rather than one fixed "
        "config's robustness. Uses the fast exact kernel for QSVM tuning, so it's cheap per point."
    ),
)

if sweep_grid:
    _est_total = sum(
        predict_item_time({
            "dataset_params": {"fixed_size": s, "imbalance_fraction": f},
            "test_size": float(test_size),
            "qsvm_params": qsvm_params_current,
        })
        for s, f in sweep_grid
    )
    _tune_note = " (plus per-point tuning time, not included in this estimate)" if tune_at_each_point else ""
    st.caption(
        f"{len(sweep_grid)} points ({sweep_grid_desc}) — "
        f"roughly {_est_total:.0f}s estimated once queued{_tune_note}."
    )
else:
    st.warning("This range produced no grid points — check the min/max/step or minority-count selection.")

_sweep_empty_grids = [
    name for name, values in [
        ("classical C", grid_c_classical), ("gamma", grid_gamma), ("class_weight", grid_class_weight_labels),
        ("QSVM feature_map", grid_feature_maps), ("QSVM reps", grid_reps),
        ("QSVM entanglement", grid_entanglement), ("QSVM C", grid_c_qsvm),
        ("QSVM class_weight", grid_class_weight_qsvm_labels),
    ] if not values
] if tune_at_each_point else []
if tune_at_each_point and _sweep_empty_grids:
    st.error(f"Grid settings can't be empty: {', '.join(_sweep_empty_grids)}. Pick at least one value each.")

if st.button("Add sweep to queue", disabled=not sweep_grid or bool(_sweep_empty_grids)):
    sweep_id = str(uuid.uuid4())
    tune_grids = {
        "c_classical": list(grid_c_classical),
        "gamma": list(grid_gamma),
        "class_weight": tuple(None if v == "None" else v for v in grid_class_weight_labels),
        "feature_maps": list(grid_feature_maps),
        "reps": list(grid_reps),
        "entanglement": list(grid_entanglement),
        "c_qsvm": list(grid_c_qsvm),
        "class_weight_qsvm": tuple(None if v == "None" else v for v in grid_class_weight_qsvm_labels),
    } if tune_at_each_point else None
    for s, f in sweep_grid:
        sweep_dataset_params = dict(dataset_params)
        sweep_dataset_params["fixed_size"] = int(s)
        sweep_dataset_params["imbalance_fraction"] = float(f)
        label_prefix = "[sweep, tuned] " if tune_at_each_point else "[sweep] "
        sweep_item = build_queue_item(
            dataset_name, sweep_dataset_params, classical_params_current, qsvm_params_current,
            test_size, split_seed, scale, pca_components, max_samples, minmax_scale,
            sweep_id=sweep_id, tune_before_run=bool(tune_at_each_point), tune_grids=tune_grids,
            label_prefix=label_prefix,
        )
        st.session_state.queue.append(sweep_item)
    st.success(f"Added {len(sweep_grid)} points to the queue (sweep {sweep_id[:8]}). Scroll down to run the queue.")
    st.rerun()

st.divider()

if add_to_queue_clicked:
    item = build_queue_item(
        dataset_name, dataset_params, classical_params_current, qsvm_params_current,
        test_size, split_seed, scale, pca_components, max_samples, minmax_scale,
    )
    st.session_state.queue.append(item)

if add_comparison_clicked:
    flipped_classical = {**classical_params_current, "class_weight": "balanced" if c_class_weight is None else None}
    flipped_qsvm = {**qsvm_params_current, "class_weight": "balanced" if q_class_weight is None else None}
    original_item = build_queue_item(
        dataset_name, dataset_params, classical_params_current, qsvm_params_current,
        test_size, split_seed, scale, pca_components, max_samples, minmax_scale,
        label_prefix="[cw compare] ",
    )
    flipped_item = build_queue_item(
        dataset_name, dataset_params, flipped_classical, flipped_qsvm,
        test_size, split_seed, scale, pca_components, max_samples, minmax_scale,
        label_prefix="[cw compare, flipped] ",
    )
    st.session_state.queue.append(original_item)
    st.session_state.queue.append(flipped_item)
    st.success("Added both class_weight variants to the queue.")
    st.rerun()

if run_clicked:
    dataset_cfg = DatasetConfig(
        name=dataset_name,
        params=dataset_params,
        test_size=float(test_size),
        split_seed=int(split_seed),
        scale=bool(scale),
        pca_components=int(pca_components) if pca_components else None,
        max_samples=int(max_samples) if max_samples else None,
        minmax_scale=bool(minmax_scale),
        cap_imbalance_to_available=True,
    )
    classical_params = classical_params_current
    qsvm_params = qsvm_params_current

    try:
        with st.spinner("Running classical SVM and QSVM (quantum kernel may take a while)..."):
            result = run_pair(dataset_cfg, classical_params, qsvm_params)
        append_history(result)
        st.success(f"Run complete — train {result['train_shape']}, test {result['test_shape']}")

        if use_imbalance:
            requested_n0 = int(round(int(fixed_size) * float(imbalance_fraction)))
            requested_n1 = int(fixed_size) - requested_n0
            actual_n0, actual_n1 = result["train_class_counts"]
            if (actual_n0, actual_n1) != (requested_n0, requested_n1):
                st.warning(
                    f"Training set requested {requested_n0} class-0 / {requested_n1} class-1 "
                    f"({fixed_size} total), but the pool only had enough for "
                    f"{actual_n0} class-0 / {actual_n1} class-1 "
                    f"({actual_n0 + actual_n1} total instead). Maxed out to what's available."
                )

        requested_test_n0 = int(round(_FIXED_TEST_SIZE * _FIXED_TEST_IMBALANCE_FRACTION))
        requested_test_n1 = _FIXED_TEST_SIZE - requested_test_n0
        actual_test_n0, actual_test_n1 = result["test_class_counts"]
        if (actual_test_n0, actual_test_n1) != (requested_test_n0, requested_test_n1):
            st.warning(
                f"Fixed balanced test set requested {_FIXED_TEST_SIZE} samples "
                f"({requested_test_n0}/{requested_test_n1} split), but the pool only had enough for "
                f"{actual_test_n0} class-0 / {actual_test_n1} class-1 "
                f"({actual_test_n0 + actual_test_n1} total instead). Maxed out to what's available."
            )

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Classical SVM")
            cm = result["classical"]["metrics"]
            st.metric("Accuracy", f"{cm['accuracy']:.3f}")
            st.metric("F1", f"{cm['f1']:.3f}")
            st.caption(f"Precision {cm['precision']:.3f} · Recall {cm['recall']:.3f} · "
                       f"Time {result['classical']['time']:.2f}s")
            st.json(cm["confusion_matrix"])
        with col2:
            st.subheader("QSVM")
            qm = result["qsvm"]["metrics"]
            st.metric("Accuracy", f"{qm['accuracy']:.3f}")
            st.metric("F1", f"{qm['f1']:.3f}")
            st.caption(f"Precision {qm['precision']:.3f} · Recall {qm['recall']:.3f} · "
                       f"Time {result['qsvm']['time']:.2f}s")
            st.json(qm["confusion_matrix"])

        st.caption(f"Train class counts {result['train_class_counts']} · "
                   f"Test class counts {result['test_class_counts']}")
    except Exception as e:
        st.error(f"Run failed: {e}")

st.divider()
st.header("Queue")

queue = st.session_state.queue
pending_count = sum(1 for it in queue if it["status"] == "pending")
running_item = next((it for it in queue if it["status"] == "running"), None)

if not queue:
    st.info("Queue is empty — configure parameters in the sidebar and click 'Add to queue'.")
    if st.session_state.last_queue_total_time is not None:
        st.caption(f"Last queue run finished in {st.session_state.last_queue_total_time:.1f}s")
else:
    for item in queue:
        cols = st.columns([0.06, 0.6, 0.14, 0.1, 0.1])
        icon = {"pending": "⏳", "running": "▶️", "done": "✅", "failed": "❌"}[item["status"]]
        cols[0].write(icon)
        cols[1].write(item["label"])
        if item["status"] == "done":
            cols[2].write(
                f"c={item['result']['classical']['metrics']['accuracy']:.3f} · "
                f"q={item['result']['qsvm']['metrics']['accuracy']:.3f}"
            )
            cols[3].write(f"{item['elapsed']:.1f}s")
        elif item["status"] == "failed":
            cols[2].write(f"error: {item['error']}")
        elif item["status"] == "running":
            cols[2].write("running…")
        else:
            cols[2].write(f"~{predict_item_time(item):.0f}s est.")

        if item["status"] == "pending":
            if cols[4].button("Remove", key=f"remove_{item['id']}"):
                st.session_state.queue = [it for it in st.session_state.queue if it["id"] != item["id"]]
                st.rerun()

    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 3])
    with btn_col1:
        run_queue_clicked = st.button(
            "Run queue ▶", type="primary",
            disabled=(pending_count == 0 or st.session_state.queue_running),
        )
    with btn_col2:
        if st.button("Clear finished"):
            st.session_state.queue = [it for it in st.session_state.queue if it["status"] in ("pending", "running")]
            st.rerun()

    if run_queue_clicked:
        st.session_state.queue_running = True
        st.session_state.queue_run_start_time = time.time()

    if st.session_state.queue_running:
        remaining_items = [it for it in st.session_state.queue if it["status"] == "pending"]
        next_item = remaining_items[0] if remaining_items else None

        elapsed_so_far = time.time() - st.session_state.queue_run_start_time
        estimated_remaining = sum(predict_item_time(it) for it in remaining_items)

        st.info(
            f"Queue running — elapsed {elapsed_so_far:.1f}s · "
            f"estimated remaining {estimated_remaining:.1f}s · "
            f"{len(remaining_items)} item(s) left"
        )

        if next_item is None:
            st.session_state.queue_running = False
            st.session_state.last_queue_total_time = elapsed_so_far
            st.success(f"Queue complete — total time {elapsed_so_far:.1f}s")
        else:
            next_item["status"] = "running"
            with st.spinner(f"Running: {next_item['label']}"):
                try:
                    dataset_cfg = DatasetConfig(
                        name=next_item["dataset_name"],
                        params=next_item["dataset_params"],
                        test_size=next_item["test_size"],
                        split_seed=next_item["split_seed"],
                        scale=next_item["scale"],
                        pca_components=next_item["pca_components"],
                        max_samples=next_item["max_samples"],
                        minmax_scale=next_item["minmax_scale"],
                        cap_imbalance_to_available=True,
                    )
                    t_item = time.time()

                    if next_item.get("tune_before_run"):
                        tg = next_item["tune_grids"]
                        X_pt, y_pt = load_dataset_raw(dataset_cfg)
                        X_train_pt, _, y_train_pt, _ = preprocess(X_pt, y_pt, cfg=dataset_cfg)
                        X_tr_pt, X_val_pt, y_tr_pt, y_val_pt = make_tuning_split(
                            X_train_pt, y_train_pt, seed=next_item["split_seed"],
                        )
                        best_c_pt = tune_classical(
                            X_tr_pt, y_tr_pt, X_val_pt, y_val_pt,
                            c_grid=tg["c_classical"], gamma_grid=tg["gamma"], class_weight_grid=tg["class_weight"],
                        )
                        best_q_pt = tune_qsvm_over_feature_maps(
                            X_tr_pt, y_tr_pt, X_val_pt, y_val_pt,
                            tg.get("feature_maps", [next_item["qsvm_params"]["feature_map"]]),
                            scale_to_pi=next_item["qsvm_params"]["scale_to_pi"],
                            seed=next_item["split_seed"],
                            reps_grid=tg["reps"], entanglement_grid=tg["entanglement"], c_grid=tg["c_qsvm"],
                            class_weight_grid=tg.get("class_weight_qsvm", (None, "balanced")),
                        )
                        next_item["classical_params"] = {
                            **next_item["classical_params"],
                            "C": best_c_pt["C"], "gamma": best_c_pt["gamma"], "class_weight": best_c_pt["class_weight"],
                        }
                        next_item["qsvm_params"] = {
                            **next_item["qsvm_params"],
                            "feature_map": best_q_pt["feature_map"], "reps": best_q_pt["reps"],
                            "entanglement": best_q_pt["entanglement"], "C": best_q_pt["C"],
                            "class_weight": best_q_pt["class_weight"],
                        }

                    result = run_pair(dataset_cfg, next_item["classical_params"], next_item["qsvm_params"])
                    item_elapsed = time.time() - t_item
                    record_observed_time(next_item, result["classical"]["time"], result["qsvm"]["time"])
                    result["sweep_id"] = next_item.get("sweep_id")
                    append_history(result)
                    next_item["status"] = "done"
                    next_item["result"] = result
                    next_item["elapsed"] = item_elapsed
                except Exception as e:
                    next_item["status"] = "failed"
                    next_item["error"] = str(e)
            st.rerun()
    elif st.session_state.last_queue_total_time is not None:
        st.caption(f"Last queue run finished in {st.session_state.last_queue_total_time:.1f}s")

st.divider()
st.header("Run history")

history = load_history()
if not history:
    st.info("No runs yet — configure parameters in the sidebar and click Run.")
else:
    df = flatten_history(history).sort_values("timestamp", ascending=False).reset_index(drop=True)

    st.caption("Select rows to plot them below.")
    selection_event = st.dataframe(
        df,
        hide_index=True,
        on_select="rerun",
        selection_mode="multi-row",
        column_config={
            "timestamp": st.column_config.DatetimeColumn("Timestamp", format="MMM D, YYYY h:mm a"),
            "c_acc": st.column_config.NumberColumn("Classical acc", format="%.3f"),
            "q_acc": st.column_config.NumberColumn("QSVM acc", format="%.3f"),
            "c_f1": st.column_config.NumberColumn("Classical F1", format="%.3f"),
            "q_f1": st.column_config.NumberColumn("QSVM F1", format="%.3f"),
            "c_precision": st.column_config.NumberColumn("Classical precision", format="%.3f"),
            "q_precision": st.column_config.NumberColumn("QSVM precision", format="%.3f"),
            "c_recall": st.column_config.NumberColumn("Classical recall", format="%.3f"),
            "q_recall": st.column_config.NumberColumn("QSVM recall", format="%.3f"),
            "majority_count": st.column_config.NumberColumn("Majority count", format="%d"),
            "minority_count": st.column_config.NumberColumn("Minority count", format="%d"),
        },
    )

    if st.button("Clear history"):
        HISTORY_FILE.unlink(missing_ok=True)
        st.rerun()

    st.subheader("Metric vs. imbalance")

    _METRIC_COLUMNS = {"Accuracy": "acc", "F1": "f1", "Precision": "precision", "Recall": "recall"}
    metric_choice_2d = st.radio(
        "Metric", list(_METRIC_COLUMNS.keys()), key="metric_choice_2d", horizontal=True,
        help="F1/precision/recall matter more than accuracy when classes are imbalanced — "
             "a model can look good on accuracy while barely detecting the minority class.",
    )
    metric_col_2d = _METRIC_COLUMNS[metric_choice_2d]

    selected_rows = df.iloc[selection_event.selection.rows]
    plottable = selected_rows.dropna(subset=["imbalance_frac", "fixed_size"])
    skipped = len(selected_rows) - len(plottable)

    if selected_rows.empty:
        st.info("Select one or more runs in the table above to plot a metric vs. imbalance.")
    elif plottable.empty:
        st.warning("None of the selected runs used a fixed-size class imbalance, so there's nothing to plot.")
    else:
        plot_df = pd.concat([
            plottable.assign(model="Classical", value=plottable[f"c_{metric_col_2d}"]),
            plottable.assign(model="QSVM", value=plottable[f"q_{metric_col_2d}"]),
        ])[["imbalance_frac", "value", "fixed_size", "model"]]

        chart = alt.Chart(plot_df).mark_circle(opacity=0.75).encode(
            x=alt.X("imbalance_frac:Q", title="Class-0 fraction (imbalance)", scale=alt.Scale(domain=[-0.05, 1.05])),
            y=alt.Y("value:Q", title=metric_choice_2d, scale=alt.Scale(domain=[-0.05, 1.05])),
            color=alt.Color("model:N", title="Model"),
            size=alt.Size("fixed_size:Q", title="Training set size"),
            tooltip=["model", "imbalance_frac", "value", "fixed_size"],
        )
        st.altair_chart(chart)
        st.caption(
            f"Point size = training set size (smallest = 100 samples). "
            f"{len(plottable)} of {len(selected_rows)} selected runs plotted"
            + (f", {skipped} skipped (no fixed-size imbalance)." if skipped else ".")
        )

st.divider()
st.header("3D imbalance manifold")

sweeps = list_sweeps(history) if history else []
if not sweeps:
    st.info("No sweeps saved yet — use the '3D imbalance sweep' section above to queue and run one.")
else:
    sweep_options = {s["label"]: s["id"] for s in sweeps}
    selected_sweep_label = st.selectbox("Sweep", list(sweep_options.keys()), key="selected_sweep_label")
    selected_sweep_id = sweep_options[selected_sweep_label]

    sweep_records = [r for r in history if r.get("sweep_id") == selected_sweep_id]
    sweep_df = flatten_history(sweep_records)

    vis_col1, vis_col2, vis_col3 = st.columns([1, 1, 2])
    with vis_col1:
        show_classical_3d = st.checkbox("Show classical", value=True, key="show_classical_3d")
    with vis_col2:
        show_qsvm_3d = st.checkbox("Show QSVM", value=True, key="show_qsvm_3d")
    with vis_col3:
        metric_choice_3d = st.radio(
            "Z-axis metric", list(_METRIC_COLUMNS.keys()), key="metric_choice_3d", horizontal=True,
        )
    metric_col_3d = _METRIC_COLUMNS[metric_choice_3d]

    fig = go.Figure()
    if show_classical_3d:
        fig.add_trace(go.Scatter3d(
            x=sweep_df["fixed_size"], y=sweep_df["imbalance_frac"], z=sweep_df[f"c_{metric_col_3d}"],
            mode="markers", name="Classical",
            marker=dict(size=5, color="royalblue"),
        ))
    if show_qsvm_3d:
        fig.add_trace(go.Scatter3d(
            x=sweep_df["fixed_size"], y=sweep_df["imbalance_frac"], z=sweep_df[f"q_{metric_col_3d}"],
            mode="markers", name="QSVM",
            marker=dict(size=5, color="firebrick"),
        ))

    if not show_classical_3d and not show_qsvm_3d:
        st.info("Select at least one of classical / QSVM to plot.")
    else:
        fig.update_layout(
            scene=dict(
                xaxis_title="Training set size",
                yaxis_title="Class-0 fraction (imbalance)",
                zaxis_title=metric_choice_3d,
                yaxis=dict(range=[-0.05, 1.05]),
                zaxis=dict(range=[-0.05, 1.05]),
            ),
            height=650,
            legend=dict(orientation="h"),
            margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(fig)
        st.caption(
            f"{len(sweep_records)} point(s) in this sweep. Raw grid points shown — "
            f"surface smoothing/interpolation isn't implemented yet."
        )
