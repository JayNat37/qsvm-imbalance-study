"""
Compute Spearman’s rank correlation between dataset noise level and model accuracy.

This script is used for the dataset noise experiments to assess the
monotonic relationship between increasing dataset noise (sweep_value)
and classification performance for each model.
"""
import json
import pandas as pd
from scipy.stats import spearmanr


def load_jsonl(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def extract_results(path, model_label):
    rows = load_jsonl(path)
    out = []

    for r in rows:
        out.append({
            "model": model_label,
            "noise": float(r["sweep_value"]),
            "repeat": int(r["repeat"]),
            "accuracy": float(r["metrics"]["accuracy"])
        })

    return pd.DataFrame(out)


# File paths for the three models' dataset noise sweeps on the moons dataset
classical_path = "results/moons/moons_b7_classical_dataset_noise_sweep.jsonl"
statevector_path = "results/moons/moons_b8_qsvm_statevector_dataset_noise_sweep.jsonl"
shot_path = "results/moons/moons_b9_qsvm_noisyaer_shotonly_dataset_noise_sweep.jsonl"

df_classical = extract_results(classical_path, "Classical SVM")
df_state = extract_results(statevector_path, "Statevector QSVM")
df_shot = extract_results(shot_path, "Shot-based QSVM")

df = pd.concat([df_classical, df_state, df_shot], ignore_index=True)

print("\n=== Spearman using mean accuracy at each noise level ===")
means = (
    df.groupby(["model", "noise"], as_index=False)
      .agg(mean_accuracy=("accuracy", "mean"))
)

for model_name, g in means.groupby("model"):
    rho, p = spearmanr(g["noise"], g["mean_accuracy"])
    print(f"{model_name}: rho = {rho:.4f}, p = {p:.6f}")