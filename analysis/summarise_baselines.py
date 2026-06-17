"""
Summarise baseline experiment results stored in JSONL files and print
LaTeX-formatted table rows.
"""

import json
import statistics as stats

def summarise_jsonl(path):
    accuracies, precisions, recalls, f1s, bals = [], [], [], [], []

    with open(path, "r") as f:
        for line in f:
            row = json.loads(line)
            m = row["metrics"]
            accuracies.append(m["accuracy"])
            precisions.append(m["precision"])
            recalls.append(m["recall"])
            f1s.append(m["f1"])
            bals.append(m["balanced_accuracy"])

    return {
        "accuracy_mean": stats.mean(accuracies),
        "accuracy_std": stats.stdev(accuracies) if len(accuracies) > 1 else 0.0,
        "precision_mean": stats.mean(precisions),
        "recall_mean": stats.mean(recalls),
        "f1_mean": stats.mean(f1s),
        "balanced_accuracy_mean": stats.mean(bals),
    }

def latex_row(label, summary):
    return (
        f"{label} & "
        f"{summary['accuracy_mean']:.4f} $\\pm$ {summary['accuracy_std']:.4f} & "
        f"{summary['precision_mean']:.4f} & "
        f"{summary['recall_mean']:.4f} & "
        f"{summary['f1_mean']:.4f} & "
        f"{summary['balanced_accuracy_mean']:.4f} \\\\"
    )

# Uncomment the dataset group you want to summarise.

# files = [
#     ("results/moons/moons_b4_classical_baseline.jsonl", "Classical SVM"),
#     ("results/moons/moons_b5_qsvm_statevector_baseline.jsonl", "QSVM (Statevector)"),
#     ("results/moons/moons_b6_qsvm_shots_noisyaer_baseline.jsonl", "QSVM (Noisy Aer)"),
# ]

# files = [
#     ("results/banknote/banknote_b0_classical_baseline.jsonl", "Classical SVM"),
#     ("results/banknote/banknote_b1_qsvm_statevector_baseline.jsonl", "QSVM (Statevector)"),
#     ("results/banknote/banknote_b2_qsvm_shot_noise_baseline.jsonl", "QSVM (Noisy Aer)"),
# ]

files = [
    ("results/banknote_pca/banknote_pca_p0_classical_baseline.jsonl", "Classical SVM"),
    ("results/banknote_pca/banknote_pca_p1_qsvm_noiseless_baseline.jsonl", "QSVM (Statevector)"),
    ("results/banknote_pca/banknote_pca_p2_qsvm_shot_noise_baseline.jsonl", "QSVM (Noisy Aer)"),
]

for path, label in files:
    summary = summarise_jsonl(path)
    print(latex_row(label, summary))