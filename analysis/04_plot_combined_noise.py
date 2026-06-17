import numpy as np
import pandas as pd
from pathlib import Path

from plot_data import (
    load_fixed_experiment_summary,
    plot_line_sweep,
)

PLOTS_DIR = Path("plots/04_combined_noise")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Result file paths
RESULTS = {
    "moons": {
        "low": "results/moons/moons_b13_combined_low.jsonl",
        "medium": "results/moons/moons_b14_combined_medium.jsonl",
        "high": "results/moons/moons_b15_combined_high.jsonl",
        "baseline": "results/moons/moons_b6_qsvm_shots_noisyaer_baseline.jsonl",
    },
    "banknote": {
        "low": "results/banknote/banknote_b6_combined_low.jsonl",
        "medium": "results/banknote/banknote_b7_combined_medium.jsonl",
        "high": "results/banknote/banknote_b8_combined_high.jsonl",
        "baseline": "results/banknote/banknote_b2_qsvm_shot_noise_baseline.jsonl",
    },
    "banknote_pca": {
        "low": "results/banknote_pca/banknote_pca_p6_combined_low.jsonl",
        "medium": "results/banknote_pca/banknote_pca_p7_combined_medium.jsonl",
        "high": "results/banknote_pca/banknote_pca_p8_combined_high.jsonl",
        "baseline": "results/banknote_pca/banknote_pca_p2_qsvm_shot_noise_baseline.jsonl",
    },
}

DATASET_LABELS = {
    "moons": "Moons",
    "banknote": "Banknote",
    "banknote_pca": "Banknote + PCA",
}

CHANNEL_LABELS = {
    "depol": "Depolarizing",
    "amplitude": "Amplitude damping",
    "phase": "Phase damping",
}

# Converts fixed low, medium, and high summaries into a curve format for plotting.
def fixed_summaries_to_curve(low, medium, high):
    return {
        "mean": np.array([low["mean"], medium["mean"], high["mean"]], dtype=float),
        "std": np.array([low["std"], medium["std"], high["std"]], dtype=float),
    }


# Plots mean accuracy across datasets for the low, medium, and high combined noise settings.
def plot_combined_noise_accuracy():
    x = np.array([0, 1, 2], dtype=float)
    x_labels = ["Low", "Medium", "High"]

    moons_low = load_fixed_experiment_summary(RESULTS["moons"]["low"], metric="accuracy")
    moons_medium = load_fixed_experiment_summary(RESULTS["moons"]["medium"], metric="accuracy")
    moons_high = load_fixed_experiment_summary(RESULTS["moons"]["high"], metric="accuracy")

    banknote_low = load_fixed_experiment_summary(RESULTS["banknote"]["low"], metric="accuracy")
    banknote_medium = load_fixed_experiment_summary(RESULTS["banknote"]["medium"], metric="accuracy")
    banknote_high = load_fixed_experiment_summary(RESULTS["banknote"]["high"], metric="accuracy")

    banknote_pca_low = load_fixed_experiment_summary(RESULTS["banknote_pca"]["low"], metric="accuracy")
    banknote_pca_medium = load_fixed_experiment_summary(RESULTS["banknote_pca"]["medium"], metric="accuracy")
    banknote_pca_high = load_fixed_experiment_summary(RESULTS["banknote_pca"]["high"], metric="accuracy")

    curves = {
        "Moons": fixed_summaries_to_curve(moons_low, moons_medium, moons_high),
        "Banknote": fixed_summaries_to_curve(banknote_low, banknote_medium, banknote_high),
        "Banknote + PCA": fixed_summaries_to_curve(
            banknote_pca_low, banknote_pca_medium, banknote_pca_high
        ),
    }

    plot_line_sweep(
        x=x,
        curves=curves,
        xlabel="Combined noise strength",
        ylabel="Mean Accuracy",
        save_path=PLOTS_DIR / "combined_noise_accuracy_by_dataset.png",
        show_std=False,
        annotate_std=False,
        xticklabels=x_labels,
    )


if __name__ == "__main__":
    plot_combined_noise_accuracy()
