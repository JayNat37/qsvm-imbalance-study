from __future__ import annotations

from pathlib import Path

import numpy as np

from plot_data import (
    load_sweep_summary,
    load_timing_summary,
    load_fixed_experiment_summary,
    plot_line_sweep,
    summary_to_curve,
)

PLOTS_DIR = Path("plots/03_single_noise")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)



# Result file paths
RESULTS = {
    "moons": {
        "depol": "results/moons/moons_b10_qsvm_noisyaer_depol_strength_sweep.jsonl",
        "amplitude": "results/moons/moons_b11_qsvm_noisyaer_amplitude_strength_sweep.jsonl",
        "phase": "results/moons/moons_b12_qsvm_noisyaer_phase_strength_sweep.jsonl",
        "baseline": "results/moons/moons_b6_qsvm_shots_noisyaer_baseline.jsonl",
        "statevector_baseline": "results/moons/moons_b5_qsvm_statevector_baseline.jsonl",
    },
    "banknote": {
        "depol": "results/banknote/banknote_b3_qsvm_noisyaer_depol_strength_sweep.jsonl",
        "amplitude": "results/banknote/banknote_b4_qsvm_noisyaer_amplitude_damping_sweep.jsonl",
        "phase": "results/banknote/banknote_b5_qsvm_noisyaer_phase_damping_sweep.jsonl",
        "baseline": "results/banknote/banknote_b2_qsvm_shot_noise_baseline.jsonl",
        "statevector_baseline": "results/banknote/banknote_b1_qsvm_statevector_baseline.jsonl",
    },
    "banknote_pca": {
        "depol": "results/banknote_pca/banknote_pca_p3_qsvm_noisyaer_depol_strength_sweep.jsonl",
        "amplitude": "results/banknote_pca/banknote_pca_p4_qsvm_noisyaer_amplitude_damping_sweep.jsonl",
        "phase": "results/banknote_pca/banknote_pca_p5_qsvm_noisyaer_phase_damping_sweep.jsonl",
        "baseline": "results/banknote_pca/banknote_pca_p2_qsvm_shot_noise_baseline.jsonl",
        "statevector_baseline": "results/banknote_pca/banknote_pca_p1_qsvm_noiseless_baseline.jsonl",
    },
}

CHANNEL_LABELS = {
    "depol": "Depolarizing",
    "amplitude": "Amplitude damping",
    "phase": "Phase damping",
}

DATASET_LABELS = {
    "moons": "Moons",
    "banknote": "Banknote",
    "banknote_pca": "Banknote + PCA",
}


# Plots mean accuracy across datasets for a given noise channel.
def plot_accuracy_by_channel(channel: str):
    moons = load_sweep_summary(RESULTS["moons"][channel], metric="accuracy")
    banknote = load_sweep_summary(RESULTS["banknote"][channel], metric="accuracy")
    banknote_pca = load_sweep_summary(RESULTS["banknote_pca"][channel], metric="accuracy")

    x_labels = ["0.01", "0.05", "0.1"]
    x_pos = np.arange(len(x_labels))

    curves = {
        "Moons": summary_to_curve(moons),
        "Banknote": summary_to_curve(banknote),
        "Banknote + PCA": summary_to_curve(banknote_pca),
    }

    plot_line_sweep(
        x=x_pos,
        curves=curves,
        xlabel="Noise strength",
        ylabel="Mean Accuracy",
        save_path=PLOTS_DIR / f"{channel}_accuracy_by_dataset.png",
        xlog=False,
        show_std=True,
        xticklabels=x_labels,
    )


# Plots all noise channels against the baseline for a given dataset.
def plot_dataset_channels_vs_baseline(dataset: str):
    depol = load_sweep_summary(RESULTS[dataset]["depol"], metric="accuracy")
    amplitude = load_sweep_summary(RESULTS[dataset]["amplitude"], metric="accuracy")
    phase = load_sweep_summary(RESULTS[dataset]["phase"], metric="accuracy")

    x_labels = ["0.01", "0.05", "0.1"]
    x_pos = np.arange(len(x_labels))

    baseline_summary = load_fixed_experiment_summary(
        RESULTS[dataset]["baseline"], metric="accuracy"
    )
    statevector_baseline_summary = load_fixed_experiment_summary(
        RESULTS[dataset]["statevector_baseline"], metric="accuracy"
    )

    curves = {
        "Depolarizing": summary_to_curve(depol),
        "Amplitude damping": summary_to_curve(amplitude),
        "Phase damping": summary_to_curve(phase),
    }

    reference_lines = [
        {
            "y": baseline_summary["mean"],
            "std": baseline_summary["std"],
            "label": "Shot-noise baseline",
            "linewidth": 1.2,
        },
        {
            "y": statevector_baseline_summary["mean"],
            "std": statevector_baseline_summary["std"],
            "label": "Statevector baseline",
            "linewidth": 1.2,
        }
    ]

    plot_line_sweep(
        x=x_pos,
        curves=curves,
        xlabel="Noise strength",
        ylabel="Mean Accuracy",
        save_path=PLOTS_DIR / f"{dataset}_channels_vs_baseline_accuracy.png",
        xlog=False,
        show_std=False,
        xticklabels=x_labels,
        reference_lines=reference_lines,
    )


# Plots mean kernel evaluation time across datasets for a given noise channel.
def plot_kernel_time_by_channel(channel: str):
    moons = load_timing_summary(RESULTS["moons"][channel], key="kernel_time")
    banknote = load_timing_summary(RESULTS["banknote"][channel], key="kernel_time")
    banknote_pca = load_timing_summary(RESULTS["banknote_pca"][channel], key="kernel_time")

    x_labels = ["0.01", "0.05", "0.1"]
    x_pos = np.arange(len(x_labels))

    curves = {
        "Moons": summary_to_curve(moons),
        "Banknote": summary_to_curve(banknote),
        "Banknote + PCA": summary_to_curve(banknote_pca),
    }

    plot_line_sweep(
        x=x_pos,
        curves=curves,
        xlabel="Noise strength",
        ylabel="Mean kernel time (s)",
        save_path=PLOTS_DIR / f"{channel}_kernel_time_by_dataset.png",
        xlog=False,
        show_std=True,
        xticklabels=x_labels,
        legend_fontsize=7,
    )


if __name__ == "__main__":
    # Accuracy vs strength for each channel
    for channel in ["depol", "amplitude", "phase"]:
        plot_accuracy_by_channel(channel)

    # For each dataset: all channels + baseline
    for dataset in ["moons", "banknote", "banknote_pca"]:
        plot_dataset_channels_vs_baseline(dataset)

    # Kernel time vs strength
    for channel in ["depol", "amplitude", "phase"]:
        plot_kernel_time_by_channel(channel)
