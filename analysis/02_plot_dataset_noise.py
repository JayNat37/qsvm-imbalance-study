from plot_data import plot_line_sweep, load_sweep_summary, load_timing_summary

# Plots mean accuracy against dataset noise for the classical SVM, Statevector QSVM, and Shot-based QSVM.
def plot_mean_accuracy_dataset_noise():
    classical_synthetic = load_sweep_summary(
        "results/moons/moons_b7_classical_dataset_noise_sweep.jsonl"
    ).sort_values("sweep_value")

    statevector_synthetic = load_sweep_summary(
        "results/moons/moons_b8_qsvm_statevector_dataset_noise_sweep.jsonl"
    ).sort_values("sweep_value")

    shot_noise_synthetic = load_sweep_summary(
        "results/moons/moons_b9_qsvm_noisyaer_shotonly_dataset_noise_sweep.jsonl"
    ).sort_values("sweep_value")

    x = classical_synthetic["sweep_value"].values

    curves = {
        "Classical SVM": {
            "mean": classical_synthetic["mean"].values,
            "std": classical_synthetic["std"].values
        },
        "QSVM (Statevector)": {
            "mean": statevector_synthetic["mean"].values,
            "std": statevector_synthetic["std"].values
        },
        "QSVM (Shot-based)": {
            "mean": shot_noise_synthetic["mean"].values,
            "std": shot_noise_synthetic["std"].values
        }
    }

    plot_line_sweep(
        x=x,
        curves=curves,
        xlabel="Dataset Noise",
        ylabel="Mean Accuracy",
        save_path="plots/02_dataset_noise/dataset_noise_accuracy.png",
        show_std=True,
        annotate_std=False,
    )


# Plots training time and kernel evaluation time against dataset noise for the classical SVM, 
# Statevector QSVM, and Shot-based QSVM.
def plot_mean_training_time_dataset_noise():
    classical_timing = load_timing_summary(
        "results/moons/moons_b7_classical_dataset_noise_sweep.jsonl",
        key="fit_time"
    ).sort_values("sweep_value")

    statevector_timing = load_timing_summary(
        "results/moons/moons_b8_qsvm_statevector_dataset_noise_sweep.jsonl",
        key="kernel_time"
    ).sort_values("sweep_value")

    shot_timing = load_timing_summary(
        "results/moons/moons_b9_qsvm_noisyaer_shotonly_dataset_noise_sweep.jsonl",
        key="kernel_time"
    ).sort_values("sweep_value")

    x = statevector_timing["sweep_value"].values

    curves = {
        "Classical SVM": {
            "mean": classical_timing["mean"].values,
            "std": classical_timing["std"].values
        },
        "QSVM (Statevector)": {
            "mean": statevector_timing["mean"].values,
            "std": statevector_timing["std"].values
        },
        "QSVM (Shot-based)": {
            "mean": shot_timing["mean"].values,
            "std": shot_timing["std"].values
        }
    }
    save_path = "plots/02_dataset_noise/kernel_time_vs_dataset_noise_qsvm.png"


    plot_line_sweep(
        x=x,
        curves=curves,
        xlabel="Dataset noise",
        ylabel="Training Time (seconds)",
        save_path=save_path,
        show_std=True,
        annotate_std=False,
    )


if __name__ == "__main__":
    plot_mean_accuracy_dataset_noise()
    plot_mean_training_time_dataset_noise()