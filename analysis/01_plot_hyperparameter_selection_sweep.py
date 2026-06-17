from plot_data import load_sweep_summary, plot_line_sweep

# Plots the classical SVM accuracy across different C values.
def plot_classical_C_sweep():
    summary = load_sweep_summary("results/moons/moons_b0_classical_C_sweep.jsonl")

    plot_line_sweep(
        x=summary["sweep_value"].values,
        curves={
            "Classical SVM": {
                "mean": summary["mean"].values,
                "std": summary["std"].values,
            }
        },
        xlabel="Regularisation Parameter (C)",
        ylabel="Mean Accuracy",
        save_path="plots/01_hyperparameter/moons_b0_classical_C_sweep.png",
        xlog=True,
        xticklabels=["0.1", "1", "10", "100"],
        include_legend=False
    )

# Plots the noiseless QSVM accuracy across different feature map repetition values.
def plot_noiseless_qsvm_reps_sweep():
    summary = load_sweep_summary("results/moons/moons_b2_qsvm_statevector_reps_sweep.jsonl")

    plot_line_sweep(
        x=summary["sweep_value"].values,
        curves={
            "QSVM": {
                "mean": summary["mean"].values,
                "std": summary["std"].values,
            }
        },
        xlabel="Feature Map Repetitions (reps)",
        ylabel="Mean Accuracy",
        save_path="plots/01_hyperparameter/moons_b2_qsvm_statevector_reps_sweep.png",
        xlog=False,
        include_legend=False
    )

# Plots the shot-based QSVM accuracy across different numbers of measurement shots.
def plot_shot_noise_qsvm_shots_sweep():
    summary = load_sweep_summary("results/moons/moons_b3_qsvm_shots_sweep.jsonl")
    plot_line_sweep(
        x=summary["sweep_value"].values,
        curves={
            "QSVM": {
                "mean": summary["mean"].values,
                "std": summary["std"].values,
            }
        },
        xlabel="Number of Shots",
        ylabel="Mean Accuracy",
        save_path="plots/01_hyperparameter/moons_b3_qsvm_shots_sweep_accuracy.png",
        show_std=True,
        annotate_std=True,
        xlog=True,
        xticklabels=["128", "256", "512", "1024", "2048", "4096"],
        include_legend=False
    )


if __name__ == "__main__":
    plot_classical_C_sweep()
    plot_noiseless_qsvm_reps_sweep()
    plot_shot_noise_qsvm_shots_sweep()