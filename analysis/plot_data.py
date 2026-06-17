import matplotlib.pyplot as plt
import scienceplots
import numpy as np
import pandas as pd


# Plots one or more line-sweep curves with optional error bars and reference lines.
def plot_line_sweep(
    x,
    curves: dict,
    xlabel,
    ylabel,
    save_path,
    title=None,
    xlog=False,
    show_std=False,
    annotate_std=False,
    figsize=(3.5, 2.4),
    legend_fontsize=5,
    xticklabels=None,
    include_legend=True,
    reference_lines=None,
):
    plt.style.use(["science", "ieee"])
    plt.rcParams["text.usetex"] = False

    fig, ax = plt.subplots(figsize=figsize)
    markers = ['o', 's', '^', 'd', 'x', '*']
    ref_linestyles = ['-', '--', ':', (0, (3, 1, 1, 1))]
    x = np.asarray(x, dtype=float)

    ax.tick_params(axis='both', which='both', top=False, right=False)

    for i, (label, stats) in enumerate(curves.items()):
        mean = np.asarray(stats["mean"], dtype=float)
        std = np.asarray(stats["std"], dtype=float)

        if show_std:
            line = ax.errorbar(
                x,
                mean,
                yerr=std,
                marker=markers[i % len(markers)],
                linestyle='-',
                linewidth=0.5,
                capsize=1,
                markersize=1,
                label=label
            )
        else:
            line = ax.plot(
                x,
                mean,
                marker=markers[i % len(markers)],
                linestyle='-',
                linewidth=1.0,
                markersize=4,
                label=label
            )

        if annotate_std and show_std:
            color = line[0].get_color()
            y_range = float(np.max(mean + std) - np.min(mean - std)) if len(mean) > 0 else 1.0
            offset = max(0.01 * y_range, 0.002)

            for xi, mu, sd in zip(x, mean, std):
                if i % 2 == 0:
                    y = mu + sd + offset
                    va = "bottom"
                else:
                    y = mu - sd - offset
                    va = "top"

                ax.text(
                    xi,
                    y,
                    f"±{sd:.3f}",
                    ha="center",
                    va=va,
                    fontsize=5,
                    color=color
                )

    if reference_lines is not None:
        for j, ref in enumerate(reference_lines):
            y = ref["y"]
            label = ref.get("label", None)
            linestyle = ref.get("linestyle", ref_linestyles[j % len(ref_linestyles)])
            linewidth = ref.get("linewidth", 1.0)
            color = ref.get("color", None)

            ax.axhline(
                y=y,
                linestyle=linestyle,
                linewidth=linewidth,
                color=color,
                label=label,
            )

            if "std" in ref and ref["std"] is not None:
                std = ref["std"]
                ax.fill_between(
                    x,
                    y - std,
                    y + std,
                    alpha=0.15,
                    color=color
                )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if xlog:
        ax.set_xscale("log")

    if xticklabels is not None:
        ax.set_xticks(x, labels=xticklabels)
    else:
        ax.set_xticks(x)

    if title:
        ax.set_title(title)

    ax.grid(True, alpha=0.3)
    if include_legend:
        ax.legend(frameon=True, fontsize=legend_fontsize)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# Loads a sweep result file and returns grouped summary statistics for a chosen metric.
def load_sweep_summary(path, metric="accuracy"):
    df = pd.read_json(path, lines=True)

    df = df.copy()
    df[metric] = df["metrics"].apply(lambda m: m[metric])

    summary = (
        df.groupby("sweep_value")[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values("sweep_value")
    )

    summary["std"] = summary["std"].fillna(0.0)
    return summary


# Loads a fixed experiment result file and returns overall summary statistics for a chosen metric.
def load_fixed_experiment_summary(path, metric="accuracy"):
    df = pd.read_json(path, lines=True)

    df = df.copy()
    df[metric] = df["metrics"].apply(lambda m: m[metric])

    summary = {
        "mean": float(df[metric].mean()),
        "std": float(df[metric].std(ddof=1) if len(df) > 1 else 0.0),
        "count": int(df[metric].count())
    }
    return summary


# Loads a sweep result file and returns grouped summary statistics for a chosen timing key.
def load_timing_summary(path, key="kernel_time"):
    df = pd.read_json(path, lines=True)

    df = df.copy()
    df["time"] = df["timing"].apply(lambda t: t.get(key, np.nan))

    summary = (
        df.groupby("sweep_value")["time"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values("sweep_value")
    )

    summary["std"] = summary["std"].fillna(0.0)
    return summary


# Converts a summary dataframe into the mean and standard deviation format 
# expected by the plotting functions.
def summary_to_curve(summary_df):
    return {
        "mean": summary_df["mean"].to_numpy(),
        "std": summary_df["std"].to_numpy(),
    }
