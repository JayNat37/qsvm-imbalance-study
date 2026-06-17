"""
Run Friedman tests and Wilcoxon signed-rank post-hoc tests on experiment results.

This script is used to analyse repeated-measures experiment outputs stored in
JSONL format.

To use the script, uncomment the relevant configuration block at the bottom
and run the file.
"""
import json
from collections import defaultdict
from itertools import combinations

from scipy.stats import friedmanchisquare, wilcoxon


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_nested_value(row, key):
    current = row
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def get_combined_strength_label(row):
    noise_type = get_nested_value(row, "qsvm.noise.type")
    depol = get_nested_value(row, "qsvm.noise.params.depol")
    amp = get_nested_value(row, "qsvm.noise.params.amp")
    phase = get_nested_value(row, "qsvm.noise.params.phase")

    if noise_type != "combined":
        return None
    if depol is None or amp is None or phase is None:
        return None

    depol = float(depol)
    amp = float(amp)
    phase = float(phase)

    eps = 1e-9

    if not (abs(depol - amp) < eps and abs(depol - phase) < eps):
        return None

    if abs(depol - 0.01) < eps:
        return "low"
    elif abs(depol - 0.05) < eps:
        return "medium"
    elif abs(depol - 0.10) < eps:
        return "high"
    else:
        return None


def holm_bonferroni(pairs_with_p):
    m = len(pairs_with_p)
    ordered = sorted(pairs_with_p, key=lambda x: x[1])

    results = []
    reject_so_far = True

    for i, (pair_name, raw_p) in enumerate(ordered):
        threshold = 0.05 / (m - i)

        reject = reject_so_far and (raw_p <= threshold)
        if not reject:
            reject_so_far = False

        results.append({
            "pair": pair_name,
            "raw_p": raw_p,
            "holm_threshold": threshold,
            "reject": reject,
        })

    return results


def prepare_paired_columns(rows,
                           metric_key,
                           block_key,
                           combined_mode=False,
                           condition_key="sweep_value"):
    block_map = defaultdict(dict)
    conditions = set()

    for row in rows:
        block = get_nested_value(row, block_key)

        if combined_mode:
            cond = get_combined_strength_label(row)
        else:
            cond = get_nested_value(row, condition_key)

        metric = get_nested_value(row, metric_key)

        if block is None or cond is None or metric is None:
            continue

        cond = str(cond)
        block_map[block][cond] = float(metric)
        conditions.add(cond)

    if combined_mode:
        order = {"low": 0, "medium": 1, "high": 2}
        conditions = sorted(conditions, key=lambda x: order[x])
    else:
        conditions = sorted(conditions, key=float)

    complete_blocks = []
    used_blocks = []

    for block in sorted(block_map):
        if all(cond in block_map[block] for cond in conditions):
            complete_blocks.append([block_map[block][cond] for cond in conditions])
            used_blocks.append(block)

    if len(conditions) < 3:
        raise ValueError("Need at least 3 conditions for Friedman.")
    if len(complete_blocks) < 2:
        raise ValueError("Not enough matched blocks.")

    columns = {cond: [] for cond in conditions}
    for row in complete_blocks:
        for cond, value in zip(conditions, row):
            columns[cond].append(value)

    return conditions, columns, used_blocks


def run_friedman_and_posthoc(path,
                             metric_key="metrics.accuracy",
                             block_key="repeat",
                             combined_mode=False,
                             condition_key="sweep_value"):
    rows = load_jsonl(path)
    conditions, columns, used_blocks = prepare_paired_columns(
        rows=rows,
        metric_key=metric_key,
        block_key=block_key,
        combined_mode=combined_mode,
        condition_key=condition_key,
    )

    friedman_inputs = [columns[c] for c in conditions]
    stat, p = friedmanchisquare(*friedman_inputs)

    print("\n=== Friedman Test ===")
    print("File:", path)
    print("Conditions:", conditions)
    print("Matched repeats used:", used_blocks)
    print()

    for cond in conditions:
        vals = columns[cond]
        mean_val = sum(vals) / len(vals)
        print(f"{cond:>10} | n = {len(vals):>2} | mean = {mean_val:.6f}")

    print()
    print(f"Friedman statistic = {stat:.6f}")
    print(f"p-value            = {p:.6g}")

    if p >= 0.05:
        print("Result: No significant overall difference. No post-hoc tests run.")
        return

    print("Result: Significant overall difference. Running pairwise Wilcoxon post-hoc tests.\n")

    raw_results = []
    for a, b in combinations(conditions, 2):
        x = columns[a]
        y = columns[b]

        w_stat, w_p = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
        raw_results.append((f"{a} vs {b}", w_p))

    holm_results = holm_bonferroni(raw_results)

    print("=== Pairwise Wilcoxon Signed-Rank Tests (Holm corrected) ===")
    for res in holm_results:
        decision = "significant" if res["reject"] else "not significant"
        print(
            f"{res['pair']:>20} | raw p = {res['raw_p']:.6g} "
            f"| Holm threshold = {res['holm_threshold']:.6g} | {decision}"
        )


# ----------------------------------------------------------------------
# Select the single-noise-channel experiment to analyse by uncommenting
# or replacing the active input_path with one of the options below.
# ----------------------------------------------------------------------

# Single channel noise experiments paths for moons:
# input_path = "results/moons/moons_b10_qsvm_noisyaer_depol_strength_sweep.jsonl"
# input_path = "results/moons/moons_b11_qsvm_noisyaer_amplitude_strength_sweep.jsonl"
# input_path = "results/moons/moons_b12_qsvm_noisyaer_phase_strength_sweep.jsonl"

# Single channel noise experiments paths for banknote:
# input_path = "results/banknote/banknote_b3_qsvm_noisyaer_depol_strength_sweep.jsonl"
# input_path = "results/banknote/banknote_b4_qsvm_noisyaer_amplitude_damping_sweep.jsonl"
# input_path = "results/banknote/banknote_b5_qsvm_noisyaer_phase_damping_sweep.jsonl"

# Single channel noise experiments paths for banknote + PCA:
# input_path = "results/banknote_pca/banknote_pca_p3_qsvm_noisyaer_depol_strength_sweep.jsonl"
# input_path = "results/banknote_pca/banknote_pca_p4_qsvm_noisyaer_amplitude_damping_sweep.jsonl"
# input_path = "results/banknote_pca/banknote_pca_p5_qsvm_noisyaer_phase_damping_sweep.jsonl"



input_path = "results/banknote_pca/banknote_pca_p3_qsvm_noisyaer_depol_strength_sweep.jsonl"
metric_key = "metrics.accuracy"
block_key = "repeat"
combined_mode = False
condition_key = "sweep_value"


# ----------------------------------------------------------------------
# Select the combined-noise experiment to analyse by uncommenting
# or replacing the active input_path with one of the options below.
# ----------------------------------------------------------------------
        
# Combined noise experiment paths:
# input_path = "results_combined_noise_all_per_dataset/moons_combined.jsonl"
# input_path = "results_combined_noise_all_per_dataset/banknote_combined.jsonl"
# input_path = "results_combined_noise_all_per_dataset/banknote_pca_combined.jsonl"

# input_path = "results_combined_noise_all_per_dataset/moons_combined.jsonl"
# metric_key = "metrics.accuracy"
# block_key = "repeat"
# combined_mode = True
# condition_key = "sweep_value"

run_friedman_and_posthoc(
    path=input_path,
    metric_key=metric_key,
    block_key=block_key,
    combined_mode=combined_mode,
    condition_key=condition_key,
)