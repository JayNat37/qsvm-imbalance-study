# Quantum Support Vector Machine Noise Analysis
This project investigates the impact of quantum noise on Quantum Support Vector Machines (QSVMs), comparing performance against classical SVM and noiseless QSVM baselines under controlled experimental conditions.

## Project Structure
```bash
├── analysis
│   ├── 01_plot_hyperparameter_selection_sweep.py
│   ├── 02_plot_dataset_noise.py
│   ├── 03_plot_physical_noise.py
│   ├── 04_plot_combined_noise.py
│   ├── plot_data.py
│   ├── statistical_tests
│   │   ├── friedman_and_wilcoxon.py
│   │   └── spearman_rank.py
│   └── summarise_baselines.py
├── configs
│   ├── banknote_dataset
│   │   ├── 0_baselines
│   │   │   ├── banknote_b0_classical_baseline.yaml
│   │   │   ├── banknote_b1_qsvm_statevector_baseline.yaml
│   │   │   └── banknote_b2_qsvm_shot_noise_baseline.yaml
│   │   ├── 1_physical_noise
│   │   │   ├── banknote_b3_qsvm_noisyaer_depol_strength_sweep.yaml
│   │   │   ├── banknote_b4_qsvm_noisyaer_amplitude_strength_sweep.yaml
│   │   │   └── banknote_b5_qsvm_noisyaer_phase_strength_sweep.yaml
│   │   └── 2_combined_noise
│   │       ├── banknote_b6_combined_low.yaml
│   │       ├── banknote_b7_combined_medium.yaml
│   │       └── banknote_b8_combined_high.yaml
│   ├── banknote_dataset_pca
│   │   ├── 0_baselines
│   │   │   ├── banknote_pca_p0_classical_baseline.yaml
│   │   │   ├── banknote_pca_p1_qsvm_noiseless_baseline.yaml
│   │   │   └── banknote_pca_p2_qsvm_shot_noise_baseline.yaml
│   │   ├── 1_physical_noise
│   │   │   ├── banknote_pca_p3_qsvm_noisyaer_depol_strength_sweep.yaml
│   │   │   ├── banknote_pca_p4_qsvm_noisyaer_amplitude_strength_sweep.yaml
│   │   │   └── banknote_pca_p5_qsvm_noisyaer_phase_strength_sweep.yaml
│   │   └── 2_combined_noise
│   │       ├── banknote_pca_p6_combined_low.yaml
│   │       ├── banknote_pca_p7_combined_medium.yaml
│   │       └── banknote_pca_p8_combined_high.yaml
│   ├── make_moons_dataset
│   │   ├── 0_hyperparameter_selection
│   │   │   ├── moons_b0_classical_C_sweep.yaml
│   │   │   ├── moons_b2_qsvm_statevector_reps_sweep.yaml
│   │   │   └── moons_b3_qsvm_shots_sweep.yaml
│   │   ├── 1_baselines
│   │   │   ├── moons_b4_classical_baseline.yaml
│   │   │   ├── moons_b5_qsvm_noiseless_baseline.yaml
│   │   │   └── moons_b6_qsvm_shot_noisyaer_baseline.yaml
│   │   ├── 2_dataset_noise_sweep
│   │   │   ├── moons_b7_classical_dataset_noise_sweep.yaml
│   │   │   ├── moons_b8_qsvm_statevector_dataset_noise_sweep.yaml
│   │   │   └── moons_b9_qsvm_noisyaer_shotonly_dataset_noise_sweep.yaml
│   │   ├── 3_physical_noise
│   │   │   ├── moons_b10_qsvm_noisyaer_depol_strength_sweep.yaml
│   │   │   ├── moons_b11_qsvm_noisyaer_amplitude_strength_sweep.yaml
│   │   │   └── moons_b12_qsvm_noisyaer_phase_strength_sweep.yaml
│   │   └── 4_combined_physical_noise
│   │       ├── moons_b13_combined_low.yaml
│   │       ├── moons_b14_combined_medium.yaml
│   │       └── moons_b15_combined_high.yaml
│   ├── mnist_imbalance
│   │   ├── mnist_imbalance_frac50_classical_svm.yaml
│   │   ├── mnist_imbalance_frac50_qsvm_noiseless.yaml
│   │   ├── mnist_imbalance_frac60_classical_svm.yaml
│   │   ├── mnist_imbalance_frac60_qsvm_noiseless.yaml
│   │   ├── mnist_imbalance_frac70_classical_svm.yaml
│   │   ├── mnist_imbalance_frac70_qsvm_noiseless.yaml
│   │   ├── mnist_imbalance_frac80_classical_svm.yaml
│   │   ├── mnist_imbalance_frac80_qsvm_noiseless.yaml
│   │   ├── mnist_imbalance_frac90_classical_svm.yaml
│   │   └── mnist_imbalance_frac90_qsvm_noiseless.yaml
│   └── run_matrix.yaml
├── data
│   └── data_banknote_authentication.txt
├── environment.yml
├── plots
│   ├── 01_hyperparameter
│   │   ├── moons_b0_classical_C_sweep.png
│   │   ├── moons_b1_qsvm_statevector_C_sweep.png
│   │   ├── moons_b2_qsvm_statevector_reps_sweep.png
│   │   └── moons_b3_qsvm_shots_sweep_accuracy.png
│   ├── 02_dataset_noise
│   │   ├── dataset_noise_accuracy.png
│   │   └── kernel_time_vs_dataset_noise_qsvm.png
│   ├── 03_single_noise
│   │   ├── amplitude_accuracy_by_dataset.png
│   │   ├── amplitude_kernel_time_by_dataset.png
│   │   ├── banknote_channels_vs_baseline_accuracy.png
│   │   ├── banknote_pca_channels_vs_baseline_accuracy.png
│   │   ├── depol_accuracy_by_dataset.png
│   │   ├── depol_kernel_time_by_dataset.png
│   │   ├── moons_channels_vs_baseline_accuracy.png
│   │   ├── phase_accuracy_by_dataset.png
│   │   └── phase_kernel_time_by_dataset.png
│   └── 04_combined_noise
│       └── combined_noise_accuracy_by_dataset.png
├── README.md
├── requirements.txt
└── src
    ├── __init__.py
    ├── classical
    │   ├── __init__.py
    │   └── classical_svm.py
    ├── datasets
    │   ├── __init__.py
    │   ├── config.py
    │   ├── loaders.py
    │   └── preprocess.py
    ├── experiments
    │   ├── __init__.py
    │   ├── config_utils.py
    │   ├── metrics.py
    │   ├── run_experiment.py
    │   └── run_matrix.py
    ├── qsvm
    │   ├── backends
    │   │   ├── noisy_aer.py
    │   │   └── statevector.py
    │   ├── config.py
    │   ├── feature_maps.py
    │   ├── kernel.py
    │   └── noise.py
    └── ui
        ├── app.py
        └── tuning.py
```

# User Guide

### Setting Up the Environment

Before running any experiments, the project environment must be created
and the required dependencies installed. The implementation was
developed in Python using a Conda environment to manage packages. This
guide will provide two options for setting up the environment.

#### Using a Conda Environment

First, create and activate a new Conda environment:

``` {#lst:conda_setup caption="Creating and activating the Conda environment." label="lst:conda_setup"}
conda create --name <qsvm-env-name> python=3.10
conda activate <qsvm-env-name>
```

Next, install the required dependencies from the `requirements.txt` file
provided in the repository:

``` {#lst:install_requirements caption="Installing project dependencies." label="lst:install_requirements"}
pip -m install -r requirements.txt
```

Alternatively, you can reate a Conda environment directly from the
`environment.yml` file:

``` {#lst:conda_env_file caption="Creating the Conda environment from the environment file." label="lst:conda_env_file"}
conda env create -f environment.yml
conda activate qsvm
```

#### Using a Python Virtual Environment

First, create and activate a virtual environment:

``` {#lst:pip_setup caption="Creating and activating the virtual environment." label="lst:pip_setup"}
python3.10 -m venv <qsvm-env-name>
source <qsvm-env-name>/bin/activate
```

Next, install the required dependencies from the `requirements.txt`
file:

``` {#lst:install_requirements caption="Installing project dependencies." label="lst:install_requirements"}
pip -m install -r requirements.txt
```

Once these steps are complete, the environment is ready and experiments
can be executed using the configuration files described in the following
sections.

### Creating an Experiment Configuration

Experiments are specified using YAML configuration files located in the
`qsvm-noise-project/configs` directory. Each configuration file defines
the dataset, model type, repetition settings, and, where required, a
parameter sweep. Rather than requiring code modification, new
experiments can be created by editing these configuration files.

## Configuration Parameters

| Section | Parameter | Description |
|---|---|---|
| General | `name` | Unique experiment name and output file prefix. |
| Dataset | `name` | Dataset identifier. |
| Dataset | `params` | Dataset-specific parameters. |
| Dataset | `test_size` | Fraction reserved for testing. |
| Dataset | `split_seed` | Random seed for train-test splitting. |
| Dataset | `scale` | Optional preprocessing flag. |
| Dataset | `pca_components` | Optional PCA dimensionality. |
| Dataset | `max_samples` | Optional cap on the pool size before splitting. |
| Dataset | `minmax_scale` | Optional MinMax scaling to `[0, 1]` after PCA. |
| Dataset | `cap_imbalance_to_available` | If `True`, fixed-size imbalance sampling reduces to whatever class counts are available instead of raising. |
| Dataset `params` | `fixed_size` | Fixed training-set size for class-imbalance sampling. |
| Dataset `params` | `imbalance_fraction` | Target class-0 fraction of the fixed-size training set. |
| Dataset `params` | `class_a`, `class_b` | Digit/category pair to compare, for `mnist_01` and `fashion_mnist` (default `0`, `1`). |
| Model | `kind` | Either `classical_svm` or `qsvm`. |
| Model (Classical) | `class_weight` | `None` or `balanced`; reweights the misclassification penalty inversely to class frequency. |
| Model (QSVM) | `feature_map` | Quantum feature map: `zz` or `pauli`. |
| Model (QSVM) | `reps` | Number of repetitions of the feature map circuit. |
| Model (QSVM) | `entanglement` | Entanglement pattern: `full`, `linear`, or `circular`. |
| Model (QSVM) | `mode` | Kernel evaluation mode: `statevector` or `noisy_aer`. |
| Model (QSVM) | `shots` | Number of simulator shots where applicable. |
| Model (QSVM) | `seed` | Random seed for simulator-based evaluation. |
| Model (QSVM) | `scale_to_pi` | Whether input features are scaled before encoding. |
| Model (QSVM) | `C` | SVM regularisation parameter. |
| Model (Noise) | `type` | Noise model type: `depolarizing`, `amplitude_damping`, `phase_damping`, or `combined`. |
| Model (Noise) | `strength` | Generic noise strength parameter. |
| Model (Noise) | `params` | Optional channel-specific parameters. |
| Model (Noise) | `apply_to` | Scope of application: `all`, `single_qubit`, or `two_qubit`. |
| Runs | `repeats` | Number of repeated runs. |
| Runs | `vary_seed` | Whether the seed changes between repeats. |
| Runs | `seed_start` | Initial seed value for repeated runs. |
| Sweep | `path` | Dot-separated parameter path to vary. |
| Sweep | `values` | List of values used in the sweep. |

*Configuration parameters supported by the experimental framework.*


### Running an Experiment

Experiments are executed from the command line after activating the
appropriate Python environment.

A single experiment configuration can be run with:

``` {#lst:run_single_experiment caption="Running a single experiment configuration." label="lst:run_single_experiment"}
python -m src.experiments.run_experiment path/to/config.yaml
```

This executes the experiment defined in the specified YAML file and
writes the output to the `results/` directory.

To execute multiple configurations automatically, a run matrix may be
used:

``` {#lst:run_matrix caption="Running multiple experiments using a run matrix." label="lst:run_matrix"}
python -m src.experiments.run_matrix configs/run_matrix.yaml
```

This is useful for reproducing larger groups of experiments in a single
command.

The results of each experiment are written to the
`qsvm-noise-project/results` directory. The output file name is
determined by the `name` field defined at the top of the corresponding
experiment configuration file.

### MNIST Class-Imbalance Replication Study

`configs/mnist_imbalance/` reproduces a class-imbalance experiment from
Josiah Chan's dissertation ([`qml-dissertation`](https://github.com/josiahwkc/qml-dissertation)):
classical SVM vs. QSVM on MNIST digits 0-vs-1, PCA-reduced to 5
components, with a fixed 100-sample training set swept across class-0
fractions of 0.5, 0.6, 0.7, 0.8, and 0.9. Ten YAML configs (one classical,
one QSVM per fraction) can be run individually with `run_experiment`, as
described above.

The dataset itself (`mnist_01`, and its generalisation `fashion_mnist`)
is fetched from OpenML on first use and cached locally by scikit-learn.
Both accept `class_a`/`class_b` parameters to compare any two digit or
category labels, not just 0 vs. 1.

The QSVM configs use `mode: noisy_aer` with 1024 shots and no noise
channel, and `minmax_scale: true` with `scale_to_pi: false` on the
dataset, matching the shot-sampled kernel and `[0, 1]` feature encoding
used in the original study — rather than this project's exact
`statevector` kernel and `[-π, π]` encoding used elsewhere. The classical
`C`/`gamma` values were obtained by running the original dissertation's
own tuning code against its own MNIST data (see `src/ui/tuning.py` for
an equivalent, reusable tuner built into this project's interactive UI).

### Producing Plots and Statistical Analysis

All plotting and statistical analysis scripts are located in the
`qsvm-noise-project/analysis` directory. The scripts are organised into
subfolders according to the experiment group they analyse, so that plots
and statistical outputs for each set of experiments are kept separate
and easier to navigate.

#### Producing Baseline Summaries

To generate LaTeX-formatted summary rows for the Classical SVM,
Statevector QSVM, and Noisy-Aer QSVM baselines, run:

    python analysis/summarise_baselines.py

The script contains separate file-path blocks for each dataset. To
produce the summary for a particular dataset, uncomment the
corresponding block in the script and leave the others commented out.

#### Running Plot Scripts

To generate plots, run the required plotting scripts from the project
root.

    python analysis/01_plot_hyperparameter_selection_sweep.py 

    python analysis/02_plot_dataset_noise.py

    python analysis/03_plot_physical_noise.py

    python analysis/04_plot_combined_noise.py

Each plotting script reads the appropriate result files from the
`results` directory and writes the generated figures to its specified
output location, in the `plots` directory.

#### Running Statistical Analysis

Statistical analysis scripts require the user to manually update the
relevant input file paths and configuration variables before execution.

##### Friedman Test and Wilcoxon Post-hoc Analysis

The user must specify the input file path and the relevant configuration
variables in the hardcoded settings section of the script. For standard
sweeps, the script uses the `sweep_value` field directly to identify the
experimental conditions. For combined-noise experiments, however,
multiple noise-strength settings are stored together within a single
`.jsonl` file, so the script must be run with `combined_mode = True`. In
this mode, the script reconstructs categorical condition labels such as
`low`, `medium`, and `high` from the depolarising, amplitude damping,
and phase damping parameters stored in each result entry.

For example:

    input_path = "results_combined_noise_all_per_dataset/banknote_pca_combined.jsonl"
    metric_key = "metrics.accuracy"
    block_key = "repeat"
    combined_mode = True
    condition_key = "sweep_value"

If `combined_mode = False`, the script treats the file as a standard
sweep and uses `sweep_value` directly. After updating these settings,
the script can be executed from the project root:

    python analysis/statistical_tests/friedman_and_wilcoxon.py

For ease of use, the script includes commented file paths for all
supported statistical test inputs. To analyse a particular experiment,
replace or uncomment the relevant `input_path`, ensure the corresponding
configuration block is active, and then run the script.

##### Spearman Rank Correlation Analysis

The script expects one file for the classical SVM, one for the
Statevector QSVM, and one for the shot-based QSVM.

For example:

    classical_path = "results/moons/moons_b7_classical_dataset_noise_sweep.jsonl"
    statevector_path = "results/moons/moons_b8_qsvm_statevector_dataset_noise_sweep.jsonl"
    shot_path = "results/moons/moons_b9_qsvm_noisyaer_shotonly_dataset_noise_sweep
        .jsonl"

The test can then be run as follows:

    python analysis/statistical_tests/spearman_rank.py

For ease of use, the script includes commented file paths for all
supported statistical test inputs. To analyse a particular experiment,
replace or uncomment the relevant file paths block, and then run the
script.

In both cases, statistical results are printed to the terminal.

## Interactive UI

`src/ui/app.py` is a [Streamlit](https://streamlit.io) app for running
classical-SVM-vs-QSVM comparisons interactively, without hand-writing a
YAML config. It reuses the same dataset, preprocessing, classical, and
QSVM code paths as `run_experiment.py`.

Run it from the project root with the environment active:

    streamlit run src/ui/app.py

Features:

- **Datasets** — `mnist_01`, `fashion_mnist`, `breast_cancer`, `banknote`,
  and `make_moons`. For `mnist_01` and `fashion_mnist`, any two digit or
  category labels can be compared (not just the default 0 vs. 1), via
  `class_a`/`class_b` fields in the sidebar.
- **Single runs** — configure dataset, preprocessing, classical SVM, and
  QSVM parameters in the sidebar and run a paired classical/QSVM
  comparison immediately.
- **Queue** — add several parameter combinations to a queue and run them
  back-to-back, with a live elapsed-time / estimated-time-remaining
  display and the ability to remove not-yet-started items.
- **Tune hyperparameters** — grid search classical (`C`, `gamma`,
  `class_weight`) and QSVM (`reps`, `entanglement`, `C`) hyperparameters
  against a validation split carved out of the current training pool
  (the QSVM search uses the exact `statevector` kernel for speed); apply
  the best combination found directly to the sidebar with one click. The
  grid itself is configurable via the "Grid settings" popover.
- **3D imbalance sweep** — sweep training-set size × class-0 fraction over
  a grid (min/max/step for each axis), queuing one classical+QSVM pair per
  grid point; optionally re-tune hyperparameters at every point instead of
  reusing one fixed configuration across the whole sweep.
- **3D imbalance manifold** — pick any past sweep and view it as an
  interactive, rotatable Plotly 3D scatter (size × imbalance × accuracy),
  with independent toggles for showing classical and/or QSVM. Every sweep
  point is saved permanently, so past sweeps can be revisited without
  re-running them.
- **Run history** — every run (single, queued, or from a sweep) is logged
  to `results/ui_run_history.jsonl` and shown in a sortable, selectable
  table; select rows to plot accuracy vs. class-imbalance fraction, with
  point size encoding training-set size.

`results/ui_run_history.jsonl` is local exploration history rather than a
designed experiment output, so it is excluded from version control.
