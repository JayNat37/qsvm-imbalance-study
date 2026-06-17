""" Implements noisy quantum kernel evaluation using Qiskit Aer with configurable noise models."""
from __future__ import annotations

from typing import List
import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from ..config import QSVMConfig
from ..feature_maps import build_feature_map
from ..noise import build_noise_model


def compute_kernel(X1: np.ndarray, X2: np.ndarray, cfg: QSVMConfig) -> np.ndarray:
    """
    Compute a quantum kernel matrix using the noisy Aer simulator.
    """
    if cfg.shots is None or cfg.shots <= 0:
        raise ValueError("noisy_aer backend requires cfg.shots > 0")

    # Build the feature map for the input dimension
    feature_map = build_feature_map(X2.shape[1], cfg)

    # Build a noise model only if noise is enabled
    noise_model = None
    if cfg.noise is not None and cfg.noise.type not in (None, "none"):
        noise_model = build_noise_model(cfg.noise)

    # Create the simulator with the chosen noise model
    sim = AerSimulator(
        noise_model=noise_model,
        seed_simulator=cfg.seed,
    )

    n1, n2 = X1.shape[0], X2.shape[0]

    # Build one fidelity circuit for each pair of samples
    circuits: List[QuantumCircuit] = []
    for i in range(n1):
        for j in range(n2):
            qc = _fidelity_circuit(feature_map, X1[i], X2[j])
            qc.metadata = {"i": i, "j": j}
            circuits.append(qc)

    # Transpile circuits for execution on Aer
    tcircs = transpile(
        circuits,
        sim,
        optimization_level=1,
        seed_transpiler=cfg.seed,
    )

    # Run all circuits and collect results
    job = sim.run(tcircs, shots=cfg.shots)
    result = job.result()

    # Store the kernel values
    K = np.zeros((n1, n2), dtype=float)
    zeros = "0" * feature_map.num_qubits

    # Estimate each kernel entry from the all-zero outcome frequency
    for idx, qc in enumerate(tcircs):
        md = qc.metadata
        counts = result.get_counts(idx)
        K[md["i"], md["j"]] = counts.get(zeros, 0) / cfg.shots

    # If computing K(X, X), enforce symmetry and set diagonal to 1
    if n1 == n2 and np.allclose(X1, X2):
        K = 0.5 * (K + K.T)
        np.fill_diagonal(K, 1.0)

    return K


def _fidelity_circuit(feature_map, x: np.ndarray, y: np.ndarray) -> QuantumCircuit:
    """Build a circuit to estimate the overlap between two encoded samples."""
    n = feature_map.num_qubits

    # Encode x and y into separate circuits
    ux = feature_map.assign_parameters(x, inplace=False)
    uy = feature_map.assign_parameters(y, inplace=False)

    # Apply U(x) followed by U(y)^(-1), then measure
    qc = QuantumCircuit(n, n)
    qc.compose(ux, inplace=True)
    qc.compose(uy.inverse(), inplace=True)
    qc.measure(range(n), range(n))
    return qc