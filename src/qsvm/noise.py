""" Constructs noise models for quantum simulation, including depolarising, amplitude damping, phase
damping, and combined noise channels."""
from __future__ import annotations

from qiskit_aer.noise import NoiseModel
from qiskit_aer.noise.errors import (
    depolarizing_error,
    amplitude_damping_error,
    phase_damping_error,
)

from .config import NoiseConfig


_ONE_Q_GATES = ["rz", "sx", "x", "rx", "ry", "h"]
_TWO_Q_GATES = ["cx", "ecr"]


def build_noise_model(noise: NoiseConfig) -> NoiseModel | None:
    """
    Build a Qiskit Aer noise model from the given noise configuration.

    Args:
        noise: Noise configuration specifying the channel type, strength,
            and application scope.

    Returns:
        A configured noise model, or None if noise is disabled.
    """
    if not noise.is_enabled():
        return None

    nm = NoiseModel()

    if noise.type == "depolarizing":
        err1 = depolarizing_error(noise.strength, 1)
        err2 = depolarizing_error(noise.strength, 2)

        if noise.apply_to in ("all", "single_qubit"):
            nm.add_all_qubit_quantum_error(err1, _ONE_Q_GATES)
        if noise.apply_to in ("all", "two_qubit"):
            nm.add_all_qubit_quantum_error(err2, _TWO_Q_GATES)

    elif noise.type == "amplitude_damping":
        err1 = amplitude_damping_error(noise.strength)
        err2 = err1.tensor(err1)

        if noise.apply_to in ("all", "single_qubit"):
            nm.add_all_qubit_quantum_error(err1, _ONE_Q_GATES)
        if noise.apply_to in ("all", "two_qubit"):
            nm.add_all_qubit_quantum_error(err2, _TWO_Q_GATES)

    elif noise.type == "phase_damping":
        err1 = phase_damping_error(noise.strength)
        err2 = err1.tensor(err1)

        if noise.apply_to in ("all", "single_qubit"):
            nm.add_all_qubit_quantum_error(err1, _ONE_Q_GATES)
        if noise.apply_to in ("all", "two_qubit"):
            nm.add_all_qubit_quantum_error(err2, _TWO_Q_GATES)

    elif noise.type == "combined":
        p = float(noise.params.get("depol", 0.0))
        g = float(noise.params.get("amp", 0.0))
        lam = float(noise.params.get("phase", 0.0))

        err1 = depolarizing_error(p, 1)
        if g > 0:
            err1 = err1.compose(amplitude_damping_error(g))
        if lam > 0:
            err1 = err1.compose(phase_damping_error(lam))

        err2 = depolarizing_error(p, 2)

        nm.add_all_qubit_quantum_error(err1, _ONE_Q_GATES)
        nm.add_all_qubit_quantum_error(err2, _TWO_Q_GATES)

    else:
        raise ValueError(f"Noise kind '{noise.type}' not implemented")

    return nm