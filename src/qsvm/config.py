""" Defines configuration dataclasses and types for QSVM experiments, including kernel mode and noise settings."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional, Literal


KernelMode = Literal["statevector", "noisy_aer"]
NoiseType = Literal[
    "none",
    "depolarizing",
    "amplitude_damping",
    "phase_damping",
    "combined",
]


@dataclass(frozen=True)
class NoiseConfig:
    """
    Configuration for synthetic noise applied during QSVM kernel evaluation.

    Attributes:
        type: Noise model type. "none" disables device noise.
        strength: Generic noise strength parameter for single-channel models.
        params: Optional channel-specific parameters.
        apply_to: Scope of noise application across gates.
    """
    type: NoiseType = "none"
    strength: float = 0.0
    params: Dict[str, Any] = field(default_factory=dict)
    apply_to: Literal["all", "single_qubit", "two_qubit"] = "all"

    def is_enabled(self) -> bool:
        """Return whether a non-trivial noise model is enabled."""
        return self.type != "none" and (
            self.strength > 0.0
            or bool(self.params)
            or self.type == "combined"
        )
    
@dataclass(frozen=True)
class QSVMConfig:
    """
    Configuration for a QSVM experiment.

    Attributes:
        feature_map: Quantum feature map type.
        reps: Number of repetitions in the feature map circuit.
        entanglement: Entanglement pattern used in the feature map.
        mode: Kernel evaluation backend.
        shots: Number of measurement shots for shot-based simulation.
        seed: Random seed for simulator execution.
        scale_to_pi: Whether to scale features into [-pi, pi] before encoding.
        noise: Noise configuration for noisy simulation.
        C: SVM regularisation parameter.
    """
    feature_map: Literal["zz", "pauli"] = "zz"
    reps: int = 2
    entanglement: Literal["full", "linear", "circular"] = "full"
    mode: KernelMode = "statevector"
    shots: Optional[int] = None
    seed: int = 42
    scale_to_pi: bool = True
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    C: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Return the configuration as a dictionary for serialisation."""
        return asdict(self)

    def validate(self) -> None:
        """ Validate the consistency of the configuration."""
        if self.reps < 1:
            raise ValueError("QSVMConfig.reps must be >= 1")

        if self.mode == "statevector":
            if self.shots is not None:
                raise ValueError("shots should be None for mode='statevector'")
            if self.noise.type != "none" and self.noise.is_enabled():
                raise ValueError("Noiseless baseline should have noise.type='none' (or strength=0).")

        if self.mode == "noisy_aer":
            if self.shots is None or self.shots <= 0:
                raise ValueError(f"shots must be a positive int for mode='{self.mode}'")
            
