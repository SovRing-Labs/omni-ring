"""
Type definitions and configurations for OMNIRING.
"""
from dataclasses import dataclass
from typing import Tuple, List, NamedTuple
import numpy as np

# Type aliases
PhaseVector = np.ndarray  # Complex128 array of shape (D,) or (N, D)
ResidueTuple = Tuple[int, ...]  # e.g., (r3, r5, r7, r13)

@dataclass(frozen=True)
class RingSpec:
    """Specification of the coprime ring system."""
    moduli: Tuple[int, ...] = (3, 5, 7, 13)
    check_modulus: int = 17
    dimension: int = 5120
    k_quantization: int = 16

    @property
    def data_capacity(self) -> int:
        """Maximum states without ambiguity for data rings: 3 * 5 * 7 * 13 = 1,365."""
        return int(np.prod(self.moduli))

    @property
    def all_moduli(self) -> Tuple[int, ...]:
        """All moduli including the check ring: (3, 5, 7, 13, 17)."""
        return self.moduli + (self.check_modulus,)

    @property
    def full_capacity(self) -> int:
        """Total capacity with check ring: 3 * 5 * 7 * 13 * 17 = 23,205."""
        return int(np.prod(self.all_moduli))

    @property
    def total_probes(self) -> int:
        """Total probe count across all rings: 3 + 5 + 7 + 13 + 17 = 45."""
        return sum(self.all_moduli)


DEFAULT_SPEC = RingSpec()


class ResonatorResult(NamedTuple):
    """Result of resonator decoding."""
    residues: List[int]
    confidences: List[float]
    converged: bool
    iterations: int
