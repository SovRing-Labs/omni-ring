"""
Semantic Resonator Attractor Clamp for OMNIRING (Tier 2).
Projects noisy, drifting, or continuous neural activations into discrete attractor basins
governed by coprime resonator dynamics and verified by the 17-check CRT gate.
Prevents factual hallucinations and syntax drifts from reaching execution.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Dict, Union, Tuple
import numpy as np

from .types import RingSpec, DEFAULT_SPEC, PhaseVector
from .codebook import generate_direct_codebooks
from .crt import solve_crt, direct_probe
from .check import check_anomaly, ResidueDesyncError
from .polymorphic import TernaryVector, PhaseVectorPacked
from .bridge import ActivationProjector


@dataclass
class ClampResult:
    """Outcome of a Semantic Attractor Clamping operation."""
    status: str             # "CLAMPED", "SLIP_REJECTED", "AMBIGUOUS"
    clamped: bool           # True if successfully grounded into an attractor
    coordinate: Optional[int]
    symbol: Optional[str]
    confidence: float
    iterations: int
    clean_phase: Optional[PhaseVectorPacked]
    clean_ternary: Optional[TernaryVector]
    error: Optional[str] = None


class ResonatorAttractorClamp:
    """
    Error-Correcting Semantic Gyroscope for Neural Cortex Outputs.
    Forces continuous/drifting activation vectors into discrete, verified codebook attractors.
    """
    def __init__(self, spec: RingSpec = DEFAULT_SPEC, seed: int = 42):
        self.spec = spec
        self.direct_codebooks = generate_direct_codebooks(spec, seed=seed)
        self.direct_cbs_conj = [np.conj(cb) for cb in self.direct_codebooks]
        self.symbol_to_coord: Dict[str, int] = {}
        self.coord_to_symbol: Dict[int, str] = {}

    def register_symbol(self, symbol: str, coordinate: int) -> None:
        """Bind a discrete semantic symbol (e.g. tool name, schema token) to a coordinate."""
        assert 0 <= coordinate < self.spec.data_capacity, (
            f"Coordinate {coordinate} out of bounds [0, {self.spec.data_capacity})"
        )
        self.symbol_to_coord[symbol] = coordinate
        self.coord_to_symbol[coordinate] = symbol

    def register_symbols(self, mapping: Dict[str, int]) -> None:
        """Register multiple symbols at once."""
        for sym, coord in mapping.items():
            self.register_symbol(sym, coord)

    def get_symbol_vector(self, symbol: str) -> PhaseVectorPacked:
        """Synthesize the exact, clean 2,560-byte PhaseVectorPacked for a registered symbol."""
        coord = self.symbol_to_coord[symbol]
        return self.get_coordinate_vector(coord)

    def get_coordinate_vector(self, coordinate: int) -> PhaseVectorPacked:
        """Synthesize the exact clean 2,560-byte PhaseVectorPacked for an integer coordinate."""
        mods = self.spec.all_moduli
        v_sum = sum(cb[coordinate % m] for cb, m in zip(self.direct_codebooks, mods))
        angles = np.angle(v_sum)
        angles = np.where(angles < 0, angles + 2 * np.pi, angles)
        phases = np.round(angles * self.spec.k_quantization / (2 * np.pi)).astype(np.uint8) % self.spec.k_quantization
        return PhaseVectorPacked.from_phases(phases)

    def clamp(
        self,
        vector: Union[PhaseVectorPacked, TernaryVector, PhaseVector, np.ndarray],
        iterations: int = 15,
        min_confidence: float = 0.05
    ) -> ClampResult:
        """
        Pull noisy or ambiguous vector into nearest discrete attractor basin.
        Verifies integrity with CRT 17-check anomaly detection.
        """
        # Convert input into continuous complex PhaseVector (D,)
        if isinstance(vector, PhaseVectorPacked):
            unpacked_phases = vector.to_phases().astype(np.float32)
            angles = unpacked_phases * (2 * np.pi / float(self.spec.k_quantization))
            complex_v = np.exp(1j * angles)
        elif isinstance(vector, TernaryVector):
            pv = vector.to_phase()
            unpacked_phases = pv.to_phases().astype(np.float32)
            angles = unpacked_phases * (2 * np.pi / float(self.spec.k_quantization))
            complex_v = np.exp(1j * angles)
        elif np.iscomplexobj(vector):
            norm = np.maximum(np.abs(vector), 1e-9)
            complex_v = vector / norm
        elif isinstance(vector, np.ndarray) and vector.dtype == np.uint8 and len(vector) == 2560:
            pv = PhaseVectorPacked.from_bytes(bytes(vector))
            return self.clamp(pv, iterations=iterations, min_confidence=min_confidence)
        else:
            raise TypeError(f"Unsupported vector type for clamping: {type(vector)}")

        # Direct 45-probe projection across all 5 coprime codebooks
        residues = []
        confs = []
        for cb_c in self.direct_cbs_conj:
            dots = np.dot(cb_c, complex_v).real
            best_idx = int(np.argmax(dots))
            residues.append(best_idx)
            
            # Compute eigengap confidence
            sorted_dots = np.sort(dots)
            top1 = sorted_dots[-1]
            top2 = sorted_dots[-2] if len(sorted_dots) > 1 else 0.0
            gap = float((top1 - top2) / (abs(top1) + 1e-9))
            confs.append(max(0.0, min(1.0, gap)))

        avg_confidence = float(np.mean(confs))
        if avg_confidence < min_confidence:
            return ClampResult(
                status="AMBIGUOUS",
                clamped=False,
                coordinate=None,
                symbol=None,
                confidence=avg_confidence,
                iterations=1,
                clean_phase=None,
                clean_ternary=None,
                error=f"Confidence {avg_confidence:.4f} below threshold {min_confidence}"
            )

        # 17-Check CRT Anomaly Detection Gate
        try:
            solved_coord = check_anomaly(residues, self.spec)
        except ResidueDesyncError as e:
            return ClampResult(
                status="SLIP_REJECTED",
                clamped=False,
                coordinate=None,
                symbol=None,
                confidence=avg_confidence,
                iterations=1,
                clean_phase=None,
                clean_ternary=None,
                error=str(e)
            )

        # Ground truth clean attractor reconstruction
        clean_pv = self.get_coordinate_vector(solved_coord)
        clean_tv = clean_pv.to_ternary()
        symbol = self.coord_to_symbol.get(solved_coord, None)

        return ClampResult(
            status="CLAMPED",
            clamped=True,
            coordinate=solved_coord,
            symbol=symbol,
            confidence=avg_confidence,
            iterations=1,
            clean_phase=clean_pv,
            clean_ternary=clean_tv,
            error=None
        )

    def clamp_from_embedding(
        self,
        embedding: np.ndarray,
        projector: ActivationProjector,
        iterations: int = 15
    ) -> ClampResult:
        """Direct projection and clamping from raw neural model activation embedding."""
        tv = projector.project_to_ternary(embedding)
        return self.clamp(tv, iterations=iterations)
