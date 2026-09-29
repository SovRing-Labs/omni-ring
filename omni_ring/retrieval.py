"""
Memory-Augmented Gated Retrieval Engine for OMNIRING.
Integrates ternary associative routing, clamped resonator unbinding, and 17-check ring verification.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any, Union
import numpy as np

from .types import RingSpec, DEFAULT_SPEC, ResonatorResult
from .codebook import generate_rotations, generate_codebooks, generate_direct_codebooks
from .chain import encode, bind, unbind
from .resonator import decode_soft
from .crt import solve_crt, direct_probe
from .check import check_anomaly, ResidueDesyncError
from .polymorphic import TernaryVector, PhaseVectorPacked, PolymorphicSlot, PolymorphicBus
from .avx2 import is_avx2_available, batch_similarity_avx2, ternary_gemv_c


@dataclass
class GatedRecallResult:
    """Outcome of a memory retrieval operation with verified status."""
    status: str  # "VERIFIED", "ABSTAIN", "REJECTED"
    coordinate: Optional[int]
    residues: Optional[List[int]]
    confidence: float
    matched_slot: int = -1
    error_message: Optional[str] = None


class GatedMemoryEngine:
    """
    Cognitive memory engine combining:
    1. Fast ternary reflex routing (10,240-D)
    2. AVX2 SIMD phase similarity (5,120-D)
    3. Direct 45-probe + CRT instant coordinate solving
    4. Soft-settling resonator with factor clamping for crowded superpositions
    5. 17-ring CRT integrity gate (100% single-ring slip detection, 54,600 exhaustive cases).
       This is an integrity check, not an accuracy guarantee: it proves a decoded
       address matches its codeword, not that the answer is right. Retrieval
       accuracy on superposition is measured separately, and abstention is
       available when no verified attractor exists.
       See reviews/2026-09-28 — OMNIRING-PEER-REVIEW-RETRIEVAL-AND-CONSOLIDATION.md §6.
    """
    def __init__(self, spec: RingSpec = DEFAULT_SPEC, seed: int = 42):
        self.spec = spec
        # Direct probe codebooks (K-quantized, 45 total vectors)
        self.direct_codebooks = generate_direct_codebooks(spec, seed=seed)
        self.direct_cbs_conj = [np.conj(cb) for cb in self.direct_codebooks]
        
        # Continuous rotation vectors for timing chain
        self.rotations = generate_rotations(spec, seed=seed)
        self.chain_codebooks = generate_codebooks(self.rotations, spec)

    def encode_direct(self, x: int) -> PhaseVectorPacked:
        """
        Encode an integer state x in [0, data_capacity) as an additive superposition
        of codevectors across all 5 rings (including 17-check ring).
        Quantized to 4-bit nibbles (2,560 bytes).
        """
        if x < 0 or x >= self.spec.data_capacity:
            raise ValueError(f"State {x} out of range [0, {self.spec.data_capacity})")

        mods = self.spec.all_moduli
        v_sum = sum(cb[x % m] for cb, m in zip(self.direct_codebooks, mods))
        
        angles = np.angle(v_sum)
        angles = np.where(angles < 0, angles + 2 * np.pi, angles)
        phases = np.round(angles * self.spec.k_quantization / (2 * np.pi)).astype(np.uint8) % self.spec.k_quantization
        return PhaseVectorPacked.from_phases(phases)

    def encode_bound(self, x: int) -> PhaseVectorPacked:
        """
        Encode an integer state x into a multiplicative timing chain vector across all rings.
        Quantized to 4-bit nibbles.
        """
        if x < 0 or x >= self.spec.data_capacity:
            raise ValueError(f"State {x} out of range [0, {self.spec.data_capacity})")

        v_prod = encode(x, self.rotations)
        angles = np.angle(v_prod)
        angles = np.where(angles < 0, angles + 2 * np.pi, angles)
        phases = np.round(angles * self.spec.k_quantization / (2 * np.pi)).astype(np.uint8) % self.spec.k_quantization
        return PhaseVectorPacked.from_phases(phases)

    def decode_direct(self, pv: PhaseVectorPacked) -> GatedRecallResult:
        """
        Perform instant 45-probe projection + CRT solve + 17-check verification.
        Latency: < 0.35 ms.
        """
        phases = pv.to_phases()
        v_complex = np.exp(1j * phases * (2.0 * np.pi / self.spec.k_quantization))
        
        residues, raw_x = direct_probe(
            v_complex,
            self.direct_codebooks,
            self.spec.all_moduli,
            codebooks_conj=self.direct_cbs_conj
        )

        try:
            verified_x = check_anomaly(residues, self.spec)
            return GatedRecallResult(
                status="VERIFIED",
                coordinate=verified_x,
                residues=residues,
                confidence=1.0
            )
        except ResidueDesyncError as e:
            return GatedRecallResult(
                status="REJECTED",
                coordinate=None,
                residues=residues,
                confidence=0.0,
                error_message=str(e)
            )

    def decode_bound_clamped(
        self,
        v_complex: np.ndarray,
        clamp_cue: int,
        clamp_idx: int = 3,
        iterations: int = 15,
        check: bool = False
    ) -> GatedRecallResult:
        """
        Decode a crowded bound superposition using the 13-ring clamped soft resonator.

        check=False (legacy, 4 data rings): status is "CONVERGED" or "ABSTAIN". Nothing verifies the answer;
            a converged settle can be a legal-looking wrong mark (RING1-CHECK, RING1-SOFTSWEEP).
        check=True (slot built with all 5 rings incl. 17, e.g. encode(x, self.rotations)): decodes all 5 rings and runs the
            17-ring CRT gate -> "VERIFIED" (converged and legal), "REJECTED" (illegal), "ABSTAIN" (not converged).
        A cue is required: unguided decoding of these rotation codebooks is seed-dependent (RING3 harness disclosure,
        2026-09-29), so this method never runs unguided.
        """
        n = 5 if check else 4
        res = decode_soft(
            v_complex,
            self.chain_codebooks[:n],
            iterations=iterations,
            clamp=clamp_cue,
            clamp_idx=clamp_idx
        )
        avg_conf = float(np.mean(res.confidences))
        if not check:
            solved_x = solve_crt(res.residues, self.spec.moduli)
            return GatedRecallResult(
                status="CONVERGED" if res.converged else "ABSTAIN",
                coordinate=solved_x,
                residues=res.residues,
                confidence=avg_conf
            )
        if not res.converged:
            return GatedRecallResult(status="ABSTAIN", coordinate=None, residues=res.residues, confidence=avg_conf)
        try:
            x = check_anomaly(res.residues, self.spec)
        except ResidueDesyncError as e:
            return GatedRecallResult(status="REJECTED", coordinate=None, residues=res.residues, confidence=0.0,
                                     error_message=str(e))
        return GatedRecallResult(status="VERIFIED", coordinate=x, residues=res.residues, confidence=avg_conf)

    def associative_scan(
        self,
        query: TernaryVector,
        candidate_slots: List[PolymorphicSlot],
        top_k: int = 3
    ) -> List[Tuple[int, int]]:
        """
        Perform ultra-fast ternary Hamming scanning over candidate slots.
        Uses native AVX2 ternary_gemv_c when available.
        Returns sorted list of (slot_index, dot_product_score).
        """
        if not candidate_slots:
            return []

        if is_avx2_available():
            matrix_sign = np.ascontiguousarray([np.frombuffer(slot.data[:1280], dtype=np.uint64) for slot in candidate_slots])
            matrix_active = np.ascontiguousarray([np.frombuffer(slot.data[1280:2560], dtype=np.uint64) for slot in candidate_slots])
            scores = ternary_gemv_c(matrix_sign, matrix_active, query.sign, query.active)
            top_indices = np.argsort(scores)[::-1][:top_k]
            return [(int(idx), int(scores[idx])) for idx in top_indices]

        scores = []
        for idx, slot in enumerate(candidate_slots):
            tv = slot.as_ternary()
            score = query.dot(tv)
            scores.append((idx, score))

        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[:top_k]

    def avx2_phase_scan(
        self,
        query: PhaseVectorPacked,
        candidate_matrix: np.ndarray,
        top_k: int = 3
    ) -> List[Tuple[int, int]]:
        """
        Perform sub-microsecond AVX2 similarity scan across packed phase vectors.
        candidate_matrix: shape (N, 2560) uint8
        """
        if not is_avx2_available():
            raise RuntimeError("AVX2 native library is not available.")

        scores = batch_similarity_avx2(query.data, candidate_matrix)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), int(scores[idx])) for idx in top_indices]
