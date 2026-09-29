"""
Continuous Soft Resonator for OMNIRING.
Implements multi-ring soft-settling power iteration.
"""
from typing import List, Optional, Tuple
import numpy as np
from .types import PhaseVector, ResonatorResult


def decode_soft(
    v: PhaseVector,
    codebooks: List[np.ndarray],
    iterations: int = 15,
    clamp: Optional[int] = None,
    clamp_idx: Optional[int] = None,
    tolerance: float = 1e-5
) -> ResonatorResult:
    """
    Perform continuous soft-settling resonator decoding on composite/superposed vector v.

    Args:
        v: Input complex phase vector of shape (D,).
        codebooks: List of codebook matrices, where codebooks[i] has shape (m_i, D).
        iterations: Number of power-iteration settling loops.
        clamp: Optional known index to lock for a specific ring.
        clamp_idx: The ring index to clamp (defaults to last ring if clamp is provided).
        tolerance: Convergence tolerance for early stopping.

    Returns:
        ResonatorResult with residues, confidences, convergence status, and iterations taken.
    """
    n_rings = len(codebooks)
    D = v.shape[-1]
    
    if clamp is not None and clamp_idx is None:
        clamp_idx = n_rings - 1

    # Pre-conjugate codebooks for efficient projection
    cb_conj = [np.conj(cb) for cb in codebooks]

    # Initialize estimates with average of each codebook
    est = [cb.mean(axis=0) for cb in codebooks]
    est_conj = [np.conj(e) for e in est]

    if clamp is not None:
        est[clamp_idx] = codebooks[clamp_idx][clamp]
        est_conj[clamp_idx] = np.conj(est[clamp_idx])

    converged = False
    it_count = 0

    for it in range(iterations):
        it_count = it + 1
        max_delta = 0.0

        for i in range(n_rings):
            if clamp is not None and i == clamp_idx:
                continue

            # Unbind all other ring estimates: u_i = V ⊙ ∏_{j ≠ i} conj(e_j)
            u = v.copy()
            for j in range(n_rings):
                if j != i:
                    u *= est_conj[j]

            # Project onto codebook: a_i = (1/D) * Re(CB_i^* @ u)
            # Use np.dot for fast 1D-2D multiplication
            a = np.dot(cb_conj[i], u).real / D

            # Soft activation: e_i = ReLU(a_i) @ CB_i
            relu_a = np.maximum(a, 0)
            e = np.dot(relu_a, codebooks[i])

            # Unit-modulus normalization
            norm = np.maximum(np.abs(e), 1e-9)
            e_norm = e / norm

            delta = np.max(np.abs(e_norm - est[i]))
            if delta > max_delta:
                max_delta = delta

            est[i] = e_norm
            est_conj[i] = np.conj(e_norm)

        if max_delta < tolerance:
            converged = True
            break

    # Extract final residues and confidence (eigengap)
    residues: List[int] = []
    confidences: List[float] = []

    for i in range(n_rings):
        dots = np.dot(cb_conj[i], est[i]).real
        best_idx = int(np.argmax(dots))
        residues.append(best_idx)

        # Compute confidence: (top1 - top2) / top1
        sorted_dots = np.sort(dots)
        top1 = sorted_dots[-1]
        top2 = sorted_dots[-2] if len(sorted_dots) > 1 else 0.0
        conf = float((top1 - top2) / top1) if top1 > 1e-9 else 0.0
        confidences.append(max(0.0, min(1.0, conf)))

    return ResonatorResult(
        residues=residues,
        confidences=confidences,
        converged=converged,
        iterations=it_count
    )
