"""
Chinese Remainder Theorem (CRT) and Direct Coordinate Probing for OMNIRING.
"""
from math import prod
from typing import List, Tuple, Optional
import numpy as np
from .types import PhaseVector


def modular_inverse(a: int, m: int) -> int:
    """Compute the modular multiplicative inverse of a modulo m."""
    return pow(a, -1, m)


def solve_crt(residues: List[int], moduli: Tuple[int, ...]) -> int:
    """
    Solve system of congruences using the Chinese Remainder Theorem:
    X ≡ r_i (mod m_i) for each i.
    
    Returns:
        Unique integer X in [0, prod(moduli) - 1].
    """
    if len(residues) != len(moduli):
        raise ValueError("Number of residues must match number of moduli")

    M = prod(moduli)
    total = 0
    for r, m in zip(residues, moduli):
        m_i = M // m
        y_i = modular_inverse(m_i, m)
        total += r * m_i * y_i

    return total % M


def direct_probe(
    v: PhaseVector,
    codebooks: List[np.ndarray],
    moduli: Tuple[int, ...],
    codebooks_conj: Optional[List[np.ndarray]] = None
) -> Tuple[List[int], int]:
    """
    Project directly onto each codebook using dot products (45 total probes for 3,5,7,13,17),
    identify the best matching residue for each ring, and solve CRT.

    Returns:
        (residues, X) where X is the reconstructed global coordinate.
    """
    if codebooks_conj is None:
        codebooks_conj = [np.conj(cb) for cb in codebooks]

    residues = []
    for cb_c in codebooks_conj:
        dots = np.dot(cb_c, v).real
        residues.append(int(np.argmax(dots)))
    
    x = solve_crt(residues, moduli)
    return residues, x
