"""
Codebook and Discretization for OMNIRING.
"""
from typing import List, Optional
import numpy as np
from .types import RingSpec, DEFAULT_SPEC, PhaseVector


def generate_rotations(spec: RingSpec = DEFAULT_SPEC, seed: Optional[int] = 42) -> List[np.ndarray]:
    """
    Generate the base rotation vectors for each ring.
    For each modulus m, r_m has entries from {0, ..., m-1} * (2*pi / m).
    
    Returns:
        List of 1D arrays of shape (D,), one per modulus in spec.all_moduli.
    """
    rng = np.random.default_rng(seed)
    rotations = [
        rng.integers(0, m, spec.dimension) * (2.0 * np.pi / m)
        for m in spec.all_moduli
    ]
    return rotations


def generate_codebooks(rotations: List[np.ndarray], spec: RingSpec = DEFAULT_SPEC) -> List[np.ndarray]:
    """
    Generate the full codebook for each ring from its rotation vector.
    For ring m and coordinate x in {0, ..., m-1}:
        CB_m[x] = exp(1j * x * r_m)
        
    Returns:
        List of 2D complex arrays of shape (m, D).
    """
    all_mods = spec.all_moduli[:len(rotations)]
    codebooks = [
        np.exp(1j * np.outer(np.arange(m), r))
        for m, r in zip(all_mods, rotations)
    ]
    return codebooks


def generate_quantized_codebook(
    size: int,
    dimension: int = DEFAULT_SPEC.dimension,
    k_quantization: int = DEFAULT_SPEC.k_quantization,
    seed: Optional[int] = 42
) -> PhaseVector:
    """
    Generate an array of K-quantized random complex phase vectors on the unit circle.
    
    Shape: (size, dimension)
    """
    rng = np.random.default_rng(seed)
    phases = rng.integers(0, k_quantization, (size, dimension)) * (2.0 * np.pi / k_quantization)
    return np.exp(1j * phases)


def generate_direct_codebooks(spec: RingSpec = DEFAULT_SPEC, seed: Optional[int] = 42) -> List[np.ndarray]:
    """
    Generate independent K-quantized random codebooks for direct probing.
    
    Returns:
        List of 2D complex arrays of shape (m, D) for each m in spec.all_moduli.
    """
    rng = np.random.default_rng(seed)
    return [
        np.exp(1j * rng.integers(0, spec.k_quantization, (m, spec.dimension)) * (2.0 * np.pi / spec.k_quantization))
        for m in spec.all_moduli
    ]


def verify_unguided(spec: RingSpec = DEFAULT_SPEC, seed: Optional[int] = 42, samples: int = 15,
                    iterations: int = 30) -> float:
    """Fraction of clean single marks that decode_soft recovers with NO cue for this codebook seed.

    Unguided factorisation of these rotation codebooks is marginal at D = 5,120 and depends on the draw
    (2026-09-29: seeds 0/1/2 -> 15/15, seeds 3/21/42 -> 1/15 over all 5 rings). Engine paths always pass a cue;
    use this before relying on any unguided decode. Cued decoding was reliable at every seed tested.
    """
    from .chain import encode
    from .resonator import decode_soft
    rot = generate_rotations(spec, seed=seed)
    cbs = [np.exp(1j * np.outer(np.arange(m), r)) for m, r in zip(spec.all_moduli, rot)]
    n = spec.data_capacity
    xs = [int(round(i * (n - 1) / max(1, samples - 1))) for i in range(samples)]
    ok = sum(decode_soft(encode(x, rot), cbs, iterations=iterations).residues == [x % m for m in spec.all_moduli]
             for x in xs)
    return ok / samples
