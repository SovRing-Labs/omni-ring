"""
Regular solids as codebooks on the phase torus (K = 16 positions per dimension).

A codebook is a matrix: one row per atom. Its Gram matrix (row · conj(row) / d, real part) is the solid's shape:
  simplex         off-diagonal exactly -1/(N-1)        (N = 4: the tetrahedron)
  cross_polytope  antipodal pairs -1, all others 0      (N = 6: the octahedron)
  hypercube       rows from {0, 8} phases = {+1, -1}    (vertices of the d-cube)
  polygon         circulant: atom k = k steps around one orbit (the coprime rings are polygons)
In dimension >= 5 these three regular families (simplex, hypercube, cross-polytope) are the only regular polytopes.

Construction: harmonic frames. Rows of the n x n DFT matrix have unit-modulus entries; drop the all-ones column and the
n rows form a simplex in n-1 dims; keep it and they are orthogonal. When n divides K = 16 every phase is an exact
16-position value. A block of b dims is tiled to d = b * L dims by adding one common random phase per copy
(v[i] + u[l]): differences cancel, so the Gram matrix is preserved exactly.
"""
from __future__ import annotations

import numpy as np

K = 16


def _harmonic(n: int, drop_ones: bool) -> np.ndarray:
    if K % n:
        raise ValueError(f"exact phases need n | {K}; got n = {n}")
    j = np.arange(n)[:, None]
    cols = np.arange(1 if drop_ones else 0, n)[None, :]
    return (j * cols * (K // n)) % K                      # (n, n-1) or (n, n) phases


def _tile(block: np.ndarray, d: int, seed: int) -> np.ndarray:
    b = block.shape[1]
    L = d // b
    u = np.random.default_rng(seed).integers(0, K, L)
    return ((block[:, :, None] + u[None, None, :]) % K).reshape(block.shape[0], b * L).astype(np.uint8)


def simplex(n: int, d: int, seed: int = 0) -> np.ndarray:
    """n atoms, pairwise phase-cosine exactly -1/(n-1). Width (n-1) * (d // (n-1))."""
    return _tile(_harmonic(n, drop_ones=True), d, seed)


def orthogonal(n: int, d: int, seed: int = 0) -> np.ndarray:
    """n mutually orthogonal atoms (pairwise cosine 0). Width n * (d // n)."""
    return _tile(_harmonic(n, drop_ones=False), d, seed)


def cross_polytope(n: int, d: int, seed: int = 0) -> np.ndarray:
    """2n atoms: n orthogonal axes and their antipodes (phase + 8)."""
    o = orthogonal(n, d, seed)
    return np.concatenate([o, (o + K // 2) % K]).astype(np.uint8)


def hypercube(n: int, d: int, seed: int = 0) -> np.ndarray:
    """n random vertices of the d-cube: every phase is 0 (+1) or 8 (-1)."""
    return (np.random.default_rng(seed).integers(0, 2, (n, d)) * (K // 2)).astype(np.uint8)


def polygon(n: int, d: int, seed: int = 0) -> np.ndarray:
    """n atoms on one orbit: atom k = k * w (mod 16); exact regular n-gon when n | 16 (same shape as a ring codebook)."""
    if K % n:
        raise ValueError(f"exact phases need n | {K}; got n = {n}")
    w = np.random.default_rng(seed).integers(0, n, d) * (K // n)
    return ((np.arange(n)[:, None] * w[None, :]) % K).astype(np.uint8)


def gram(codebook: np.ndarray) -> np.ndarray:
    """Real phase-cosine Gram matrix: mean over dims of cos(2 pi (a - b) / K)."""
    z = np.exp(2j * np.pi * codebook.astype(np.float64) / K)
    return (z @ z.conj().T).real / codebook.shape[1]


def min_separation(codebook: np.ndarray) -> float:
    """Smallest pairwise distance on the unit sphere: sqrt(2 - 2 cos) for the closest pair (larger = safer)."""
    g = gram(codebook)
    np.fill_diagonal(g, -np.inf)
    return float(np.sqrt(max(0.0, 2 - 2 * g.max())))
