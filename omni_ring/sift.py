"""
Exhaustive lattice sift, prune-then-join star reads, and an automatic readout (TORUS / EXH / STAR-0, 2026-09-29).

Measured (``_Reviews/TORUS/TORUS-RESULTS.md``): comparing a slot with every legal codeword reads 24-192 marks per slot with 0
fakes and 382.6/384 at 384; cued resonance tops out near 12. The slot must keep its magnitude (int8 real/imag here): a
phase-only slot stops separating members from non-members near 192 marks.

Rule: enumerate what the budget allows, resonate the rest.
"""
import ctypes
import os
from typing import Callable, List, Optional, Sequence, Set, Tuple

import numpy as np
import numpy.ctypeslib as npct

from .chain import encode
from .types import DEFAULT_SPEC, RingSpec

K_PHASE = 16
THRESHOLD = 0.5                       # a stored mark scores ~1.0, crosstalk sd ~ sqrt(M / 2D)
LATTICE_BUDGET_BYTES = 512 * 2 ** 20  # J 2026-09-29: "if it's under 500 MB, store it in RAM"

_C8 = np.round(127 * np.cos(2 * np.pi * np.arange(K_PHASE) / K_PHASE)).astype(np.int32)
_S8 = np.round(127 * np.sin(2 * np.pi * np.arange(K_PHASE) / K_PHASE)).astype(np.int32)

_lib = None
try:
    _lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "libomniring_avx2.so"))
    _lib.omniring_sift_cplx.argtypes = [
        npct.ndpointer(dtype=np.uint8, ndim=2, flags="C_CONTIGUOUS"), ctypes.c_size_t, ctypes.c_size_t,
        npct.ndpointer(dtype=np.int8, ndim=1, flags="C_CONTIGUOUS"),
        npct.ndpointer(dtype=np.int8, ndim=1, flags="C_CONTIGUOUS"),
        npct.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),
    ]
    _lib.omniring_sift_cplx.restype = None
except (OSError, AttributeError):
    _lib = None


def has_kernel() -> bool:
    """True when the AVX2 sift kernel is built (``./build_kernel.sh``)."""
    return _lib is not None


def phase_indices(vectors: np.ndarray) -> np.ndarray:
    """Unit-modulus codewords -> uint8 phase index per dimension (0..15), the lattice storage format."""
    return (np.round(np.angle(vectors) / (2 * np.pi / K_PHASE)) % K_PHASE).astype(np.uint8)


def ring_lattice(rotations: List[np.ndarray], spec: RingSpec = DEFAULT_SPEC, path: Optional[str] = None) -> np.ndarray:
    """Every legal coordinate x in [0, N) encoded as a phase-index row (N x D uint8). Row x is coordinate x.
    With `path`, the lattice is saved once and memory-mapped afterwards (the one-off build is not repeated)."""
    if path and os.path.exists(path):
        return np.load(path, mmap_mode="r")
    n = int(np.prod(spec.moduli))
    lat = np.empty((n, spec.dimension), dtype=np.uint8)
    for x0 in range(0, n, 512):
        lat[x0:x0 + 512] = phase_indices(np.stack([encode(x, rotations) for x in range(x0, min(n, x0 + 512))]))
    if path:
        np.save(path, lat)
    return lat


def quantize_slot(v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Slot (complex, WITH magnitude) -> int8 real, int8 imag, scale. Zero slot -> scale 0."""
    m = float(max(np.abs(v.real).max(initial=0.0), np.abs(v.imag).max(initial=0.0)))
    if m == 0.0:
        z = np.zeros(v.shape[-1], dtype=np.int8)
        return z, z.copy(), 0.0
    return (np.round(v.real / m * 127).astype(np.int8), np.round(v.imag / m * 127).astype(np.int8), m / 127.0)


def scores(v: np.ndarray, lattice: np.ndarray, backend: str = "auto") -> np.ndarray:
    """Re(conj(w_n) v) / D for every lattice row n (a stored mark scores ~1). backend: auto | c | numpy."""
    n, d = lattice.shape
    qr, qi, scale = quantize_slot(v)
    if scale == 0.0:
        return np.zeros(n, dtype=np.float64)
    use_c = backend == "c" or (backend == "auto" and _lib is not None and d % 32 == 0)
    if use_c:
        if _lib is None:
            raise RuntimeError("AVX2 kernel not built; run ./build_kernel.sh")
        out = np.empty(n, dtype=np.int32)
        _lib.omniring_sift_cplx(np.ascontiguousarray(lattice), n, d, qr, qi, out)
        raw = out.astype(np.float64)
    else:
        raw = np.empty(n, dtype=np.float64)
        qr32, qi32 = qr.astype(np.int32), qi.astype(np.int32)
        for s0 in range(0, n, 1024):
            blk = lattice[s0:s0 + 1024]
            raw[s0:s0 + 1024] = (_C8[blk] * qr32).sum(1) + (_S8[blk] * qi32).sum(1)
    return raw * scale / (127.0 * d)


def sift(v: np.ndarray, lattice: np.ndarray, threshold: float = THRESHOLD, backend: str = "auto") -> List[int]:
    """Indices of lattice rows scoring >= threshold, best first. An empty slot returns []."""
    s = scores(v, lattice, backend)
    hit = np.nonzero(s >= threshold)[0]
    return [int(i) for i in hit[np.argsort(-s[hit])]]


def prune_join(centre: np.ndarray, points: Sequence[np.ndarray], facet_books: Sequence[np.ndarray],
               joint: Callable[[np.ndarray], np.ndarray], known: Optional[dict] = None,
               threshold: float = THRESHOLD) -> Tuple[Set[tuple], int]:
    """Star read: each point sifts its own facet codebook; the centre tests only the product of survivors.
    `joint(combos)` maps an (m x F) int array of facet values to (m x D) bound codewords (including any checksum ring).
    Returns (records found, codewords tested). Pays when points are sparse (STAR-0: 83x fewer at 6 records)."""
    d = centre.shape[-1]
    cands, tested = [], 0
    for i, (p, book) in enumerate(zip(points, facet_books)):
        if known and i in known:
            cands.append([known[i]])
            continue
        tested += len(book)
        s = (book.conj() @ p).real / d
        cands.append(list(np.nonzero(s >= threshold)[0]))
    if not all(cands):
        return set(), tested
    combos = np.array(np.meshgrid(*cands, indexing="ij")).reshape(len(cands), -1).T
    found = set()
    for chunk in np.array_split(combos, max(1, len(combos) // 2048)):
        s = (joint(chunk).conj() @ centre).real / d
        found |= {tuple(int(x) for x in chunk[k]) for k in np.nonzero(s >= threshold)[0]}
    return found, tested + len(combos)


def read(v: np.ndarray, rotations: List[np.ndarray], spec: RingSpec = DEFAULT_SPEC,
         lattice: Optional[np.ndarray] = None, budget_bytes: int = LATTICE_BUDGET_BYTES) -> Tuple[List[int], str]:
    """Automatic readout of a ring slot. Returns (coordinates, mode).
    sift  — the legal lattice fits the budget (built on demand if not given): exhaustive, exact, hundreds of marks.
    peel  — the lattice is too large to enumerate: cued sweeps + explaining-away (``peel.read_all``)."""
    n = int(np.prod(spec.moduli))
    if lattice is not None or n * spec.dimension <= budget_bytes:
        lat = lattice if lattice is not None else ring_lattice(rotations, spec)
        return sift(v, lat), "sift"
    from .peel import read_all
    return read_all(v, rotations, spec), "peel"
