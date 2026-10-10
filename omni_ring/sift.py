"""
Exhaustive lattice sift, prune-then-join star reads, and an automatic readout (TORUS / EXH / STAR-0, 2026-09-29).

Measured (``TECHNICAL-DISCLOSURE.md §12``): comparing a slot with every legal codeword reads 24-192 marks per slot with 0
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

# AVX-512 kernels (OMNIKIT 2026-10-08, ``./build_kernel_avx512.sh``): VNNI sift (bit-identical to the AVX2 scores) + VPOPCNTDQ
# Hamming / ternary sift. Loaded ONLY when /proc/cpuinfo shows every flag below (the .so would SIGILL elsewhere);
# OMNI_RING_NO_AVX512=1 forces it off.
AVX512_FLAGS = ("avx512f", "avx512bw", "avx512vl", "avx512_vpopcntdq", "avx512_vnni")


def cpu_has_avx512(cpuinfo: str = "/proc/cpuinfo") -> bool:
    """True when the CPU reports every flag the AVX-512 kernels use."""
    try:
        with open(cpuinfo) as f:
            for line in f:
                if line.startswith("flags"):
                    have = set(line.split(":", 1)[1].split())
                    return all(fl in have for fl in AVX512_FLAGS)
    except OSError:
        pass
    return False


_lib512 = None
if not os.environ.get("OMNI_RING_NO_AVX512") and cpu_has_avx512():
    try:
        _lib512 = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "libomniring_avx512.so"))
        _u8_2d = npct.ndpointer(dtype=np.uint8, ndim=2, flags="C_CONTIGUOUS")
        _i8_1d = npct.ndpointer(dtype=np.int8, ndim=1, flags="C_CONTIGUOUS")
        _i8_2d = npct.ndpointer(dtype=np.int8, ndim=2, flags="C_CONTIGUOUS")
        _i32_1d = npct.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS")
        _i32_2d = npct.ndpointer(dtype=np.int32, ndim=2, flags="C_CONTIGUOUS")
        _u64_1d = npct.ndpointer(dtype=np.uint64, ndim=1, flags="C_CONTIGUOUS")
        _u64_2d = npct.ndpointer(dtype=np.uint64, ndim=2, flags="C_CONTIGUOUS")
        _sz = ctypes.c_size_t
        _lib512.omniring_sift_cplx_avx512.argtypes = [_u8_2d, _sz, _sz, _i8_1d, _i8_1d, _i32_1d]
        _lib512.omniring_sift_batch_cplx_avx512.argtypes = [_u8_2d, _sz, _sz, _i8_2d, _i8_2d, _sz, _i32_2d]
        _lib512.omniring_hamming_sift_avx512.argtypes = [_u64_2d, _sz, _sz, _u64_1d, _i32_1d]
        _lib512.omniring_ternary_sift_avx512.argtypes = [_u64_2d, _u64_2d, _sz, _sz, _u64_1d, _u64_1d, _i32_1d]
        for _fn in ("omniring_sift_cplx_avx512", "omniring_sift_batch_cplx_avx512", "omniring_hamming_sift_avx512",
                    "omniring_ternary_sift_avx512"):
            getattr(_lib512, _fn).restype = None
    except (OSError, AttributeError):
        _lib512 = None


def has_kernel() -> bool:
    """True when the AVX2 sift kernel is built (``./build_kernel.sh``)."""
    return _lib is not None


def has_avx512() -> bool:
    """True when the CPU has the AVX-512 flags AND libomniring_avx512.so loaded (``./build_kernel_avx512.sh``)."""
    return _lib512 is not None


def backend_for(d: int, backend: str = "auto") -> str:
    """The backend scores() will use for a lattice of width d: avx512 > c (AVX2, d % 32 == 0) > numpy."""
    if backend != "auto":
        return backend
    if _lib512 is not None:
        return "avx512"
    if _lib is not None and d % 32 == 0:
        return "c"
    return "numpy"


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
    """Re(conj(w_n) v) / D for every lattice row n (a stored mark scores ~1). backend: auto | avx512 | c (AVX2) | numpy.
    All backends return identical values (integer scores, same scaling)."""
    n, d = lattice.shape
    qr, qi, scale = quantize_slot(v)
    if scale == 0.0:
        return np.zeros(n, dtype=np.float64)
    be = backend_for(d, backend)
    if be == "avx512":
        if _lib512 is None:
            raise RuntimeError("AVX-512 kernel unavailable (CPU flags missing or ./build_kernel_avx512.sh not run)")
        out = np.empty(n, dtype=np.int32)
        _lib512.omniring_sift_cplx_avx512(np.ascontiguousarray(lattice), n, d, qr, qi, out)
        return out.astype(np.float64) * scale / (127.0 * d)
    use_c = be == "c"
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


def scores_batch(vs: Sequence[np.ndarray], lattice: np.ndarray, backend: str = "auto") -> np.ndarray:
    """scores() for many slots at once (nq x n). The AVX-512 path runs them in one parallel region; else per slot."""
    n, d = lattice.shape
    if backend_for(d, backend) != "avx512":
        return np.stack([scores(v, lattice, backend) for v in vs])
    qs = [quantize_slot(v) for v in vs]
    qr = np.ascontiguousarray(np.stack([q[0] for q in qs]))
    qi = np.ascontiguousarray(np.stack([q[1] for q in qs]))
    out = np.empty((len(qs), n), dtype=np.int32)
    _lib512.omniring_sift_batch_cplx_avx512(np.ascontiguousarray(lattice), n, d, qr, qi, len(qs), out)
    sc = np.array([q[2] for q in qs])[:, None]
    return out.astype(np.float64) * sc / (127.0 * d)


def pack_bits(bits: np.ndarray) -> np.ndarray:
    """(n x D) or (D,) bool/0-1 array -> (n x W) / (W,) uint64 words, bit i of dim at word i//64, bit i%64 (zero-padded)."""
    b = np.atleast_2d(np.asarray(bits, dtype=bool))
    n, dd = b.shape
    w = (dd + 63) // 64
    pad = np.zeros((n, w * 64), dtype=bool)
    pad[:, :dd] = b
    out = np.packbits(pad, axis=1, bitorder="little").view(np.uint64).copy()
    return out if np.ndim(bits) == 2 else out[0]


def pack_ternary(t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """{-1, 0, +1} array (n x D) or (D,) -> (sign plane, active plane) uint64 words (the omniring_polymorphic layout)."""
    t = np.asarray(t)
    return pack_bits(t < 0), pack_bits(t != 0)


def hamming_scores(q: np.ndarray, lattice: np.ndarray, backend: str = "auto") -> np.ndarray:
    """Hamming distance of packed binary query q (W,) uint64 to every row of lattice (n x W) uint64. backend: auto|avx512|numpy."""
    lattice = np.ascontiguousarray(lattice, dtype=np.uint64)
    q = np.ascontiguousarray(q, dtype=np.uint64)
    n, w = lattice.shape
    if backend == "avx512" or (backend == "auto" and _lib512 is not None):
        if _lib512 is None:
            raise RuntimeError("AVX-512 kernel unavailable")
        out = np.empty(n, dtype=np.int32)
        _lib512.omniring_hamming_sift_avx512(lattice, n, w, q, out)
        return out
    return np.bitwise_count(lattice ^ q).sum(1, dtype=np.int64).astype(np.int32)


def ternary_scores(q_sign: np.ndarray, q_active: np.ndarray, lat_sign: np.ndarray, lat_active: np.ndarray,
                   backend: str = "auto") -> np.ndarray:
    """Ternary dot <q, row_n> for 2-bit-plane packed vectors (= omniring_ternary_dot per row). backend: auto|avx512|numpy."""
    ls = np.ascontiguousarray(lat_sign, dtype=np.uint64)
    la = np.ascontiguousarray(lat_active, dtype=np.uint64)
    qs = np.ascontiguousarray(q_sign, dtype=np.uint64)
    qa = np.ascontiguousarray(q_active, dtype=np.uint64)
    n, w = ls.shape
    if backend == "avx512" or (backend == "auto" and _lib512 is not None):
        if _lib512 is None:
            raise RuntimeError("AVX-512 kernel unavailable")
        out = np.empty(n, dtype=np.int32)
        _lib512.omniring_ternary_sift_avx512(ls, la, n, w, qs, qa, out)
        return out
    both = la & qa
    return (np.bitwise_count(both).sum(1, dtype=np.int64)
            - 2 * np.bitwise_count(both & (ls ^ qs)).sum(1, dtype=np.int64)).astype(np.int32)


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
