"""
SuperSeed Spec v1 — the portable core of the ring (RING-SEED plan, 2026-09-30).

Why: ring heads and shards are DATA (marks) plus a deterministic encoder, so moving to another language (Rust, ISPC, GLSL,
CUDA, WGSL) needs no retraining — provided every dialect produces byte-identical marks. Before v1, codewords came from numpy's
PCG64 (`default_rng(crc32(label)).integers`), which any port would have to re-implement exactly. v1 uses only integer
operations that every dialect has, plus IEEE single-precision + - * / (correctly rounded everywhere; no transcendentals).

Four working parts + one check (J's 4 + 1):
  1. codeword(label, seed, d)   4-bit phases from splitmix64 over FNV-1a-64(label) — integer only.
  2. ternary_projection(...)    {-1, 0, +1} matrices from the same hash (P(0) = 1/2, P(+-1) = 1/4 each).
  3. encode(x, proj)            float32 -> int8 (per-row absmax, round-half-to-even) -> int32 ternary sums (re, im) ->
                                 16-sector phase by integer comparisons against fixed tangent constants (no atan2).
  4. sift_phase(q, lattice)     query given as phase indices -> exact int8 cos/sin table -> AVX2 kernel (int32 scores).
  +1 conformance: `golden()` hashes of fixed codewords / projections / encodings / scores; tests/golden/seed_v1.json pins them.
"""
from __future__ import annotations

import hashlib
import os
import json
from typing import Dict, Iterable, Optional, Sequence

import numpy as np

SPEC_VERSION = 1
D = 5120
K = 16
M64 = np.uint64(0xFFFFFFFFFFFFFFFF)
_GOLD = np.uint64(0x9E3779B97F4A7C15)
# floor(tan(theta) * 2^16) at the 16-sector boundaries inside one quadrant: 11.25, 33.75, 56.25, 78.75 degrees.
TAN_Q16 = (13035, 43789, 98081, 329471)
C8 = np.array([127, 117, 90, 49, 0, -49, -90, -117, -127, -117, -90, -49, 0, 49, 90, 117], dtype=np.int8)
S8 = np.array([0, 49, 90, 117, 127, 117, 90, 49, 0, -49, -90, -117, -127, -117, -90, -49], dtype=np.int8)


def fnv1a64(data: bytes) -> int:
    h = 0xCBF29CE484222325
    for b in data:
        h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def splitmix64(x: np.ndarray) -> np.ndarray:
    """Vectorised splitmix64 finaliser over uint64 (wrap-around arithmetic, as in C/Rust/ISPC)."""
    with np.errstate(over="ignore"):
        z = (x + _GOLD) & M64
        z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & M64
        z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & M64
        return z ^ (z >> np.uint64(31))


def _key(label: str, seed: int) -> np.uint64:
    return np.uint64((fnv1a64(label.encode("utf-8")) ^ ((seed * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF)) & 0xFFFFFFFFFFFFFFFF)


def nibbles(label: str, seed: int = 0, d: int = D) -> np.ndarray:
    """d 4-bit values: block b = splitmix64(key + b) supplies nibbles 16b .. 16b+15, low nibble first."""
    nb = (d + 15) // 16
    with np.errstate(over="ignore"):
        h = splitmix64(_key(label, seed) + np.arange(nb, dtype=np.uint64))
    shifts = (np.arange(16, dtype=np.uint64) * np.uint64(4))
    return ((h[:, None] >> shifts[None, :]) & np.uint64(15)).astype(np.uint8).reshape(-1)[:d]


def codeword(label: str, seed: int = 0, d: int = D) -> np.ndarray:
    """Phase indices (0..15) of the codeword for `label` — the v1 replacement for shards.symbol / vec_idx."""
    return nibbles("CW:" + label, seed, d)


def ternary_projection(name: str, rows: int, cols: int, seed: int = 0) -> np.ndarray:
    """rows x cols int8 in {-1, 0, +1}: nibble < 4 -> -1, nibble >= 12 -> +1, else 0."""
    out = np.empty((rows, cols), dtype=np.int8)
    for r in range(rows):
        nb = nibbles(f"TP:{name}:{r}", seed, cols)
        out[r] = np.where(nb < 4, -1, np.where(nb >= 12, 1, 0)).astype(np.int8)
    return out


class Projection:
    """The encoder's fixed ternary projection (re and im halves), in_dim -> d phases."""

    def __init__(self, in_dim: int, d: int = D, seed: int = 0, name: str = "enc"):
        self.in_dim, self.d, self.seed, self.name = in_dim, d, seed, name
        self.re = ternary_projection(f"{name}:re", in_dim, d, seed)
        self.im = ternary_projection(f"{name}:im", in_dim, d, seed)
        self.re_f, self.im_f = self.re.astype(np.float32), self.im.astype(np.float32)

    @property
    def id(self) -> str:
        return f"seedv{SPEC_VERSION}/tp:{self.name}/{self.in_dim}x{self.d}/s{self.seed}"


def quantize_int8(x: np.ndarray) -> np.ndarray:
    """Per-row absmax to int8, IEEE float32 ops only, round half to even (np.rint)."""
    x = np.asarray(x, dtype=np.float32)
    x2 = np.atleast_2d(x)
    m = np.abs(x2).max(axis=1, keepdims=True)
    s = np.where(m > 0, np.float32(127.0) / np.where(m > 0, m, np.float32(1.0)), np.float32(0.0)).astype(np.float32)
    q = np.rint(x2 * s).astype(np.int8)
    return q if x.ndim > 1 else q[0]


def _clib():
    import ctypes
    import numpy.ctypeslib as npct
    from .sift import _lib
    if _lib is None or not hasattr(_lib, "omniring_sector16"):
        return None
    if not getattr(_lib, "_seed_typed", False):
        _lib.omniring_sector16.argtypes = [npct.ndpointer(np.int32, flags="C_CONTIGUOUS"), npct.ndpointer(np.int32, flags="C_CONTIGUOUS"),
                                           ctypes.c_size_t, npct.ndpointer(np.uint8, flags="C_CONTIGUOUS")]
        _lib.omniring_sector16.restype = None
        _lib.omniring_sift_batch.argtypes = [npct.ndpointer(np.uint8, flags="C_CONTIGUOUS"), ctypes.c_size_t, ctypes.c_size_t,
                                             npct.ndpointer(np.uint8, flags="C_CONTIGUOUS"), ctypes.c_size_t,
                                             npct.ndpointer(np.int32, flags="C_CONTIGUOUS")]
        _lib.omniring_sift_batch.restype = None
        if hasattr(_lib, "omniring_sift_batch_tiled"):
            _lib.omniring_sift_batch_tiled.argtypes = [npct.ndpointer(np.uint8, flags="C_CONTIGUOUS"), ctypes.c_size_t, ctypes.c_size_t,
                                                       npct.ndpointer(np.uint8, flags="C_CONTIGUOUS"), ctypes.c_size_t,
                                                       npct.ndpointer(np.int32, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int, ctypes.c_int]
            _lib.omniring_sift_batch_tiled.restype = None
        if hasattr(_lib, "omniring_sift_p4"):
            u8, i32 = npct.ndpointer(np.uint8, flags="C_CONTIGUOUS"), npct.ndpointer(np.int32, flags="C_CONTIGUOUS")
            _lib.omniring_sift_p4.argtypes = [u8, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, u8, i32]
            _lib.omniring_sift_p4.restype = None
            _lib.omniring_sift_batch_p4.argtypes = [u8, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, u8, ctypes.c_size_t,
                                                    ctypes.c_size_t, i32]
            _lib.omniring_sift_batch_p4.restype = None
        _lib._seed_typed = True
    return _lib


def sector16(re: np.ndarray, im: np.ndarray, backend: str = "auto") -> np.ndarray:
    """Exact 16-sector phase index of (re, im) integer pairs; equals round(angle / 22.5deg) mod 16 off the boundaries.
    C path (OpenMP, all cores) when the values fit int32; numpy path is the reference."""
    lib = _clib() if backend != "numpy" else None
    re_a, im_a = np.asarray(re), np.asarray(im)
    if lib is not None and np.abs(re_a).max(initial=0) < 2 ** 31 and np.abs(im_a).max(initial=0) < 2 ** 31:
        r32 = np.ascontiguousarray(re_a, dtype=np.int32).ravel()
        i32 = np.ascontiguousarray(im_a, dtype=np.int32).ravel()
        out = np.empty(r32.size, dtype=np.uint8)
        lib.omniring_sector16(r32, i32, r32.size, out)
        return out.reshape(re_a.shape)
    ax = np.abs(re.astype(np.int64))
    ay = np.abs(im.astype(np.int64))
    sub = np.zeros(ax.shape, dtype=np.int64)
    for t in TAN_Q16:
        sub += (ay << 16) > ax * t
    xneg, yneg = re < 0, im < 0
    p = np.where(~xneg & ~yneg, sub, np.where(xneg & ~yneg, 8 - sub, np.where(xneg & yneg, 8 + sub, 16 - sub)))
    return (p % 16).astype(np.uint8)


def encode(x: np.ndarray, proj: Projection, centre: Optional[np.ndarray] = None) -> np.ndarray:
    """Float vectors (n x in_dim) -> phase marks (n x d, uint8). Optional centre (float32) subtracted first."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float32))
    if centre is not None:
        x = (x - np.asarray(centre, dtype=np.float32)).astype(np.float32)
    # Spec: exact int32 sums of int8 x ternary. |sum| <= in_dim * 127 < 2**24, so float32 BLAS computes them exactly
    # (every integer below 2**24 is representable) on all cores; numpy has no multithreaded integer matmul.
    assert proj.in_dim * 127 < 2 ** 24
    q = quantize_int8(x).astype(np.float32)
    re = np.rint(q @ proj.re_f).astype(np.int64)
    im = np.rint(q @ proj.im_f).astype(np.int64)
    return sector16(re, im)


def query_int8(phase_idx: np.ndarray):
    """A pure phase query -> exact int8 (re, im) via the kernel's own table (no float rounding)."""
    p = np.asarray(phase_idx, dtype=np.uint8)
    return np.ascontiguousarray(C8[p]), np.ascontiguousarray(S8[p])


def sift_phase(q_idx: np.ndarray, lattice: np.ndarray, backend: str = "auto") -> np.ndarray:
    """Integer scores (int32) of a phase-index query against every lattice row. 127*127*d = identical codeword."""
    from .sift import _lib
    lat = np.ascontiguousarray(lattice, dtype=np.uint8)
    n, d = lat.shape
    qr, qi = query_int8(q_idx)
    if backend != "numpy" and _lib is not None and d % 32 == 0:
        out = np.empty(n, dtype=np.int32)
        _lib.omniring_sift_cplx(lat, n, d, qr, qi, out)
        return out
    return ((C8.astype(np.int32)[lat] * qr.astype(np.int32)).sum(1)
            + (S8.astype(np.int32)[lat] * qi.astype(np.int32)).sum(1)).astype(np.int32)


def tile_shape(d: int) -> tuple:
    """TILE-1: the cube kernel's tile (TQ queries x TM marks) sized so TM*d + TQ*2d fits this CPU's L2 (measured
    1.18x at 1 thread, 1.33x at 4 threads over the hand-picked 32 x 64 on a 45 MB head; every shape bit-identical).
    OMNIRING_TILE="TQ,TM" overrides."""
    env = os.environ.get("OMNIRING_TILE")
    if env:
        tq, tm = (int(x) for x in env.split(","))
        return tq, tm
    try:
        txt = open("/sys/devices/system/cpu/cpu0/cache/index2/size").read().strip().upper()
        l2 = int(txt[:-1]) * (1024 if txt.endswith("K") else 1024 * 1024) if txt[-1] in "KM" else int(txt)
    except (OSError, ValueError):
        l2 = 256 * 1024
    tq = 8
    tm = max(8, min(256, ((l2 - tq * 2 * d) // max(d, 1)) // 8 * 8))
    return tq, tm


def sift_batch(queries: np.ndarray, lattice: np.ndarray, backend: str = "auto") -> np.ndarray:
    """Integer scores (nq x n int32) of many phase queries against a lattice in one parallel C call (threads stay busy)."""
    lat = np.ascontiguousarray(lattice, dtype=np.uint8)
    qs = np.ascontiguousarray(np.atleast_2d(queries), dtype=np.uint8)
    n, d = lat.shape
    lib = _clib() if backend != "numpy" else None
    if lib is not None and d % 32 == 0 and d <= 8192:
        out = np.empty((len(qs), n), dtype=np.int32)
        if len(qs) >= 4 and hasattr(lib, "omniring_sift_batch_tiled"):
            # TILE-1 geometry: tile (queries x marks) fitted to this CPU's L2 (tile_shape); bit-identical
            lib.omniring_sift_batch_tiled(lat, n, d, qs, len(qs), out, *tile_shape(d), 1)
        else:
            lib.omniring_sift_batch(lat, n, d, qs, len(qs), out)
        return out
    return np.stack([sift_phase(q, lat, backend="numpy") for q in qs])


# -- SuperSeed v1.1: packed lattice + cascade (TORUS-TEL step 7) --------------------------------------------------------
# Packing is storage only: 4 bits per phase, layout "block-64 split" (for each 64-dim block, byte j = phase[64b+j] low
# nibble | phase[64b+32+j] high nibble). Scores are bit-identical to the unpacked lattice. A cascade scores every row on
# the first d0 dims (every output dim is an independent ternary projection, so a prefix is an unbiased, noisier estimate),
# keeps the top M, and rescores those on all dims.
def pack4(lattice: np.ndarray) -> np.ndarray:
    lat = np.asarray(lattice, dtype=np.uint8)
    n, d = lat.shape
    assert d % 64 == 0, "packed layout needs d % 64 == 0"
    b = lat.reshape(n, d // 64, 2, 32)
    return np.ascontiguousarray(((b[:, :, 1, :] & 15) << 4 | (b[:, :, 0, :] & 15)).reshape(n, d // 2), dtype=np.uint8)


def unpack4(packed: np.ndarray) -> np.ndarray:
    p = np.asarray(packed, dtype=np.uint8)
    n, h = p.shape
    b = p.reshape(n, h // 32, 32)
    return np.ascontiguousarray(np.stack([b & 15, b >> 4], axis=2).reshape(n, h * 2))


def sift_batch_p4(queries: np.ndarray, packed: np.ndarray, d_used: Optional[int] = None) -> np.ndarray:
    """Scores (nq x n int32) of phase queries against a packed lattice on the first d_used dims (default: all)."""
    pk = np.ascontiguousarray(packed, dtype=np.uint8)
    qs = np.ascontiguousarray(np.atleast_2d(queries), dtype=np.uint8)
    n, stride = pk.shape
    du = d_used or stride * 2
    # IR-JIT-NEXT batch reads (2026-10-06): the C p4 batch kernel walks queries x rows as thin lines (every query re-reads the whole
    # packed plane). For >= 4 queries use JIT-5's register-blocked p4 kernel (8 queries per row load, L2 row tiles): bit-identical,
    # 4-8x faster on a real head (8,738 x 5,120, prefix 1,280). OMNIRING_P4_JIT=0 keeps the C kernel; any JIT failure falls back.
    if len(qs) >= 4 and du % 64 == 0 and os.environ.get("OMNIRING_P4_JIT", "1") != "0":
        try:
            from . import jit
            return jit.sift_batch8_jit(qs, pk, os.cpu_count() or 1, packed=True, d_used=du)
        except Exception:
            pass
    lib = _clib()
    if lib is not None and hasattr(lib, "omniring_sift_batch_p4"):
        out = np.empty((len(qs), n), dtype=np.int32)
        lib.omniring_sift_batch_p4(pk, n, stride, du, qs, qs.shape[1], len(qs), out)
        return out
    return sift_batch(qs[:, :du], unpack4(pk)[:, :du], backend="numpy")


def sift_rows_p4(query: np.ndarray, packed: np.ndarray, rows: np.ndarray, d_used: Optional[int] = None) -> np.ndarray:
    """Scores of ONE phase query against selected rows of a packed lattice (the cascade's rescoring pass)."""
    pk = np.ascontiguousarray(packed, dtype=np.uint8)
    q = np.ascontiguousarray(query, dtype=np.uint8)
    r = np.ascontiguousarray(rows, dtype=np.int64)
    du = d_used or pk.shape[1] * 2
    lib = _clib()
    if lib is not None and hasattr(lib, "omniring_sift_p4"):
        out = np.empty(len(r), dtype=np.int32)
        lib.omniring_sift_p4(pk, pk.shape[1], du, r.ctypes.data, len(r), q, out)
        return out
    return sift_phase(q[:du], unpack4(pk[r])[:, :du], backend="numpy")


def soft_sift(queries: np.ndarray, lattice: np.ndarray, beta: float) -> np.ndarray:
    """Blending clean-up (Plate §VII.A): softmax over normalised sift scores = an attention distribution over the lattice.
    Rows sum to 1; beta is the sharpness (inverse temperature) on the score scale where a stored codeword scores 1."""
    lat = np.ascontiguousarray(lattice, dtype=np.uint8)
    sc = sift_batch(queries, lat).astype(np.float64) / float(sift_phase(lat[0], lat[:1])[0])
    z = beta * sc
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _h(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:32]


def golden(seed: int = 0) -> Dict[str, str]:
    """Conformance vector set: every dialect must reproduce these hashes byte-for-byte."""
    labels = [f"golden-{i}" for i in range(16)] + ["", "ümlaut", "E:Paris", "R:P36"]
    cws = np.stack([codeword(s, seed) for s in labels])
    proj = Projection(64, 256, seed, "golden")
    xs = np.stack([(nibbles(f"GX:{i}", seed, 64).astype(np.float32) - 7.5) / np.float32(7.5) for i in range(16)])
    enc = encode(xs, proj)
    sc = np.stack([sift_phase(cws[i], cws, backend="numpy") for i in range(4)])
    return {"spec": str(SPEC_VERSION), "codewords": _h(cws), "proj_re": _h(proj.re), "proj_im": _h(proj.im),
            "encodings": _h(enc), "scores": _h(sc), "score_self": str(int(sc[0, 0]))}


def check_golden(path: str, seed: int = 0) -> Iterable[str]:
    """Yield the names of fields that differ from the pinned golden file (empty = conformant)."""
    want = json.load(open(path))
    got = golden(seed)
    for k, v in want.items():
        if got.get(k) != v:
            yield k
