"""CORE-ROUTE-0 — the decode ladder for product-encoded residue slots: SIEVE -> PART -> PEEL -> ABSTAIN, each with a CHECK.
Cheapest first; escalate only when the check fails (the zoom-lens: widen on a low margin). Every rung leaves a mark.
  SIEVE: elementwise power kills every ring but one (A3: a cyclic factor dies at its own order); exact on clean phases.
  PART : partitioned (fractal-compression) resonator — joint search on a few blocks, then per-ring block votes on the
         residual unbound by the other rings' estimates (sub-linear in M); robust to noise and 4-bit phases.
  PEEL : PART, subtract the exact codeword, repeat (explaining-away) for superposed slots.
  CHECK: re-encode the answer, cosine against the (residual) slot."""
from __future__ import annotations
import time
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
from .crt import solve_crt
from . import seed as _seed

_LAT: Dict = {}   # (id(cbs), block start, block len) -> (uint8 lattice M x d_b, residue table)


def encode_prod(x: int, rots: Sequence[np.ndarray]) -> np.ndarray:
    return np.exp(1j * sum(x * r for r in rots))


def codeword(res: Sequence[int], cbs: Sequence[np.ndarray]) -> np.ndarray:
    v = np.ones(cbs[0].shape[1], dtype=complex)
    for c, r in zip(cbs, res):
        v = v * c[r]
    return v


def check(v: np.ndarray, res: Sequence[int], cbs: Sequence[np.ndarray]) -> float:
    w = codeword(res, cbs)
    return float(abs(np.vdot(w, v)) / (np.linalg.norm(w) * max(np.linalg.norm(v), 1e-12)))


def sieve(v: np.ndarray, cbs: Sequence[np.ndarray], ms: Sequence[int]) -> List[int]:
    res = []
    for j, m in enumerate(ms):
        p = int(np.prod([mm for k, mm in enumerate(ms) if k != j]))
        w = v ** p
        r = int(np.argmax((np.conj(cbs[j]) @ w).real))
        res.append((r * pow(p, -1, m)) % m)
    return res


def _block_lattice(cbs: Sequence[np.ndarray], ms: Sequence[int], blk: np.ndarray):
    """The product codebook restricted to one block, as a 16-phase uint8 lattice for the AVX2 sift kernel (built once)."""
    key = (id(cbs[0]), len(ms), int(blk[0]), len(blk), int(blk[-1]), int(blk[len(blk) // 2]))
    if key in _LAT:
        return _LAT[key]
    n = len(cbs); M = int(np.prod(ms))
    g = np.stack(np.meshgrid(*[np.arange(m) for m in ms], indexing="ij"), -1).reshape(-1, n)
    ang = np.zeros((M, len(blk)))
    for i in range(n):
        ang += np.angle(cbs[i][:, blk])[g[:, i]]
    lat = (np.round(ang / (2 * np.pi / 16)).astype(np.int64) % 16).astype(np.uint8)
    _LAT[key] = (np.ascontiguousarray(lat), g)
    return _LAT[key]


def partition_blocks(kind: str, D: int, B: int = 32, rng: Optional[np.random.Generator] = None) -> List[np.ndarray]:
    """The partition SHAPE: CONTIG · STRIDE · TRI (residue classes of the dim index) · HEX (hex cells on a grid) · RANDOM."""
    idx = np.arange(D)
    if kind == "CONTIG":
        return [idx[k * D // B:(k + 1) * D // B] for k in range(B)]
    if kind == "STRIDE":
        return [idx[k::B] for k in range(B)]
    if kind == "TRI":
        cls = (idx % 3) * 5 + (idx % 5); half = idx // (D // 2)
        return [idx[(cls == c) & (half == h)] for c in range(15) for h in range(2)]
    if kind == "HEX":
        W = int(np.sqrt(D)); H = D // W; r = max(1, int(np.sqrt(D / B / 3)))
        x = idx % W; y = idx // W
        q = np.floor(x / (1.5 * r)).astype(int); rr = np.floor((y - (q % 2) * 0.87 * r) / (1.73 * r)).astype(int)
        cell = q * 1000 + rr
        return [idx[cell == c] for c in np.unique(cell)]
    if kind == "RANDOM":
        p = (rng or np.random.default_rng(0)).permutation(D)
        return [p[k * D // B:(k + 1) * D // B] for k in range(B)]
    raise ValueError(kind)


def part(v: np.ndarray, cbs: Sequence[np.ndarray], ms: Sequence[int], B: int = 32, joint_blocks: int = 2, iters: int = 3,
         rng: Optional[np.random.Generator] = None, exact_seed: bool = False, partition: str = "CONTIG") -> Tuple[List[int], float]:
    """Partitioned resonator, v2: the joint seed search runs on the AVX2 sift kernel over a 16-phase block lattice
    (M x d_b uint8, built once per codebook); then per-ring block votes on the unbound residual (cheap)."""
    D = v.shape[0]; n = len(cbs)
    blocks = partition_blocks(partition, D, B, rng)
    blocks = [np.ascontiguousarray(b[: (len(b) // 32) * 32]) if len(b) >= 32 else b for b in blocks]   # kernel needs d % 32 == 0
    votes = [np.zeros(m) for m in ms]
    for blk in [b for b in blocks if len(b) >= 32][:joint_blocks]:
        lat, g = _block_lattice(cbs, ms, blk)
        if exact_seed:                                     # superposed slots: keep the sum's magnitude (complex matched filter)
            prod = np.ones((len(g), len(blk)), dtype=complex)
            for i in range(n):
                prod *= cbs[i][:, blk][g[:, i]]
            sc = (np.conj(prod) @ v[blk]).real
        else:
            q = (np.round(np.angle(v[blk]) / (2 * np.pi / 16)).astype(np.int64) % 16).astype(np.uint8)
            sc = _seed.sift_phase(q, lat)                 # int32 scores of the block query against every combination
        top = int(np.argmax(sc))
        for i in range(n):
            votes[i][g[top, i]] += 1.0
    est = [int(np.argmax(w)) for w in votes]
    agree = 0.0
    for _ in range(iters):
        new = []; agree = 0.0
        for i in range(n):
            u = v.copy()
            for j in range(n):
                if j != i:
                    u = u * np.conj(cbs[j][est[j]])
            w = np.zeros(ms[i])
            for blk in blocks:
                s = (np.conj(cbs[i][:, blk]) @ u[blk]).real
                w[int(np.argmax(s))] += 1.0
            new.append(int(np.argmax(w))); agree += w.max() / B
        est = new; agree /= n
    return est, agree


def peel(v: np.ndarray, cbs: Sequence[np.ndarray], ms: Sequence[int], max_marks: int = 8, min_cos: float = 0.25) -> List[List[int]]:
    """Explaining-away over a superposed slot: PART, accept if the codeword still explains the residual, subtract, repeat."""
    resid = v.astype(complex).copy(); found = []
    for _ in range(max_marks):
        est, agree = part(resid, cbs, ms, joint_blocks=1, exact_seed=True)
        w = codeword(est, cbs)
        c = float((np.vdot(w, resid)).real / len(w))        # projection of the residual on the candidate (1.0 = one clean mark)
        if c < min_cos or agree < 0.5:
            break
        found.append(est); resid = resid - c * w
    return found


def _sift_cplx_int8(qr: np.ndarray, qi: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Exact-magnitude matched filter on the AVX2 kernel: arbitrary int8 (re, im) query against a 16-phase lattice."""
    from .sift import _lib
    lat = np.ascontiguousarray(lat, dtype=np.uint8); n, d = lat.shape
    out = np.empty(n, dtype=np.int32)
    _lib.omniring_sift_cplx(lat, n, d, np.ascontiguousarray(qr, dtype=np.int8), np.ascontiguousarray(qi, dtype=np.int8), out)
    return out


def peel_kernel(v: np.ndarray, cbs: Sequence[np.ndarray], ms: Sequence[int], max_marks: int = 32, min_proj: float = 0.5,
                blk_len: int = 128) -> List[List[int]]:
    """PEEL-2c: cued peel with the exact-magnitude seed on the kernel (int8 re/im of the residual block vs the 2-ring lattice)."""
    D = v.shape[0]; resid = v.astype(complex).copy(); found = []
    blk = np.arange(blk_len); lat, g = _block_lattice(cbs[:2], (ms[0], ms[1]), blk)
    cb3c = np.conj(cbs[2][:, blk])                              # m3 x blk: unbinding each cue on the block only
    for _ in range(max_marks):
        rb = resid[blk]; U = rb[None, :] * cb3c                 # m3 x blk residual unbound by every cue at once
        s = 127.0 / np.maximum(np.abs(U).max(1, keepdims=True), 1e-9)      # per-cue int8 scale
        qr = np.rint(U.real * s).astype(np.int8); qi = np.rint(U.imag * s).astype(np.int8)
        best = None
        for c in range(ms[2]):
            sc = _sift_cplx_int8(qr[c], qi[c], lat); top = int(np.argmax(sc))
            cand = [int(g[top, 0]), int(g[top, 1]), c]
            if best is None or sc[top] > best[1]:
                best = (cand, int(sc[top]))
        w = codeword(best[0], cbs); proj = float((np.vdot(w, resid)).real / D)
        if proj < min_proj:
            break
        found.append(best[0]); resid = resid - w
    return found


def peel_cued(v: np.ndarray, cbs: Sequence[np.ndarray], ms: Sequence[int], max_marks: int = 32, min_proj: float = 0.6,
              blk_len: int = 32) -> List[List[int]]:
    """PEEL-2: cue ring 3 (clamp each residue c), joint-search rings 1-2 on one block via the kernel lattice, check, subtract."""
    D = v.shape[0]; resid = v.astype(complex).copy(); found = []
    ms12 = (ms[0], ms[1]); blk = np.arange(blk_len)
    lat, g = _block_lattice(cbs[:2], ms12, blk)
    for _ in range(max_marks):
        best = None
        for c in range(ms[2]):
            u = resid * np.conj(cbs[2][c])
            q = (np.round(np.angle(u[blk]) / (2 * np.pi / 16)).astype(np.int64) % 16).astype(np.uint8)
            sc = _seed.sift_phase(q, lat); top = int(np.argmax(sc))
            cand = [int(g[top, 0]), int(g[top, 1]), c]
            w = codeword(cand, cbs); proj = float((np.vdot(w, resid)).real / D)
            if proj >= min_proj and (best is None or proj > best[1]):
                best = (cand, proj, w)
        if best is None:
            break
        cand, proj, w = best; found.append(cand); resid = resid - w
    return found


def route(v: np.ndarray, cbs: Sequence[np.ndarray], ms: Sequence[int], superposed_hint: bool = False) -> Dict:
    """The ladder. Returns {rung, answer (x or list of xs), checks, ms}."""
    t0 = time.time(); marks = []
    if not superposed_hint:
        res = sieve(v, cbs, ms); c = check(v, res, cbs); marks.append(("SIEVE", c, time.time() - t0))
        if c >= 0.9:
            return {"rung": 1, "x": solve_crt(res, ms), "marks": marks, "ms": 1000 * (time.time() - t0)}
        res, agree = part(v, cbs, ms); c = check(v, res, cbs); marks.append(("PART", c, time.time() - t0))
        if c >= 0.5 and agree >= 0.6:
            return {"rung": 2, "x": solve_crt(res, ms), "marks": marks, "ms": 1000 * (time.time() - t0)}
    found = peel_kernel(v, cbs, ms); marks.append(("PEEL", float(len(found)), time.time() - t0))
    if found:
        return {"rung": 3, "x": [solve_crt(r, ms) for r in found], "marks": marks, "ms": 1000 * (time.time() - t0)}
    return {"rung": 4, "x": None, "marks": marks, "ms": 1000 * (time.time() - t0)}
