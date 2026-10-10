"""
Cued multi-sweep readout (PEEL) for OMNIRING.

Engine-level readout of a *crowded* slot: a phase vector holding a superposition
of many marks rather than one. A single cued settle recovers roughly a third of
the marks; repeated sweeps over the residual recover far more, because each
accepted mark is subtracted (explaining away) so the next sweep sees a cleaner
slot.

Provenance: `omniring-research docs/06-ring3-peel-diamond-correl-tabu-fret.md` -> "## PEEL: PASS".
Pre-registered: two 13-ring clamp sweeps WITH explaining-away recover >= 1.25x
the true marks of one plain sweep, with accepted fakes Wilson-99% upper bound
<= 0.05. Measured there at M=24: 14.77 vs 8.45 true marks/slot, fakes 82/2298
(3.6%).

The cue is mandatory, not an optimisation. The RING3 harness disclosure records
that unguided settles on the rotation rings are a *seed lottery*: codebook seeds
0/1/2 decode a clean single mark 15/15, while seeds 3/21/42 give 1/15 -- and 42
is the engine default. Every test and every caller of this module therefore runs
cued: we sweep the known 13-ring value as a clamp and let the soft iteration
settle the other four rings.

The 17-ring is the only legality filter. `check.check_anomaly` CRT-resolves the
five residues and rejects any settle landing outside [0, data_capacity); a settle
that survives it is 17-legal, a settle that raises `ResidueDesyncError` is not
and is skipped.
"""
from typing import List, Optional

import numpy as np

from .chain import encode
from .check import ResidueDesyncError, check_anomaly
from .resonator import decode_soft
from .crt import solve_crt
from .types import DEFAULT_SPEC, PhaseVector, RingSpec

__all__ = [
    "build_codebooks",
    "read_one",
    "read_all",
    "NULL_MAGNITUDE_TOL",
    "DEFAULT_MIN_CONFIDENCE",
]


# Mean component magnitude below which a residual is treated as carrying no mark.
# A full mark has |v_d| == 1 on every dimension (mean 1.0); the null vector has
# mean 0.0. Nothing in between occurs for a slot of <= 24 marks, so this
# threshold only ever fires on a genuinely empty residual.
NULL_MAGNITUDE_TOL = 1e-9


# Eigengap floor on the *uncued* rings for a settle to be accepted.
#
# The cue alone is not a correctness gate. It pins ring 3, so that ring's
# confidence is ~1.0 for every cue value, right or wrong; the evidence lives in
# the other four rings. Measured on a clean single mark (seed 0), a settle that
# CRT-resolves to the true coordinate scores min uncued confidence 0.984, while
# the nearest *17-legal impostor* scores 0.217 and illegal settles score lower
# still. The bands do not overlap, so 0.5 sits in the gap: far above anything a
# false settle produced, far below anything a true settle produced.
#
# Consequence: fakes fall to 0.0% at M=24 while recall *rises* (measured sweep,
# 8 slots/arm -- see omniring-research docs/07-peel-read-all-module.md "Measured").
DEFAULT_MIN_CONFIDENCE = 0.5


def build_codebooks(
    rotations: List[np.ndarray], spec: RingSpec = DEFAULT_SPEC
) -> List[np.ndarray]:
    """
    Build the 5 ring codebooks: codebooks[i] = exp(1j * outer(arange(m_i), rotations[i]))
    for m_i in spec.all_moduli = (3, 5, 7, 13, 17).

    Returns:
        List of complex arrays of shape (m_i, D), indexed by ring position --
        index 3 is the 13-ring that `read_all` cues.
    """
    if len(rotations) < len(spec.all_moduli):
        raise ValueError(
            f"read_all needs {len(spec.all_moduli)} rotations, got {len(rotations)}"
        )
    return [
        np.exp(1j * np.outer(np.arange(m), r))
        for m, r in zip(spec.all_moduli, rotations)
    ]


def _is_null(v: PhaseVector) -> bool:
    """True when the residual carries no phase energy left to read."""
    return bool(float(np.mean(np.abs(v))) < NULL_MAGNITUDE_TOL)


def _settle_cued(
    residual: PhaseVector,
    codebooks: List[np.ndarray],
    cue: int,
    spec: RingSpec,
    cue_ring: int,
    iterations: int,
    min_confidence: float,
) -> Optional[int]:
    """
    One cued settle against `residual`, or None when it is not 17-legal and
    confident.

    The cue value is clamped into ring `cue_ring`; the soft iteration settles the
    remaining four. A settle is kept only if the 17-ring check passes *and* every
    uncued ring clears `min_confidence`. A settle that fails either is discarded,
    never guessed at.
    """
    r = decode_soft(
        residual, codebooks, iterations=iterations, clamp=cue, clamp_idx=cue_ring
    )
    uncued_conf = min(
        c for i, c in enumerate(r.confidences) if i != cue_ring
    )
    if uncued_conf < min_confidence:
        return None
    try:
        return check_anomaly(r.residues, spec)
    except ResidueDesyncError:
        return None


def read_one(
    v: PhaseVector,
    rotations: List[np.ndarray],
    cue: int,
    spec: RingSpec = DEFAULT_SPEC,
    cue_ring: int = 3,
    iterations: int = 30,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> Optional[int]:
    """
    A single cued settle of `v`. Returns the 17-legal coordinate, or None when
    the settle is illegal, not confident, or the slot carries no mark.

    `cue` is the known 13-ring value to clamp; `cue_ring` is its ring index
    (3 in the 5-ring (3,5,7,13,17) layout).

    Args:
        v: Complex phase vector of shape (D,).
        rotations: Base rotation vectors, one per ring.
        cue: Known value to clamp into the cue ring.
        spec: Ring specification.
        cue_ring: Ring index to clamp (default 3 == the 13-ring).
        iterations: Soft-settling power-iteration count.
        min_confidence: Eigengap floor on the uncued rings.

    Returns:
        The verified coordinate, or None.
    """
    if _is_null(v):
        return None
    codebooks = build_codebooks(rotations, spec)
    return _settle_cued(
        v, codebooks, cue, spec, cue_ring, iterations, min_confidence
    )


def read_all(
    v: PhaseVector,
    rotations: List[np.ndarray],
    spec: RingSpec = DEFAULT_SPEC,
    cue_ring: int = 3,
    sweeps: int = 2,
    iterations: int = 30,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> List[int]:
    """
    Engine-level readout of a crowded slot by cued sweeps with explaining-away.

    For each of `sweeps` sweeps, sweep every cue value c of the cue ring; run a
    cued settle of the current residual; keep the result only if it is 17-legal,
    confident on every uncued ring, and not already accepted; then subtract its
    exact codeword from the slot so the next sweep reads a cleaner residual.

    A sweep against an exhausted (null) residual stops the readout: the soft
    iteration's ReLU-plus-normalise path settles to residue 0 on every ring for a
    null input, which CRT-resolves to coordinate 0 -- a 17-*legal* coordinate.
    Reading it would report a mark that is not in the slot. See OPEN ISSUES in
    `omniring-research docs/07-peel-read-all-module.md`.

    Args:
        v: Complex phase vector of shape (D,) holding the superposition.
        rotations: Base rotation vectors, one per ring.
        spec: Ring specification.
        cue_ring: Ring index to clamp (default 3 == the 13-ring).
        sweeps: Number of full cue sweeps over the residual.
        iterations: Soft-settling power-iteration count per settle.
        min_confidence: Eigengap floor on the uncued rings.

    Returns:
        Accepted coordinates in acceptance order, deduplicated.
    """
    if not 0 <= cue_ring < len(spec.all_moduli):
        raise ValueError(
            f"cue_ring {cue_ring} out of range for {len(spec.all_moduli)} rings"
        )

    codebooks = build_codebooks(rotations, spec)
    n_cues = spec.all_moduli[cue_ring]

    accepted: List[int] = []
    accepted_set = set()
    residual = v.copy()

    for _ in range(sweeps):
        if _is_null(residual):
            break
        for c in range(n_cues):
            x = _settle_cued(
                residual, codebooks, c, spec, cue_ring, iterations, min_confidence
            )
            if x is None or x in accepted_set:
                continue
            accepted.append(x)
            accepted_set.add(x)
            # Explaining away: subtract the exact codeword of the accepted mark.
            residual = residual - encode(x, rotations)

    return accepted


def read_all_kernel(v: PhaseVector, rotations: List[np.ndarray], spec: RingSpec = DEFAULT_SPEC, cue_ring: int = 3,
                    blk_len: int = 1024, min_proj: float = 0.5, max_marks: int = 128) -> List[int]:
    """BUS-PEEL-0 (2026-10-02): cued explaining-away on the AVX2 kernel. Cue one ring (default the 13-ring, as read_all), joint-
    search the other four rings on a `blk_len`-dim block of the residual via a 16-phase product lattice (built once), accept
    the best candidate if it is 17-legal and still explains >= min_proj of a unit mark, subtract its exact codeword, repeat.
    Measured on the bus format (D 5,120): 24/24 · 32/32 · 48/48 · 64/64 true marks per slot, 0 fakes, 0.2-0.4 s (read_all: 16.65/24)."""
    from . import route as _route
    mods = spec.all_moduli; D = spec.dimension; N = spec.data_capacity
    cbs = build_codebooks(rotations, spec)
    J = [i for i in range(len(mods)) if i != cue_ring]
    blk = np.arange(min(blk_len, D))
    lat, g = _route._block_lattice([cbs[i] for i in J], tuple(mods[i] for i in J), blk)
    cbc = np.conj(cbs[cue_ring][:, blk])
    resid = np.asarray(v, dtype=complex).copy(); found: List[int] = []
    for _ in range(max_marks):
        U = resid[blk][None, :] * cbc
        s = 127.0 / np.maximum(np.abs(U).max(1, keepdims=True), 1e-9)
        qr = np.rint(U.real * s).astype(np.int8); qi = np.rint(U.imag * s).astype(np.int8)
        best = None
        for c in range(mods[cue_ring]):
            sc = _route._sift_cplx_int8(qr[c], qi[c], lat); top = int(np.argmax(sc))
            if best is None or sc[top] > best[1]:
                res = [0] * len(mods)
                for k, i in enumerate(J):
                    res[i] = int(g[top, k])
                res[cue_ring] = c; best = (res, int(sc[top]))
        res = best[0]; x = solve_crt(res, mods)
        w = np.ones(D, dtype=complex)
        for i, r in enumerate(res):
            w *= cbs[i][r]
        proj = float((np.vdot(w, resid)).real / D)
        if proj < min_proj or x >= N or x in found:
            break
        found.append(x); resid = resid - w
    return found
