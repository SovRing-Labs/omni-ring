"""CORTEX-2 step 1 — the sifter as code in the space.

A program is ONE phase vector (SOURCE-3's scheme: instruction atoms bound to slot positions, bundled, two coprime check
slots, int8-stored). It lives in a register like any other vector. Each tick the VM reads the program register, refuses it
if either check fails (keeps the last good program), and executes its instructions against named registers and a mark
lattice using the SuperSeed kernels (seed.sift_phase — bit-identical across Python / C AVX2 / SSE / C++).

Opcodes (SOURCE-3's six + the sift's own):
  NOP · HOLD m (accumulator momentum m/8) · SHARPEN a (temperature += a/2) · RELEASE (clear accumulator, flags, temp)
  FREEZE r (register r read-only this tick) · FOCUS g (sift only rows of region g) · LOAD r (query := register r)
  BIND r (query := query ⊕ r, phase add) · UNBIND r (query := query ⊖ r) · BUNDLE r (accumulator += r; query := sector(acc))
  SIFT k (scores := sift(query, lattice); top-k) · THRESHOLD t (keep scores ≥ t/8 of a perfect match; abstain if none or
  margin < spark) · PROG a (IF abstaining: program register := stored program a — the program rewriting itself on its
  own margin; takes effect next tick) · JUMP a (unconditional rewrite to stored program a, next tick)
  MODE m (which sift method the next SIFT fires: 0 exact lattice · 1 cascade = prefix d/4 then rescore top 64 · 2 soft
  blending (seed.soft_sift, temperature = SHARPEN) — the sift methods are instructions placed on the program's slots)
Every tick leaves a mark: {tick, program sha, op trace, top-k, abstain}.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import os

import numpy as np

from . import seed

OPS = ["NOP", "HOLD", "SHARPEN", "RELEASE", "FREEZE", "FOCUS", "LOAD", "BIND", "UNBIND", "BUNDLE", "SIFT", "THRESHOLD", "PROG", "JUMP", "MODE"]
NARG = 8
NI = len(OPS) * NARG            # 120 instruction atoms
L = 8                           # instruction slots
NCHECK = 2
K = 16
PH = np.exp(2j * np.pi * np.arange(K) / K).astype(np.complex64)
PR = (3, 5, 7, 11, 13, 17, 19, 23)   # coprime weights for check 2


def iid(op: str, arg: int = 0) -> int:
    assert 0 <= arg < NARG, arg
    return OPS.index(op) * NARG + arg


def decode_iid(i: int) -> Tuple[str, int]:
    return OPS[i // NARG], i % NARG


class ProgramCodebook:
    """Instruction atoms + slot positions (deterministic from a seed); write/read programs as one vector.
    With `key` (bytes) the two check slots are HMAC-SHA256(key, ids) (PROG-INTEGRITY-0: forgery 1.2e-4 vs 1.0 for public
    checksums); a program written without the key is refused. Default key = env OMNIRING_PROGRAM_KEY, else public checksums."""

    def __init__(self, d: int = seed.D, rseed: int = 77, key: Optional[bytes] = None):
        import os as _os
        g = np.random.default_rng(rseed)
        self.d = d
        env = _os.environ.get("OMNIRING_PROGRAM_KEY")
        self.key = key if key is not None else (env.encode() if env else None)
        self.INSTR = g.integers(0, K, (NI, d)).astype(np.uint8)
        self.POS = g.integers(0, K, (L + NCHECK, d)).astype(np.uint8)
        self.IC = np.cos(2 * np.pi / K * self.INSTR).astype(np.float32)
        self.IS = np.sin(2 * np.pi / K * self.INSTR).astype(np.float32)

    @staticmethod
    def checksum1(ids: Sequence[int]) -> int:
        return int(sum((k + 1) * i for k, i in enumerate(ids)) % NI)

    @staticmethod
    def checksum2(ids: Sequence[int]) -> int:
        return int(sum(p * i for p, i in zip(PR, ids)) % 103)   # 103 prime < NI

    def checks(self, ids: Sequence[int]) -> Tuple[int, int]:
        if self.key is None:
            return self.checksum1(ids), self.checksum2(ids)
        import hashlib, hmac
        h = hmac.new(self.key, bytes(int(i) % 256 for i in ids), hashlib.sha256).digest()
        return int.from_bytes(h[:4], "big") % NI, int.from_bytes(h[4:8], "big") % 103

    def write(self, ids: Sequence[int], check: Optional[Tuple[int, int]] = None) -> np.ndarray:
        ids = list(ids) + [iid("NOP")] * (L - len(ids))
        assert len(ids) == L
        c1, c2 = self.checks(ids) if check is None else check
        acc = sum(PH[(self.POS[k].astype(int) + self.INSTR[i].astype(int)) % K] for k, i in enumerate(ids))
        acc = acc + PH[(self.POS[L].astype(int) + self.INSTR[c1].astype(int)) % K]
        acc = acc + PH[(self.POS[L + 1].astype(int) + self.INSTR[c2].astype(int)) % K]
        s = max(np.abs(acc.real).max(), np.abs(acc.imag).max()) / 127
        return ((np.rint(acc.real / s) + 1j * np.rint(acc.imag / s)) * s).astype(np.complex64)

    def read(self, vec: np.ndarray) -> Optional[List[int]]:
        U = vec[None, :] * np.conj(PH[self.POS])
        sc = U.real @ self.IC.T + U.imag @ self.IS.T
        ids = [int(x) for x in sc.argmax(1)]
        ok = tuple(self.checks(ids[:L])) == (ids[L], ids[L + 1])
        return ids[:L] if ok else None


def program_sha(ids: Sequence[int]) -> str:
    return hashlib.sha256(bytes(int(i) for i in ids)).hexdigest()[:16]


@dataclass
class Mark:
    tick: int
    program: str
    trace: List[str]
    topk: List[int]
    scores_topk: List[int]
    abstain: bool
    refused: bool


@dataclass
class VM:
    lattice: np.ndarray                                   # n x d uint8 phase indices (the marks)
    codebook: ProgramCodebook
    regions: Dict[int, np.ndarray] = field(default_factory=dict)   # FOCUS g -> row indices
    programs: Dict[int, np.ndarray] = field(default_factory=dict)  # PROG a -> program vector
    spark: float = 0.02                                   # abstain if (top1 - top2) / perfect < spark
    backend: str = "auto"                                 # "jit" = the rendered sift (jit.py lut, row-parallel; JIT-3)
    jit_threads: int = 0                                  # 0 = os.cpu_count()
    cascade_plane: str = "p4"                             # "p4" (4-bit packed, +12.5% RAM; JIT-4) | "u8" (+25% RAM; JIT-3)

    def __post_init__(self):
        n, d = self.lattice.shape
        self.d = d
        self.perfect = int(seed.sift_phase(self.lattice[0], self.lattice[:1], backend="numpy")[0])
        self.reg: Dict[int, np.ndarray] = {}              # r -> uint8 phase vector
        self.query = np.zeros(d, dtype=np.uint8)
        self.acc = np.zeros(d, dtype=np.complex64)
        self.temp = 1.0
        self.momentum = 0.0
        self.program_vec: Optional[np.ndarray] = None
        self.program: List[int] = [iid("NOP")] * L
        self.next_program_vec: Optional[np.ndarray] = None
        self.tick_no = 0
        self.marks: List[Mark] = []
        self.scores: Optional[np.ndarray] = None
        self.topk: List[int] = []
        # IR-JIT-NEXT C1a: exact tick memo. Most ticks of a firing loop re-sift an UNCHANGED query (CORTEX-2 step 3: 175,725
        # idle ticks vs 3,304 events in 1 h). Key = query bytes + mode (+ temp for soft) + focus region, valid while the SAME
        # lattice / region objects are in place (the convention the cascade plane already relies on: replace, never mutate).
        # Memoised scores are read-only, so an in-place write raises instead of going stale. OMNIRING_VM_MEMO=0 disables.
        self.memo = os.environ.get("OMNIRING_VM_MEMO", "1") != "0"
        self._memo = None
        self.memo_hits = 0
        self.abstain = False
        self.mode = 0
        # Decode memo: ProgramCodebook.read is a 10 x d @ d x NI matmul (~3 ms at d=5120). The program vector is
        # immutable between rewrites, so re-decoding it every tick is pure waste and was the whole idle-tick cost
        # (3.65 ms p50 vs the 1 ms budget). Keyed on the exact bytes, so a PROG rewrite (or any tamper) still
        # re-decodes and still gets refused on the next tick.
        self._decoded_key: Optional[bytes] = None
        self._decoded_ids: Optional[List[int]] = None
        self.decode_count = 0
        self.decode_skipped = 0

    # -- registers --------------------------------------------------------------------------------------------------
    def set(self, r: int, phase_idx: np.ndarray) -> None:
        self.reg[r] = np.ascontiguousarray(np.asarray(phase_idx, dtype=np.uint8))

    def load_program(self, vec: np.ndarray) -> None:
        self.program_vec = np.asarray(vec, dtype=np.complex64)

    def _exact(self, q: np.ndarray, lat: np.ndarray) -> np.ndarray:
        if self.backend == "jit":
            from . import jit
            return jit.sift_jit(q, lat, "lut", self.jit_threads or os.cpu_count())
        return seed.sift_phase(q, lat, backend=self.backend)

    def _sift(self, focus: Optional[int]) -> np.ndarray:
        """One sift method, chosen by MODE: 0 exact · 1 cascade (prefix d/4, rescore top 64 exactly) · 2 soft blend.
        Memoised: the same query on the same lattice/region/mode returns the same scores without sifting again."""
        fr = focus if (focus is not None and focus in self.regions) else None
        key = (self.query.tobytes(), self.mode, fr, self.temp if self.mode == 2 else None)
        m = self._memo
        if self.memo and m is not None and m[0] is self.lattice and m[1] == key and m[2] is (self.regions[fr] if fr is not None else None):
            self.memo_hits += 1
            return m[3]
        sc = self._sift_uncached(focus)
        sc.flags.writeable = False
        if self.memo:
            self._memo = (self.lattice, key, self.regions[fr] if fr is not None else None, sc)
        return sc

    def _order(self, sc: np.ndarray) -> np.ndarray:
        """Rows by score desc, lower index on ties (stable) — computed once per score vector object, so a memoised tick
        pays neither the sift nor the O(n log n) sort again."""
        c = getattr(self, "_ord", None)
        if c is not None and c[0] is sc:
            return c[1]
        o = np.argsort(-sc.astype(np.int64), kind="stable")
        self._ord = (sc, o)
        return o

    def append(self, rows: np.ndarray) -> None:
        """Write new marks (rows appended to the lattice). A memoised EXACT whole-lattice score vector is extended by sifting
        only the new rows (scores are per-row independent); any other memo is dropped."""
        rows = np.atleast_2d(np.asarray(rows, dtype=np.uint8))
        new = np.vstack([self.lattice, rows])
        m = self._memo
        if self.memo and m is not None and m[0] is self.lattice and m[1][1] == 0 and m[1][2] is None:
            q = np.frombuffer(m[1][0], dtype=np.uint8)
            ext = np.concatenate([m[3], np.asarray(self._exact(q, np.ascontiguousarray(rows)), dtype=np.int32)])
            ext.flags.writeable = False
            self._memo = (new, m[1], None, ext)
        else:
            self._memo = None
        self.lattice = new

    def _sift_uncached(self, focus: Optional[int]) -> np.ndarray:
        lat, rows = self.lattice, None
        if focus is not None and focus in self.regions:
            rows = self.regions[focus]; lat = self.lattice[rows]
        if self.mode == 1 and lat.shape[1] % 128 == 0 and len(lat) > 64:
            d0 = lat.shape[1] // 4
            if self.backend == "jit" and lat.flags.c_contiguous:
                from . import jit
                th = self.jit_threads or os.cpu_count()
                if rows is None:
                    # CASCADE-2 / JIT-3/4: the prefix plane (n x d0, contiguous; p4 = 4 bits/phase) is built ONCE per lattice and kept;
                    # streamed contiguously it is 0.29x the exact pass, vs 0.47x for the strided in-place read and a
                    # 128 MB copy per query before (SCALE-0 defect). Rebuilt only if the lattice object or d0 changes.
                    key = (id(self.lattice), d0, self.cascade_plane)
                    if getattr(self, "_plane_key", None) != key:
                        pl = np.ascontiguousarray(self.lattice[:, :d0])
                        self._plane = seed.pack4(pl) if self.cascade_plane == "p4" else pl; self._plane_key = key
                    if self.cascade_plane == "p4":    # JIT-4: half the bytes, nibbles split in registers — faster AND smaller
                        coarse = jit.sift_p4_jit(self.query, self._plane, d0, th)
                    else:
                        coarse = jit.sift_jit(self.query[:d0], self._plane, "lut", th)
                else:                                                # focus subset: prefix read in place, strided
                    coarse = jit.sift_prefix_jit(self.query, lat, d0, th)
            else:
                coarse = seed.sift_phase(self.query[:d0], np.ascontiguousarray(lat[:, :d0]), backend=self.backend)
            # the same 64 rows a stable argsort would keep (score desc, lower index on ties), in O(n): partition, not sort
            kth = np.partition(coarse, len(coarse) - 64)[len(coarse) - 64]
            above = np.flatnonzero(coarse > kth)
            top = np.concatenate([above, np.flatnonzero(coarse == kth)[:64 - len(above)]])
            sc = np.full(len(lat), np.iinfo(np.int32).min, dtype=np.int32)
            sc[top] = self._exact(self.query, np.ascontiguousarray(lat[top]))
        elif self.mode == 2:
            w = seed.soft_sift(self.query[None, :], lat, beta=4.0 * self.temp)[0]
            sc = np.rint(w / max(float(w.max()), 1e-12) * self.perfect).astype(np.int32)   # best row = a perfect score
        else:
            sc = self._exact(self.query, lat)
        if rows is not None:
            full = np.full(len(self.lattice), np.iinfo(np.int32).min, dtype=np.int32); full[rows] = sc; sc = full
        return sc

    # -- one tick ---------------------------------------------------------------------------------------------------
    def tick(self) -> Mark:
        self.tick_no += 1
        if self.next_program_vec is not None:              # a PROG from last tick takes effect now
            self.program_vec = self.next_program_vec
            self.next_program_vec = None
        refused = False
        if self.program_vec is not None:
            key = self.program_vec.tobytes()
            if key == self._decoded_key:                    # same bytes -> same decode, provably
                ids = self._decoded_ids
                self.decode_skipped += 1
            else:
                ids = self.codebook.read(self.program_vec)
                self._decoded_key, self._decoded_ids = key, ids
                self.decode_count += 1
        else:
            ids = None
        if ids is None:
            refused = True                                 # tampered / corrupted: keep the last good program
        else:
            self.program = ids
        frozen: set = set()
        focus: Optional[int] = None
        trace: List[str] = []
        self.abstain = False
        for i in self.program:
            op, arg = decode_iid(i)
            trace.append(f"{op}{arg}" if op not in ("NOP", "RELEASE") else op)
            if op == "NOP":
                continue
            if op == "HOLD":
                self.momentum = arg / NARG
            elif op == "SHARPEN":
                self.temp += 0.5 * arg
            elif op == "RELEASE":
                self.acc[:] = 0; self.temp = 1.0; self.momentum = 0.0; self.scores = None; self.topk = []
            elif op == "FREEZE":
                frozen.add(arg)
            elif op == "FOCUS":
                focus = arg
            elif op == "LOAD":
                if arg in self.reg:
                    self.query = self.reg[arg].copy()
            elif op == "BIND":
                if arg in self.reg:
                    self.query = ((self.query.astype(np.int16) + self.reg[arg]) % K).astype(np.uint8)
            elif op == "UNBIND":
                if arg in self.reg:
                    self.query = ((self.query.astype(np.int16) - self.reg[arg]) % K).astype(np.uint8)
            elif op == "BUNDLE":
                if arg in self.reg and arg not in frozen:
                    self.acc = self.momentum * self.acc + PH[self.reg[arg]]
                    self.query = seed.sector16(np.rint(self.acc.real * 127).astype(np.int64),
                                               np.rint(self.acc.imag * 127).astype(np.int64))
            elif op == "MODE":
                self.mode = arg
            elif op == "SIFT":
                k = max(1, arg)
                sc = self._sift(focus)
                self.scores = sc
                self.topk = [int(x) for x in self._order(sc)[:k]]
            elif op == "THRESHOLD":
                if self.scores is None:
                    self.abstain = True
                else:
                    t = (arg / NARG) * self.perfect
                    keep = [r for r in self.topk if self.scores[r] >= t]
                    if not keep:
                        self.abstain = True
                    else:
                        o = self._order(self.scores)       # the two best values = np.sort(scores)[::-1][:2]
                        margin = (int(self.scores[o[0]]) - int(self.scores[o[1]])) / self.perfect if len(o) > 1 else 1.0
                        self.abstain = margin < self.spark
                    self.topk = keep
            elif op == "PROG":
                if self.abstain and arg in self.programs:
                    self.next_program_vec = self.programs[arg]   # rewrites itself when unsure; next tick
            elif op == "JUMP":
                if arg in self.programs:
                    self.next_program_vec = self.programs[arg]   # unconditional rewrite; next tick
        m = Mark(self.tick_no, program_sha(self.program), trace, list(self.topk),
                 [int(self.scores[r]) for r in self.topk] if self.scores is not None else [], self.abstain, refused)
        self.marks.append(m)
        return m
