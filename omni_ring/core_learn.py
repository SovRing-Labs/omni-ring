"""CORE-ROUTE-2 — the core learns its own program (J 2026-10-02: "alternate/jitter through all modes every tick, learning the
precise combination to find any query on its own").

The three mechanisms are OPCODES of the vector VM's program language (SIEVE · PART · PEEL · CHECK), a program is one phase
vector with two check slots (vm.ProgramCodebook, keyed checks recommended: PROG-INTEGRITY-0), and the core holds a small
library of stored programs = the 6 orderings of the three modes, each followed by CHECK and a PROG-on-fail to the next rung.
Each tick: read the slot's signals (17-ring legality, magnitude profile, first-check margin) → Thompson-sample a program from
the library for that signal bucket → execute it (modes fire in its order, CHECK after each, stop on the first pass) → leave a
mark (signals, program sha, cost, correct) → update the posterior. The program register therefore changes every tick, chosen
by the core from its own marks. Field: algorithm selection by bandit (Thompson sampling, Agrawal & Goyal 2012); ours: the VM
(CORTEX-2 G1-G4), the ladder (CORE-ROUTE-0), the learner (CORE-ROUTE-1)."""
from __future__ import annotations
import hashlib, itertools, time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple
import numpy as np
from .crt import solve_crt
from .route import sieve, part, peel_kernel, check, codeword
from . import vm as _vm

MODES = ["SIEVE", "PART", "PEEL"]
SEQS = list(itertools.permutations(MODES))
# extend the VM's opcode table with the three mechanisms + CHECK (same 8-arg scheme; ids beyond the base table)
EXT_OPS = _vm.OPS + ["SIEVE", "PART", "PEEL", "CHECK"]


def ext_iid(op: str, arg: int = 0) -> int:
    return EXT_OPS.index(op) * _vm.NARG + arg


@dataclass
class Mark:
    tick: int
    signals: Tuple[int, int]
    program: str
    seq: List[str]
    fired: List[str]
    correct: Optional[bool]
    cost_s: float


@dataclass
class LearningCore:
    cbs: Sequence[np.ndarray]
    ms: Sequence[int]
    seed: int = 0
    library: Dict[str, np.ndarray] = field(default_factory=dict)       # program sha -> program vector
    seq_of: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    alpha: np.ndarray = field(default_factory=lambda: np.ones((3, 2, len(SEQS))))
    beta: np.ndarray = field(default_factory=lambda: np.ones((3, 2, len(SEQS))))
    marks: List[Mark] = field(default_factory=list)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.pcb = _vm.ProgramCodebook(1280, rseed=77)
        self.shas = []
        for s in SEQS:                                                  # the library: one program vector per ordering
            ids = []
            for m in s:
                ids += [ext_iid(m), ext_iid("CHECK")]
            ids = ids[: _vm.L]
            vec = self._write(ids); sha = hashlib.sha256(vec.tobytes()).hexdigest()[:16]
            self.library[sha] = vec; self.seq_of[sha] = s; self.shas.append(sha)
        self.tick_no = 0

    def _write(self, ids: List[int]) -> np.ndarray:
        # the base codebook has 120 atoms; extended ids index the same INSTR table modulo its size (deterministic, checkable)
        ids = [i % _vm.NI for i in ids] + [_vm.iid("NOP")] * (_vm.L - len(ids))
        return self.pcb.write(ids)

    # -- signals -----------------------------------------------------------------------------------------------------
    def signals(self, v: np.ndarray) -> Tuple[int, int]:
        mag = np.abs(v); prof = float(mag.std() / max(mag.mean(), 1e-9))
        b_mag = 0 if prof < 0.05 else (1 if prof < 0.5 else 2)
        res = sieve(v, self.cbs, self.ms); c = check(v, res, self.cbs)      # the first check's margin as a signal (1 ms)
        return b_mag, int(c >= 0.9)

    # -- one tick -------------------------------------------------------------------------------------------------------
    def tick(self, v: np.ndarray, truth: Optional[Set[int]] = None) -> Tuple[Optional[Set[int]], Mark]:
        self.tick_no += 1; t0 = time.time()
        s0, s1 = self.signals(v)
        k = int(np.argmax(self.rng.beta(self.alpha[s0, s1], self.beta[s0, s1])))
        sha = self.shas[k]; seq = self.seq_of[sha]
        ans: Optional[Set[int]] = None; fired = []
        for mode in seq:                                                 # the program's modes fire in its order
            fired.append(mode)
            if mode == "SIEVE":
                res = sieve(v, self.cbs, self.ms)
                if check(v, res, self.cbs) >= 0.9:
                    ans = {solve_crt(res, self.ms)}; break
            elif mode == "PART":
                res, agree = part(v, self.cbs, self.ms)
                if check(v, res, self.cbs) >= 0.5 and agree >= 0.6:
                    ans = {solve_crt(res, self.ms)}; break
            else:
                f = peel_kernel(v, self.cbs, self.ms)
                if f:
                    ans = {solve_crt(r, self.ms) for r in f}; break
        cost = time.time() - t0
        correct = None
        if truth is not None:
            correct = ans is not None and len(ans & truth) >= max(1, int(0.75 * len(truth))) and ans <= truth
            reward = (1.0 if correct else 0.0) * max(0.0, 1.0 - cost / 1.5)   # correct AND cheap: fractional Beta update
            self.alpha[s0, s1, k] += reward; self.beta[s0, s1, k] += 1.0 - reward
        m = Mark(self.tick_no, (s0, s1), sha, list(seq), fired, correct, cost)
        self.marks.append(m)
        return ans, m

    def learned(self) -> Dict[str, List[str]]:
        out = {}
        for s0 in range(3):
            for s1 in range(2):
                p = self.alpha[s0, s1] / (self.alpha[s0, s1] + self.beta[s0, s1])
                out[f"mag{s0}_sieveok{s1}"] = list(SEQS[int(np.argmax(p))])
        return out
