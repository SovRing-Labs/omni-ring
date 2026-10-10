#!/usr/bin/env python3
"""SOURCE-2 (PREREGISTRATIONS-OMNITRI.md): the source as code on the torus. Each tick the loop decodes its own program vector
(instruction slots + checksum, bound to position atoms) and executes it; a program that fails its checksum is refused."""
import json, os, sys, time
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import source0 as S0                                     # same rings, slots, intents and CRT as SOURCE-0

D = 5120
PH = np.exp(2j * np.pi * np.arange(16) / 16).astype(np.complex64)
OPS = ["NOP", "HOLD", "SHARPEN", "RELEASE", "FREEZE", "FOCUS"]
NARG = 8
NI = len(OPS) * NARG                                     # 48 instruction atoms
L = 4                                                    # instruction slots
_g = np.random.default_rng(77)
INSTR = _g.integers(0, 16, (NI, D))                      # instruction codebook
POS = _g.integers(0, 16, (L + 1, D))                     # slot position atoms (last = checksum slot)
IC, IS = np.cos(np.pi / 8 * INSTR).astype(np.float32), np.sin(np.pi / 8 * INSTR).astype(np.float32)


def iid(op, arg=0):
    return OPS.index(op) * NARG + arg


def checksum(ids):
    return int(sum((k + 1) * i for k, i in enumerate(ids)) % NI)


def write_program(ids, check=None):
    """Program -> one bundled vector (int8-stored), like any memory on the torus."""
    check = checksum(ids) if check is None else check
    acc = sum(PH[(POS[k] + INSTR[i]) % 16] for k, i in enumerate(ids)) + PH[(POS[L] + INSTR[check]) % 16]
    s = max(np.abs(acc.real).max(), np.abs(acc.imag).max()) / 127
    return ((np.rint(acc.real / s) + 1j * np.rint(acc.imag / s)) * s).astype(np.complex64)


def read_program(vec):
    """Decode every slot by unbinding its position and cleaning up; verify the checksum. -> ids or None (refused)."""
    U = vec[None, :] * np.conj(PH[POS])                  # (L+1, D)
    sc = U.real @ IC.T + U.imag @ IS.T
    ids = [int(x) for x in sc.argmax(1)]
    return ids[:L] if checksum(ids[:L]) == ids[L] else None


NORMAL = [iid("HOLD", 4), iid("SHARPEN", 1), iid("NOP"), iid("NOP")]          # momentum 0.5, temp +0.5 per tick
RELEASE = [iid("RELEASE"), iid("HOLD", 4), iid("SHARPEN", 1), iid("NOP")]


class Source:
    def __init__(self, v):
        self.v = v; self.est = [cb.mean(0) for cb in S0.CB]; self.temp = 1.0
        self.program = NORMAL; self.refused = 0

    def tick(self, program_vec, intent):
        ids = read_program(program_vec)
        if ids is None:
            self.refused += 1                            # tampered / corrupted: keep the last good program
        else:
            self.program = ids
        m, frozen, focus = 0.5, set(), None
        for i in self.program:
            op, arg = OPS[i // NARG], i % NARG
            if op == "HOLD":
                m = arg / NARG
            elif op == "SHARPEN":
                self.temp += 0.5 * arg
            elif op == "RELEASE":
                self.est = [cb.mean(0) for cb in S0.CB]; self.temp = 1.0
            elif op == "FREEZE":
                frozen.add(arg)
            elif op == "FOCUS":
                focus = arg
        self.est[S0.CLAMP] = S0.CB[S0.CLAMP][intent]
        for r in range(len(S0.CB)):
            if r == S0.CLAMP or r in frozen:
                continue
            u = self.v.copy()
            for j in range(len(S0.CB)):
                if j != r:
                    u = u * np.conj(self.est[j])
            a = (np.conj(S0.CB[r]) @ u).real / D                          # the sift
            t = self.temp * (2 if focus == r else 1)
            w = np.exp(t * (a - a.max())); w /= w.sum()
            e = w @ S0.CB[r]; e = e / np.maximum(np.abs(e), 1e-9)
            self.est[r] = m * self.est[r] + (1 - m) * e
            self.est[r] = self.est[r] / np.maximum(np.abs(self.est[r]), 1e-9)

    def answer(self):
        return S0.decoded_x([int(np.argmax((np.conj(cb) @ e).real)) for cb, e in zip(S0.CB, self.est)])


PV_NORMAL, PV_RELEASE = write_program(NORMAL), write_program(RELEASE)


def trial(args):
    M, k = args
    g = np.random.default_rng(10_000 * M + k)                           # identical slots to SOURCE-0
    while True:
        xs = g.choice(S0.DEFAULT_SPEC.data_capacity, M, replace=False)
        r13 = [x % 13 for x in xs]; uniq = [x for x in xs if r13.count(x % 13) == 1]
        if len(uniq) >= 2:
            break
    a, b = int(uniq[0]), int(uniq[1])
    v = sum(S0.encode(int(x)) for x in xs)
    src = Source(v)
    for _ in range(40):
        src.tick(PV_NORMAL, a % 13)
    rec = src.answer() == a
    src.tick(PV_RELEASE, b % 13)                                        # intent changes: the source rewrites itself
    for _ in range(39):
        src.tick(PV_NORMAL, b % 13)
    return M, bool(rec), bool(src.answer() == b)


def live_rewrite(k):
    g = np.random.default_rng(500 + k)
    xs = g.choice(S0.DEFAULT_SPEC.data_capacity, 12, replace=False)
    v = sum(S0.encode(int(x)) for x in xs)
    src = Source(v); intent = int(xs[0]) % 13
    for _ in range(20):
        src.tick(PV_NORMAL, intent)
    freeze = write_program([iid("FREEZE", 1), iid("HOLD", 4), iid("SHARPEN", 1), iid("NOP")])
    before = [e.copy() for e in src.est]
    src.tick(freeze, intent)
    held = np.allclose(src.est[1], before[1])
    moved = any(not np.allclose(src.est[r], before[r]) for r in (0, 2, 4))
    b1 = src.est[1].copy(); src.tick(PV_NORMAL, intent)
    resumed = not np.allclose(src.est[1], b1)
    return held and moved and resumed


def safety(k):
    g = np.random.default_rng(900 + k)
    ids = [int(x) for x in g.integers(0, NI, L)]
    clean_ok = read_program(write_program(ids)) == ids
    tampered = list(ids); tampered[2] = (tampered[2] + 1 + int(g.integers(0, NI - 1))) % NI
    tamper_refused = read_program(write_program(tampered, check=checksum(ids))) is None
    rand = (PH[g.integers(0, 16, D)] * 3).astype(np.complex64)
    rand_refused = read_program(rand) is None
    return clean_ok, tamper_refused, rand_refused


if __name__ == "__main__":
    Ms = (8, 12, 16)
    with Pool(4) as pool:
        out = pool.map(trial, [(M, k) for M in Ms for k in range(200)])
        lr = pool.map(live_rewrite, range(100))
        sf = pool.map(safety, range(200))
    prev = json.load(open(os.path.expanduser("~/gpurun-out-0930/source0.result.json")))
    res = {}
    for M in Ms:
        o = [x for x in out if x[0] == M]
        res[M] = {"recall": sum(x[1] for x in o) / len(o), "switch": sum(x[2] for x in o) / len(o),
                  "source0_recall": prev[str(M)]["source"], "source0_switch": prev[str(M)]["switch"]}
        print(f"SOURCE-2 M={M:2d}: recall {res[M]['recall']:.3f} (SOURCE-0 {res[M]['source0_recall']:.3f}) | "
              f"intent switch {res[M]['switch']:.3f} (SOURCE-0 {res[M]['source0_switch']:.3f})", flush=True)
    res["live_rewrite"] = sum(lr) / len(lr)
    res["safety"] = {"clean_accepted": sum(s[0] for s in sf) / 200, "tampered_refused": sum(s[1] for s in sf) / 200,
                     "random_refused": sum(s[2] for s in sf) / 200}
    print("live rewrite (FREEZE ring 1 takes effect next tick, others move, unfreeze resumes):", res["live_rewrite"], flush=True)
    print("safety:", res["safety"], flush=True)
    xs = np.random.default_rng(1).choice(1365, 12, replace=False); v = sum(S0.encode(int(x)) for x in xs)
    s = Source(v); t = time.perf_counter()
    for _ in range(50):
        s.tick(PV_NORMAL, int(xs[0]) % 13)
    tick = (time.perf_counter() - t) / 50
    t = time.perf_counter()
    for _ in range(50):
        read_program(PV_NORMAL)
    dec = (time.perf_counter() - t) / 50
    res["timing"] = {"ticks_per_s": 1 / tick, "program_decode_ms": dec * 1000, "tick_ms": tick * 1000}
    print("timing:", {k: round(v, 2) for k, v in res["timing"].items()}, flush=True)
    res["verdict"] = {"T1": all(res[M]["recall"] >= res[M]["source0_recall"] - 0.02 for M in Ms), "T2": res[12]["switch"] >= 0.80,
                      "T3": res["live_rewrite"] == 1.0,
                      "T4": res["safety"]["tampered_refused"] == 1.0 and res["safety"]["random_refused"] == 1.0 and res["safety"]["clean_accepted"] == 1.0}
    print("SOURCE-2:", res["verdict"], flush=True)
    json.dump({str(k): v for k, v in res.items()}, open(os.path.expanduser("~/gpurun-out-0930/source2.result.json"), "w"), indent=1)
