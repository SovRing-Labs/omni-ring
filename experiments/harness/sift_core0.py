#!/usr/bin/env python3
"""SIFT-CORE-0 — buckets with a firing core: error-driven refinement of the superposition (PREREGISTRATIONS-OMNITRI.md)."""
import json, os, sys
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tri_wave import Space, PH, D, int8

def trit(a):
    f = lambda x: (np.sign(x) * (np.abs(x) >= 0.5 * np.sqrt(np.mean(x ** 2)))).astype(np.float32)  # noqa: E731
    return (f(a.real) + 1j * f(a.imag)).astype(np.complex64)

sp = Space(1000, 50, seed=13); res = {}
def evaluate(acc, T):
    """per-corner correct + margins + confusers for all members."""
    ok = np.ones(len(T), bool); marg = np.full(len(T), 9.0); conf = []
    for miss in range(3):
        kn = [c for c in range(3) if c != miss]
        sc = sp.scores(np.broadcast_to(acc, (len(T), D)), miss, sp.key(miss, T[:, kn[0]], T[:, kn[1]]))
        tru = sc[np.arange(len(T)), T[:, miss]]; sc[np.arange(len(T)), T[:, miss]] = -9
        best = sc.argmax(1); m = tru - sc[np.arange(len(T)), best]
        ok &= m > 0; marg = np.minimum(marg, m)
        wrong = T.copy(); wrong[:, miss] = best; conf.append((m, wrong))
    return ok, marg, conf

def job(args):
    N, b = args; g = np.random.default_rng(14 * 1000 + N * 10 + b)          # one seeded stream per bucket (parallel-safe)
    T = np.stack([g.integers(0, 1000, N), g.integers(0, 50, N), g.integers(0, 1000, N)], 1)
    ph = PH[sp.tri(T[:, 0], T[:, 1], T[:, 2])]; acc = ph.sum(0)
    curve = {"int8": [], "trit": []}
    for p in range(11):
        for f, fn in (("int8", int8), ("trit", trit)):
            curve[f].append(float(evaluate(fn(acc), T)[0].mean()))
        if p == 10:
            break
        _, _, conf = evaluate(acc, T)
        for m, wrong in conf:
            w = m < 0.1
            if w.any():
                acc = acc + 0.5 * ph[w].sum(0) - 0.5 * PH[sp.tri(wrong[w, 0], wrong[w, 1], wrong[w, 2])].sum(0)
    A = np.stack([g.integers(0, 1000, 250), g.integers(0, 50, 250)], 1)
    present = {(s, r) for s, r, _ in T}; A = np.array([a for a in A if tuple(a) not in present])
    sc = sp.scores(np.broadcast_to(int8(acc), (len(A), D)), 2, sp.key(2, A[:, 0], A[:, 1]))
    return N, curve, sc.max(1).tolist()


Ns = (256, 384, 512, 768, 1024)
with Pool(4) as pool:                                                         # 4 physical cores
    out = pool.map(job, [(N, b) for N in Ns for b in range(8)])
for N in Ns:
    rows = [o for o in out if o[0] == N]
    res[N] = {f: np.mean([r[1][f] for r in rows], 0).round(4).tolist() for f in ("int8", "trit")}
    res[N]["absent_p99"] = float(np.percentile(sum((r[2] for r in rows), []), 99))
    print(f"SIFT-CORE N={N:4d} int8 pass0 {res[N]['int8'][0]:.4f} -> pass10 {res[N]['int8'][-1]:.4f} | trit "
          f"{res[N]['trit'][0]:.4f} -> {res[N]['trit'][-1]:.4f} | absent p99 {res[N]['absent_p99']:.3f}", flush=True)
capN = lambda f, i: max([N for N in res if isinstance(N, int) and res[N][f][i] >= 0.99] or [0])  # noqa: E731
res["cap"] = {f"{f}_{w}": capN(f, i) for f in ("int8", "trit") for w, i in (("plain", 0), ("core", -1))}
res["verdict"] = {"PASS": res["cap"]["int8_core"] >= 384 and all(res[N]["absent_p99"] < 0.3 for N in (256, 384, 512, 768, 1024)),
                  "KILL": res["cap"]["int8_core"] <= res["cap"]["int8_plain"]}
print("SIFT-CORE-0:", res["cap"], res["verdict"], flush=True)
json.dump({str(k): v for k, v in res.items()}, open(os.path.expanduser("~/gpurun-out-0930/sift_core0.result.json"), "w"), indent=1)
