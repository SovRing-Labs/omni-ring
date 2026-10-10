#!/usr/bin/env python3
"""SIMPLEX-0 (PREREGISTRATIONS-OMNITRI.md): random vs regular-simplex relation codebook in triangle buckets."""
import json, os, sys
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, ".")
from omni_ring import solids as SO

NR, NE = 16, 1000
SIMPLEX = SO.simplex(NR, 5120).astype(np.int64)
D = SIMPLEX.shape[1]                                       # 5115
g0 = np.random.default_rng(100)
RANDOM = g0.integers(0, 16, (NR, D))
ENT = g0.integers(0, 16, (NE, D))
P = [g0.permutation(D) for _ in range(3)]
PH = np.exp(2j * np.pi * np.arange(16) / 16).astype(np.complex64)


def run(args):
    arm, fill, b = args
    R = SIMPLEX if arm == "simplex" else RANDOM
    rS, rR, rO = ENT[:, P[0]], R[:, P[1]], ENT[:, P[2]]
    g = np.random.default_rng(1000 * fill + b)                   # same triangles in both arms
    T = np.stack([g.integers(0, NE, fill), g.integers(0, NR, fill), g.integers(0, NE, fill)], 1)
    acc = PH[(rS[T[:, 0]] + rR[T[:, 1]] + rO[T[:, 2]]) % 16].sum(0)
    s = max(np.abs(acc.real).max(), np.abs(acc.imag).max()) / 127
    acc = ((np.rint(acc.real / s) + 1j * np.rint(acc.imag / s)) * s).astype(np.complex64)
    cr, ci = np.cos(np.pi / 8 * rR).astype(np.float32), np.sin(np.pi / 8 * rR).astype(np.float32)
    def score(si, oi):
        u = acc[None] * np.conj(PH[(rS[si] + rO[oi]) % 16])
        return (u.real @ cr.T + u.imag @ ci.T) / D
    sc = score(T[:, 0], T[:, 2])
    tru = sc[np.arange(fill), T[:, 1]].copy(); sc2 = sc.copy(); sc2[np.arange(fill), T[:, 1]] = -9
    margin = tru - sc2.max(1)
    present = {(a, c) for a, _, c in T}
    A = np.array([p for p in g.integers(0, NE, (400, 2)) if tuple(p) not in present][:250])
    ab = score(A[:, 0], A[:, 1]).max(1) / np.median(tru)
    return arm, fill, float((sc.argmax(1) == T[:, 1]).mean()), margin.tolist(), ab.tolist()


if __name__ == "__main__":
    fills = (256, 512, 1024, 2048, 4096)
    with Pool(4) as pool:
        out = pool.map(run, [(a, f, b) for a in ("random", "simplex") for f in fills for b in range(8)])
    res = {"D": D, "min_sep": {"random": SO.min_separation(RANDOM.astype(np.uint8)), "simplex": SO.min_separation(SIMPLEX.astype(np.uint8))}}
    for f in fills:
        row = {}
        for a in ("random", "simplex"):
            o = [x for x in out if x[0] == a and x[1] == f]
            m = np.concatenate([x[3] for x in o]); ab = np.concatenate([x[4] for x in o])
            row[a] = {"recovery": float(np.mean([x[2] for x in o])), "margin_mean": float(m.mean()),
                      "margin_p5": float(np.percentile(m, 5)), "absent_ratio_p99": float(np.percentile(ab, 99))}
        res[f] = row
        r, s = row["random"], row["simplex"]
        print(f"SIMPLEX-0 fill {f:4d} | recovery random {r['recovery']:.4f} simplex {s['recovery']:.4f} | margin p5 "
              f"{r['margin_p5']:.3f} vs {s['margin_p5']:.3f} | absent/present p99 {r['absent_ratio_p99']:.3f} vs {s['absent_ratio_p99']:.3f}", flush=True)
    rec = [(res[f]["simplex"]["recovery"] - res[f]["random"]["recovery"], res[f]["random"]["recovery"]) for f in fills]
    mg = [res[f]["simplex"]["margin_p5"] / res[f]["random"]["margin_p5"] - 1 if res[f]["random"]["margin_p5"] > 0 else None for f in fills]
    res["verdict"] = {"PASS": any(d >= 0.01 and r0 < 0.99 for d, r0 in rec) or all(m is not None and m >= 0.10 for m in mg),
                      "KILL": all(abs(d) <= 0.005 for d, _ in rec) and all(m is not None and abs(m) <= 0.05 for m in mg),
                      "margin_p5_gain": mg}
    print("SIMPLEX-0:", res["min_sep"], res["verdict"], flush=True)
    json.dump({str(k): v for k, v in res.items()}, open(os.path.expanduser("~/gpurun-out-0930/simplex0.result.json"), "w"), indent=1)
