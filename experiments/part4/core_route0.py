#!/usr/bin/env python3
"""CORE-ROUTE-0 (PIFS-RES plan (unpublished)): the ladder on the largest recorded size, mixed stream."""
import json, os, sys, time
import numpy as np
sys.path.insert(0, ".")
from omni_ring.route import route, encode_prod  # noqa: E402

OUT = os.path.expanduser("~/gpurun-out-0930/core_route0.result.json")
MS = (113, 127, 131); M = int(np.prod(MS))
DS = [int(x) for x in os.environ.get("DS", "1024").split(",")]
N = {"CLEAN": 50, "NOISY": 50, "QUANT": 50, "SUPER": 30}


def main():
    out = {"M": M, "moduli": MS}
    for D in DS:
        rng = np.random.default_rng(D)
        rots = [rng.integers(0, m, D) * (2 * np.pi / m) for m in MS]
        cbs = [np.exp(1j * np.outer(np.arange(m), r)) for m, r in zip(MS, rots)]
        out[str(D)] = {}
        for cls, n in N.items():
            hits = 0; rungs = []; ms_ = []; recall = []
            for _ in range(n):
                if cls == "SUPER":
                    xs = [int(x) for x in rng.choice(M, 4, replace=False)]
                    v = sum(encode_prod(x, rots) for x in xs)
                    r = route(v, cbs, MS, superposed_hint=True)
                    got = set(r["x"] or []) if isinstance(r["x"], list) else set()
                    recall.append(len(got & set(xs)) / 4); hits += int(len(got & set(xs)) >= 3)
                else:
                    x = int(rng.integers(0, M)); ang = sum(x * rr for rr in rots)
                    if cls == "NOISY":
                        ang = ang + rng.vonmises(0, 16.0, D)
                    if cls == "QUANT":
                        ang = np.round(ang / (2 * np.pi / 16)) * (2 * np.pi / 16)
                    v = np.exp(1j * ang)
                    r = route(v, cbs, MS)
                    hits += int(r["x"] == x)
                rungs.append(r["rung"]); ms_.append(r["ms"])
            row = {"acc": hits / n, "rung_hist": {str(k): int(sum(1 for q in rungs if q == k)) for k in (1, 2, 3, 4)},
                   "ms_p50": float(np.median(ms_)), "ms_max": float(np.max(ms_))}
            if recall:
                row["recall_mean"] = float(np.mean(recall))
            out[str(D)][cls] = row
            print(f"D {D} {cls:5s}: acc {hits / n:.3f} rungs {row['rung_hist']} ms p50 {np.median(ms_):.1f} max {np.max(ms_):.0f}" + (f" recall {np.mean(recall):.2f}" if recall else ""), flush=True)
    d = str(DS[0]); r = out[d]
    out["verdict"] = {"PASS": r["CLEAN"]["acc"] == 1.0 and r["CLEAN"]["ms_p50"] < 10 and r["NOISY"]["acc"] >= 0.95 and r["QUANT"]["acc"] >= 0.95
                      and r["NOISY"]["ms_max"] < 2000 and r["SUPER"].get("recall_mean", 0) >= 0.9,
                      "KILL": r["NOISY"]["acc"] < 0.8 or r["QUANT"]["acc"] < 0.8}
    json.dump(out, open(OUT, "w"), indent=1); print("CORE-ROUTE-0", out["verdict"])


if __name__ == "__main__":
    main()
