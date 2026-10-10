#!/usr/bin/env python3
"""PIFS-RES-1 (PIFS-RES plan (unpublished)): uncapped ladder, partition shapes on the kernel, noisy slots."""
import json, os, sys, time
import numpy as np
sys.path.insert(0, ".")
from omni_ring.route import part, encode_prod, _LAT  # noqa: E402
from omni_ring.crt import solve_crt  # noqa: E402
OUT = os.path.expanduser("~/gpurun-out-0930/pifs_res1.result.json")
PR = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137]
SETS = [tuple(PR[i:i + 3]) for i in range(0, len(PR) - 2, 4)]
DS = [int(x) for x in os.environ.get("DS", "512,1024").split(",")]
ARMS = os.environ.get("ARMS", "CONTIG,STRIDE,TRI,HEX,RANDOM").split(",")
NSTATE, SEEDS, KAPPA = 30, 2, 16.0


def main():
    out = {"ladder": [int(np.prod(s)) for s in SETS], "arms": {}}
    for arm in ARMS:
        out["arms"][arm] = {}
        for D in DS:
            curve = []
            for ms in SETS:
                M = int(np.prod(ms)); per_seed = []; tt = []
                for s in range(SEEDS):
                    rng = np.random.default_rng(100 * s + D); _LAT.clear()
                    rots = [rng.integers(0, m, D) * (2 * np.pi / m) for m in ms]
                    cbs = [np.exp(1j * np.outer(np.arange(m), r)) for m, r in zip(ms, rots)]
                    hit = 0
                    for _ in range(NSTATE):
                        x = int(rng.integers(0, M)); ang = sum(x * r for r in rots) + rng.vonmises(0, KAPPA, D); v = np.exp(1j * ang)
                        t1 = time.time(); res, ag = part(v, cbs, ms, partition=arm, rng=np.random.default_rng(7)); tt.append(time.time() - t1)
                        hit += int(solve_crt(list(res), ms) == x)
                    per_seed.append(hit / NSTATE)
                acc = float(np.mean(per_seed)); curve.append({"M": M, "acc": acc, "per_seed": per_seed, "ms": 1000 * float(np.median(tt))})
                print(f"{arm:6s} D {D:5d} M {M:>10,}: acc {acc:.3f} seeds {per_seed} {1000 * np.median(tt):.0f} ms", flush=True)
            good = [c["M"] for c in curve if c["acc"] >= 0.95]; C = max(good) if good else 0
            out["arms"][arm][str(D)] = {"curve": curve, "C": C, "seed_spread": float(np.mean([abs(c["per_seed"][0] - c["per_seed"][1]) for c in curve]))}
            print(f"{arm:6s} D {D}: C {C:,} spread {out['arms'][arm][str(D)]['seed_spread']:.2f}", flush=True)
    v = {}
    for D in DS:
        C = {a: out["arms"][a][str(D)]["C"] for a in ARMS}
        v[str(D)] = {"C": C, "geom_ge_contig": max(C.get("TRI", 0), C.get("HEX", 0)) >= C.get("CONTIG", 0)}
    out["verdict"] = v; json.dump(out, open(OUT, "w"), indent=1); print("PIFS-RES-1", v)


if __name__ == "__main__":
    main()
