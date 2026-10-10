#!/usr/bin/env python3
"""RESON-1b (declared 2026-10-02 BEFORE running; corrects CAL-1 arm B's ladder): Kymn's capacity curve is measured with THREE
rings of growing consecutive primes (C(512) = 42.7k ~ 31*37*41; the paper: 'K consecutive primes', K fixed at 3 in the
capacity figure), NOT by adding rings — resonator capacity falls steeply with the number of factors (Kent et al. 2020), which
is the ring-count failure CAL-1 saw. Ladder: 3 consecutive primes from p_k, p_k = 3, 5, ..., up to (127,131,137) = 2.28M.
Arms: SOFT (ours, decode_soft 200 it) and STD-SUPER (phasor resonator, superposition init + 0.3 noise, 300 it, 5 restarts,
async sweep). D in {512, 1024, 4096}; 30 states per set; 2 seeds; stop a D when acc < 0.5 twice.
FIELD PREDICTS: C(D) ~ 0.13 D^2 + 24 D - 3,700 (42.7k / 157k / 2.28M).
PASS (RESON-1): STD-SUPER C(D) >= 0.8x field at every D. KILL: < 0.5x. SOFT reported.
"""
import json, os, sys, time
import numpy as np

sys.path.insert(0, ".")
from omni_ring.crt import solve_crt  # noqa: E402
from omni_ring.resonator import decode_soft  # noqa: E402

OUT = os.path.expanduser("~/gpurun-out-0930/reson1b.result.json")
PR = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151]
SETS = [tuple(PR[i:i + 3]) for i in range(0, len(PR) - 2, 2)]
DS = [512, 1024, 4096]
NSTATE, SEEDS = 30, 2


def field_C(D):
    return 0.13 * D * D + 24 * D - 3700


def std_super(v, cbs, rng, iters=300, restarts=5):
    n, D = len(cbs), v.shape[0]; cbc = [np.conj(c) for c in cbs]; best = None
    for _ in range(restarts):
        est = []
        for c in cbs:
            s = c.sum(0) + 0.3 * np.exp(1j * rng.uniform(0, 2 * np.pi, D)); est.append(s / np.maximum(np.abs(s), 1e-12))
        for it in range(iters):
            prev = [e.copy() for e in est]
            for i in range(n):
                u = v.copy()
                for j in range(n):
                    if j != i:
                        u *= np.conj(est[j])
                e = (cbc[i] @ u) @ cbs[i]; est[i] = e / np.maximum(np.abs(e), 1e-12)
            if min(abs(np.vdot(p, e)) / D for p, e in zip(prev, est)) >= 0.999 and it > 5:
                break
        res = tuple(int(np.argmax((cbc[i] @ est[i]).real)) for i in range(n))
        venc = np.ones(D, dtype=complex)
        for i, r in enumerate(res):
            venc *= cbs[i][r]
        sc = abs(np.vdot(venc, v)) / D
        if best is None or sc > best[1]:
            best = (res, sc)
        if sc >= 0.95:
            break
    return best[0]


def main():
    t0 = time.time(); out = {"ladder": [int(np.prod(s)) for s in SETS], "arms": {}}
    for arm in ("SOFT", "STD-SUPER"):
        out["arms"][arm] = {}
        for D in DS:
            curve, low = [], 0
            for ms in SETS:
                M = int(np.prod(ms)); accs = []
                for s in range(SEEDS):
                    rng = np.random.default_rng(100 * s + D)
                    rots = [rng.integers(0, m, D) * (2 * np.pi / m) for m in ms]
                    cbs = [np.exp(1j * np.outer(np.arange(m), r)) for m, r in zip(ms, rots)]
                    hit = 0
                    for _ in range(NSTATE):
                        x = int(rng.integers(0, M)); v = np.exp(1j * sum(x * r for r in rots))
                        res = tuple(decode_soft(v, cbs, iterations=200, tolerance=1e-4).residues) if arm == "SOFT" else std_super(v, cbs, rng)
                        hit += int(solve_crt(list(res), ms) == x)
                    accs.append(hit / NSTATE)
                acc = float(np.mean(accs)); curve.append({"M": M, "moduli": ms, "acc": acc})
                print(f"{arm:9s} D {D:5d} M {M:>10,} {ms}: acc {acc:.3f}", flush=True)
                low = low + 1 if acc < 0.5 else 0
                if low >= 2:
                    break
            good = [c["M"] for c in curve if c["acc"] >= 0.95]; C = max(good) if good else 0
            out["arms"][arm][str(D)] = {"curve": curve, "C": C, "C_field": field_C(D), "ratio": C / field_C(D)}
            print(f"{arm:9s} D {D}: C {C:,} vs field {field_C(D):,.0f} ratio {C / field_C(D):.3f}", flush=True)
    rs = [out["arms"]["STD-SUPER"][str(D)]["ratio"] for D in DS]
    out["verdict"] = {"RESON-1": "PASS" if min(rs) >= 0.8 else ("KILL" if min(rs) < 0.5 else "FAIL"), "ratios": rs}
    out["seconds"] = time.time() - t0
    json.dump(out, open(OUT, "w"), indent=1)
    print("RESON-1b", out["verdict"], f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
