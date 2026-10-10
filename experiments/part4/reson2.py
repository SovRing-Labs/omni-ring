#!/usr/bin/env python3
"""RESON-2 (RESON-2 plan (unpublished); pre-registered before this file existed): four resonator arms on the same residue sets.
SOFT (ours) · STD (published, verbatim) · STD-CHAOS (IFS chaos-game schedule) · OSC (coupled oscillators, injection-locked).
IFS diagnostics: contraction ratio to truth, Lyapunov proxy, limit-cycle rate. See the plan for FIELD PREDICTS and bars."""
import json, os, sys, time
import numpy as np

sys.path.insert(0, ".")
from omni_ring.crt import solve_crt  # noqa: E402
from omni_ring.resonator import decode_soft  # noqa: E402

OUT = os.path.expanduser("~/gpurun-out-0930/reson2.result.json")
SMOKE = os.environ.get("SMOKE") == "1"
ARMS = os.environ.get("ARMS", "SOFT,STD,STD-CHAOS,OSC").split(",")
DS = [512, 1024] if SMOKE else [512, 1024, 4096]
SETS = [(3, 5, 7), (3, 5, 7, 11), (3, 5, 7, 11, 13), (3, 5, 7, 11, 13, 17), (3, 5, 7, 11, 13, 17, 19)]
NSTATE = 6 if SMOKE else 50
SEEDS = 1 if SMOKE else 3


def field_C(D):
    return 0.13 * D * D + 24 * D - 3700


def make(ms, D, rng):
    rots = [rng.integers(0, m, D) * (2 * np.pi / m) for m in ms]
    cbs = [np.exp(1j * np.outer(np.arange(m), r)).astype(np.complex128) for m, r in zip(ms, rots)]
    return rots, cbs


def dist_to_truth(est, cbs, true_res):
    return float(np.mean([1 - abs(np.vdot(cbs[i][r], e)) / len(e) for i, (e, r) in enumerate(zip(est, true_res))]))


def run_std(v, cbs, rng, chaos=False, iters=1000, restarts=5, true_res=None):
    n, D = len(cbs), v.shape[0]
    cbc = [np.conj(c) for c in cbs]
    best, ratios, cycle = None, [], False
    for _ in range(restarts):
        est = [np.exp(1j * rng.uniform(0, 2 * np.pi, D)) for _ in cbs]
        hist = []
        for it in range(1, iters + 1):
            prev = [e.copy() for e in est]
            d0 = dist_to_truth(est, cbs, true_res) if true_res is not None else None
            order = rng.permutation(n) if chaos else range(n)
            for i in order:
                u = v.copy()
                for j in range(n):
                    if j != i:
                        u *= np.conj(est[j])
                e = (cbc[i] @ u) @ cbs[i]
                est[i] = e / np.maximum(np.abs(e), 1e-12)
            if true_res is not None and d0 and d0 > 1e-9:
                ratios.append(dist_to_truth(est, cbs, true_res) / d0)
            cos = min(abs(np.vdot(p, e)) / D for p, e in zip(prev, est))
            res = tuple(int(np.argmax((cbc[i] @ est[i]).real)) for i in range(n))
            if cos >= 0.95:
                break
            if res in hist[-50:]:
                cycle = True
            hist.append(res)
        res = tuple(int(np.argmax((cbc[i] @ est[i]).real)) for i in range(n))
        venc = np.ones(D, dtype=complex)
        for i, r in enumerate(res):
            venc *= cbs[i][r]
        score = float(abs(np.vdot(venc, v)) / D)
        if best is None or score > best[1]:
            best = (res, score, it)
        if score >= 0.95:
            break
    return best[0], ratios, cycle, best[2]


def run_osc(v, cbs, rng, steps=2000, dt=0.1, true_res=None):
    n, D = len(cbs), v.shape[0]
    ms = [len(c) for c in cbs]
    phi = [np.angle(c[1]) for c in cbs]                      # per-dim phase step of each ring (legal lattice offset)
    th = [rng.uniform(0, 2 * np.pi, D) for _ in cbs]
    cbc = [np.conj(c) for c in cbs]
    ratios, cycle, hist = [], False, []
    for t in range(steps):
        kh = 2.0 * t / steps; beta = 2.0 + 18.0 * t / steps
        E = [np.exp(1j * x) for x in th]
        d0 = dist_to_truth(E, cbs, true_res) if true_res is not None else None
        for i in range(n):
            u = v.copy()
            for j in range(n):
                if j != i:
                    u *= np.conj(E[j])
            a = (cbc[i] @ E[i]).real / D                         # codebook scores
            w = np.exp(beta * (a - a.max())); w /= w.sum()
            psi = np.angle(w @ cbs[i])                           # mean field (lateral competition)
            dth = np.sin(np.angle(u) - th[i]) + np.sin(psi - th[i]) - kh * np.sin(ms[i] * (th[i] - phi[i] * 0))
            th[i] = (th[i] + dt * dth) % (2 * np.pi)
        E = [np.exp(1j * x) for x in th]
        if true_res is not None and d0 and d0 > 1e-9:
            ratios.append(dist_to_truth(E, cbs, true_res) / d0)
        res = tuple(int(np.argmax((cbc[i] @ E[i]).real)) for i in range(n))
        order = min(abs(np.mean(E[i] * np.conj(cbs[i][res[i]]))) for i in range(n))
        if order >= 0.95 and t > 50:
            break
        if t % 25 == 0:
            if res in hist[-20:] and t > 500:
                cycle = True
            hist.append(res)
    return res, ratios, cycle, t


def main():
    t0 = time.time(); out = {"arms": {}, "D": DS, "sets": [int(np.prod(s)) for s in SETS]}
    for arm in ARMS:
        out["arms"][arm] = {}
        for D in DS:
            curve = []
            for ms in SETS:
                M = int(np.prod(ms)); accs, cyc, lyap = [], [], []
                for s in range(SEEDS):
                    rng = np.random.default_rng(1000 * s + D)
                    rots, cbs = make(ms, D, rng)
                    hit = 0
                    for _ in range(NSTATE):
                        x = int(rng.integers(0, M)); tr = [x % m for m in ms]
                        v = np.exp(1j * sum(x * r for r in rots))
                        if arm == "SOFT":
                            r = decode_soft(v, cbs, iterations=200, tolerance=1e-4); res, ratios, c = tuple(r.residues), [], False
                        elif arm == "STD":
                            res, ratios, c, _ = run_std(v, cbs, rng, chaos=False, true_res=tr)
                        elif arm == "STD-CHAOS":
                            res, ratios, c, _ = run_std(v, cbs, rng, chaos=True, true_res=tr)
                        else:
                            res, ratios, c, _ = run_osc(v, cbs, rng, true_res=tr)
                        hit += int(solve_crt(list(res), ms) == x); cyc.append(c)
                        if ratios:
                            lyap.append(float(np.exp(np.mean(np.log(np.clip(ratios, 1e-6, 1e6))))))
                    accs.append(hit / NSTATE)
                acc = float(np.mean(accs))
                curve.append({"M": M, "rings": len(ms), "acc": acc, "cycle_rate": float(np.mean(cyc)) if cyc else None,
                              "contraction_gm": float(np.mean(lyap)) if lyap else None})
                print(f"{arm:9s} D {D:5d} M {M:>10,} ({len(ms)} rings): acc {acc:.3f}  cycles {np.mean(cyc) if cyc else 0:.2f}  contraction {np.mean(lyap) if lyap else float('nan'):.3f}", flush=True)
                if acc < 0.3 and len(curve) >= 2:
                    break
            good = [c["M"] for c in curve if c["acc"] >= 0.95]; C = max(good) if good else 0
            out["arms"][arm][str(D)] = {"curve": curve, "C": C, "C_field": field_C(D), "ratio": C / field_C(D)}
            print(f"{arm:9s} D {D}: C {C:,} vs field {field_C(D):,.0f} ratio {C / field_C(D):.3f}", flush=True)
    v = {}
    if "STD" in out["arms"]:
        rs = [out["arms"]["STD"][str(D)]["ratio"] for D in DS]
        v["RESON-1_STD"] = "PASS" if min(rs) >= 0.8 else ("KILL" if min(rs) < 0.5 else "FAIL")
    if "OSC" in out["arms"] and "STD" in out["arms"]:
        v["RESON-2_OSC_ge_STD"] = all(out["arms"]["OSC"][str(D)]["C"] >= out["arms"]["STD"][str(D)]["C"] for D in DS)
    if "STD-CHAOS" in out["arms"] and "STD" in out["arms"]:
        v["CHAOS_C_ge_STD"] = all(out["arms"]["STD-CHAOS"][str(D)]["C"] >= out["arms"]["STD"][str(D)]["C"] for D in DS)
    out["verdict"] = v; out["seconds"] = time.time() - t0; out["smoke"] = SMOKE
    json.dump(out, open(OUT, "w"), indent=1)
    print("RESON-2", v, f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
