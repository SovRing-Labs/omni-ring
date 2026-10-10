#!/usr/bin/env python3
"""CAL-1 POST-HOC DIAGNOSTIC (declared 2026-10-02 after reading the CAL-1 arm-B curves at D 512 / 1,024; NOT re-scored
against CAL-1's bar). Observation: accuracy tracks the NUMBER OF RINGS, not the range or D -- 3-ring sets decode to M 12,673
at 0.67-0.83, 4-ring sets fall to 0.26-0.81, the 5-ring set (3,5,7,11,13; M 15,015) decodes 0.000 at BOTH D 512 and D 1,024.
Capacity did not grow with D (4,199 @512 vs 2,431 @1,024). That is a convergence failure of the resonator VARIANT, not a
capacity limit of the encoding. Hypothesis: omni_ring.decode_soft (ReLU on the projection, codebook-MEAN init -- which is
~0 on every dim where the ring's phases cancel -- 200 iterations, no restarts) is a weak resonator. Diagnostic: the standard
complex resonator (Frady, Kent, Olshausen, Sommer 2020; Kymn 2025): est_i <- phasor-normalize(CB_i^T (CB_i^* (v (.) prod_{j!=i}
est_j^*))) with RANDOM unit-phasor init, no ReLU, up to 1,000 iterations, up to 5 restarts, convergence when every ring's
argmax is stable for 10 iterations. Same encoding, same moduli sets, same D, 100 states, 3 seeds.
Reported, no verdict: accuracy per set vs decode_soft's CAL-1 numbers; capacity C(D) under the standard resonator vs Kymn.
If the standard resonator reaches Kymn's range, the fix is the variant (ship it as decode_std); if it does not, the gap is
in the encoding or in how Kymn measure, and that becomes its own pre-registration.
"""
import json, os, sys, time
import numpy as np

sys.path.insert(0, ".")
from omni_ring.crt import solve_crt  # noqa: E402

OUT = os.path.expanduser("~/gpurun-out-0930/cal1_diag.result.json")
PRIMES = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41]
SMOKE = os.environ.get("SMOKE") == "1"


def field_C(D):
    return 0.13 * D * D + 24 * D - 3700


def moduli_sets():
    sets = {}
    for w in range(3, 9):
        for s in range(0, len(PRIMES) - w + 1):
            ms = tuple(PRIMES[s:s + w]); M = int(np.prod(ms))
            if M <= 2e7:
                sets.setdefault(M, ms)
    Ms = sorted(sets)
    return [(M, sets[M]) for M in (Ms[::4] if SMOKE else Ms)]


def decode_std(v, cbs, rng, iters=1000, restarts=5, stable=10):
    n = len(cbs); D = v.shape[0]
    cbc = [np.conj(c) for c in cbs]
    best = None
    for _ in range(restarts):
        est = [np.exp(1j * rng.uniform(0, 2 * np.pi, D)) for _ in cbs]
        last = None; same = 0; it = 0
        for it in range(1, iters + 1):
            for i in range(n):
                u = v.copy()
                for j in range(n):
                    if j != i:
                        u *= np.conj(est[j])
                a = cbc[i] @ u                        # complex coefficients (no ReLU, no real-part clamp)
                e = a @ cbs[i]
                est[i] = e / np.maximum(np.abs(e), 1e-12)
            res = tuple(int(np.argmax((cbc[i] @ est[i]).real)) for i in range(n))
            same = same + 1 if res == last else 0
            last = res
            if same >= stable:
                break
        # score the candidate: cosine between v and the re-encoded factorization
        venc = np.ones(D, dtype=complex)
        for i, r in enumerate(res):
            venc *= cbs[i][r]
        cos = float(np.abs(np.vdot(venc, v)) / D)
        if best is None or cos > best[1]:
            best = (res, cos, it)
        if cos >= 0.95:
            break
    return best


def run_set(D, ms, seed, n=100):
    rng = np.random.default_rng(seed)
    rots = [rng.integers(0, m, D) * (2 * np.pi / m) for m in ms]
    cbs = [np.exp(1j * np.outer(np.arange(m), r)) for m, r in zip(ms, rots)]
    M = int(np.prod(ms)); hit = 0; its = 0; t0 = time.time()
    k = n if not SMOKE else 10
    for _ in range(k):
        x = int(rng.integers(0, M))
        v = np.exp(1j * sum(x * r for r in rots))
        res, cos, it = decode_std(v, cbs, rng)
        hit += int(solve_crt(list(res), ms) == x); its += it
    return hit / k, its / k, 1000 * (time.time() - t0) / k


def main():
    t0 = time.time(); out = {}
    for D in ([512, 1024] if SMOKE else [512, 1024, 4096]):
        curve = []
        for M, ms in moduli_sets():
            r = [run_set(D, ms, 100 * s + D) for s in range(3)]
            acc = float(np.mean([a for a, _, _ in r])); its = float(np.mean([i for _, i, _ in r])); ms_ = float(np.mean([t for _, _, t in r]))
            curve.append({"M": M, "moduli": ms, "acc": acc, "iters": its, "ms": ms_})
            print(f"STD D {D} M {M:>9,} {ms}: acc {acc:.3f} iters {its:.0f} {ms_:.1f} ms", flush=True)
            if acc < 0.5 and len(curve) >= 3 and all(c["acc"] < 0.5 for c in curve[-2:]):
                break
        good = [c["M"] for c in curve if c["acc"] >= 0.95]; C = max(good) if good else 0
        out[str(D)] = {"curve": curve, "C_std": C, "C_field": field_C(D), "ratio": C / field_C(D)}
        print(f"STD D {D}: C_std {C:,} vs field {field_C(D):,.0f} (ratio {C / field_C(D):.3f})", flush=True)
    out["seconds"] = time.time() - t0; out["post_hoc"] = True
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"CAL-1 DIAG done ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
