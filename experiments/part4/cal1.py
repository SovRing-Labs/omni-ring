#!/usr/bin/env python3
"""CAL-1 (MAIN-0-HYPOTHESES plan (unpublished) §4, pre-registered 2026-10-02 BEFORE this harness was written): Residue HDC capacity
on the omniring -- reproduce the field (Kymn, Kleyko, Frady, Kanerva, Sommer, Olshausen 2025, arXiv 2311.04872).

Ground truth from reading omni_ring (2026-10-02): the omniring has TWO encodings.
  (A) encode_direct: ADDITIVE superposition of one random K16 codeword per ring (3,5,7,13 + check 17), phases snapped to
      16 levels; decoded by direct_probe (sum of moduli = 45 inner products) + CRT; the 17 ring flags a slip.
  (B) encode (chain.py): Kymn's encoding exactly -- Hadamard PRODUCT of per-ring phasors exp(i x r_m) with r_m in 2*pi*k/m
      over the full D; decoded by the resonator decode_soft (ReLU soft power iteration, codebook-mean init).
FIELD PREDICTS: (A) exact at kappa >= 16, > 0.99 at von Mises kappa = 1 for D >= 1,024 (codebook decode is exact within range).
(B) Kymn Fig. 2 resonator effective range C(D) ~ 0.13 D^2 + 24 D - 3,700  (42.7k @512, 157k @1,024, 2.28M @4,096), accuracy
'almost perfect up to a range, then rapid decay', convergence cos >= 0.95.
PASS: (A) as predicted; (B) our C(D) >= 0.8 x theirs at every D.  KILL: (B) < 0.5 x at any D.
Reported, no verdict: ms/decode both arms; check-ring catch rate and false-flag rate under kappa = 1.
Verify: python3 cal1.py ; result ~/gpurun-out-0930/cal1.result.json
"""
import json, math, os, sys, time
import numpy as np

sys.path.insert(0, ".")
from omni_ring.crt import solve_crt, direct_probe  # noqa: E402
from omni_ring.resonator import decode_soft  # noqa: E402

OUT = os.path.expanduser("~/gpurun-out-0930/cal1.result.json")
SMOKE = os.environ.get("SMOKE") == "1"
K = 16
PRIMES = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41]


def field_C(D):
    return 0.13 * D * D + 24 * D - 3700


# ---------------- arm A: additive K16 superposition + direct probe (omniring encode_direct) ----------------
def arm_a(D, kappa, rng):
    mods = (3, 5, 7, 13, 17)
    cbs = [np.exp(1j * rng.integers(0, K, (m, D)) * (2 * np.pi / K)) for m in mods]
    cbc = [np.conj(c) for c in cbs]
    M = 3 * 5 * 7 * 13
    xs = range(M) if not SMOKE else range(0, M, 25)
    ok = flagged_clean = caught = tot = 0
    t0 = time.time()

    def enc(res):
        v = sum(cb[r] for cb, r in zip(cbs, res))
        ang = np.angle(v)
        if kappa is not None:
            ang = ang + rng.vonmises(0, kappa, D)
        ph = np.round(ang * K / (2 * np.pi)).astype(int) % K
        return np.exp(1j * ph * 2 * np.pi / K)

    for x in xs:
        res = [x % m for m in mods]
        _, xd = direct_probe(enc(res), cbs, mods, cbc)
        ok += int(xd == x); flagged_clean += int(xd >= M); tot += 1
        i = int(rng.integers(0, 5)); bad = list(res); bad[i] = (bad[i] + int(rng.integers(1, mods[i]))) % mods[i]
        _, xb = direct_probe(enc(bad), cbs, mods, cbc)
        caught += int(xb >= M or xb != x)
    return {"exact": ok / tot, "false_flag": flagged_clean / tot, "corruption_caught": caught / tot,
            "ms_per_decode": 1000 * (time.time() - t0) / (2 * tot)}


# ---------------- arm B: Kymn product encoding + resonator (omniring encode / decode_soft) ----------------
def moduli_sets():
    sets = {}
    for w in range(3, 9):
        for s in range(0, len(PRIMES) - w + 1):
            ms = tuple(PRIMES[s:s + w]); M = int(np.prod(ms))
            if M <= 2e7:
                sets.setdefault(M, ms)
    Ms = sorted(sets)
    if SMOKE:
        Ms = Ms[::4]
    return [(M, sets[M]) for M in Ms]


def arm_b(D, ms, rng, n=100, iters=200):
    rots = [rng.integers(0, m, D) * (2 * np.pi / m) for m in ms]
    cbs = [np.exp(1j * np.outer(np.arange(m), r)) for m, r in zip(ms, rots)]
    M = int(np.prod(ms)); hit = 0; its = 0; t0 = time.time()
    for _ in range(n if not SMOKE else 10):
        x = int(rng.integers(0, M))
        v = np.exp(1j * sum(x * r for r in rots))
        r = decode_soft(v, cbs, iterations=iters, tolerance=1e-4)
        hit += int(solve_crt(r.residues, ms) == x); its += r.iterations
    k = n if not SMOKE else 10
    return hit / k, its / k, 1000 * (time.time() - t0) / k


def main():
    t0 = time.time(); rng = np.random.default_rng(0)
    A = {}
    for D in ([576, 1024] if SMOKE else [576, 1024, 5120]):
        for kap in (None, 16.0, 1.0):
            A[f"D{D}_k{kap}"] = arm_a(D, kap, rng)
            print(f"A D {D} kappa {kap}: {A[f'D{D}_k{kap}']}", flush=True)
    B = {}; Ccur = {}
    for D in ([512, 1024] if SMOKE else [512, 1024, 4096]):
        curve = []
        for M, ms in moduli_sets():
            accs = [arm_b(D, ms, np.random.default_rng(100 * s + D), n=100)[0] for s in range(3)]
            acc = float(np.mean(accs)); curve.append({"M": M, "moduli": ms, "acc": acc})
            print(f"B D {D} M {M:>9,} {ms}: acc {acc:.3f}", flush=True)
            if acc < 0.5 and len(curve) >= 3 and all(c["acc"] < 0.5 for c in curve[-2:]):
                break
        good = [c["M"] for c in curve if c["acc"] >= 0.95]
        C = max(good) if good else 0
        B[str(D)] = curve; Ccur[str(D)] = {"C_ours": C, "C_field": field_C(D), "ratio": C / field_C(D)}
        print(f"B D {D}: C_ours {C:,} vs field {field_C(D):,.0f} (ratio {C / field_C(D):.3f})", flush=True)
    a_pass = all(v["exact"] >= 0.999 for k, v in A.items() if "kNone" in k or "k16" in k) and \
             all(v["exact"] > 0.99 for k, v in A.items() if "k1.0" in k and ("D1024" in k or "D5120" in k))
    ratios = [v["ratio"] for v in Ccur.values()]
    b_label = "PASS" if min(ratios) >= 0.8 else ("KILL" if min(ratios) < 0.5 else "FAIL")
    res = {"A": A, "A_pass": a_pass, "B": B, "capacity": Ccur, "B_label": b_label,
           "verdict": {"PASS": a_pass and b_label == "PASS", "A": a_pass, "B": b_label},
           "seconds": time.time() - t0, "smoke": SMOKE, "resonator": "decode_soft ReLU, codebook-mean init, 200 it, tol 1e-4"}
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"CAL-1 A {'PASS' if a_pass else 'FAIL'} | B {b_label} {Ccur} | {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
