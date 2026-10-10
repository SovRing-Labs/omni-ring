#!/usr/bin/env python3
"""CAL-2 (MAIN-0-HYPOTHESES plan (unpublished) §4, pre-registered 2026-10-02 BEFORE this harness was written):
Hrrformer's hologram ceiling vs our capacity law L1.

HRR (Plate 1995), real Gaussian atoms in H dims, binding = circular convolution (FFT), approximate inverse = involution.
beta = sum_i k_i (*) v_i over T pairs (Hrrformer eq. 1). For each t: v_hat = k_t^dagger (*) beta; recovery = argmax over
the T candidate values of cos(v_hat, v_j) equals t. T in {16, 64, 256, 1024, 4096}, H in {256, 1024}, 5 seeds.

FIELD PREDICTS: unbind SNR ~ sqrt(H/T) (Plate; Hrrformer sec. 3). In L1's form acc = Phi(a*sqrt(D/n))^(C-1) with D=H,
n=T, C=T that is a ~= 1 (ideal). OURS (xlaw.py): a per family 1.02-1.33.
PASS: L1 form fits HRR with a free a at rmse <= 0.02 AND a_HRR in [0.9, 1.4]. FAIL: rmse > 0.05.
Reported, no verdict: a_HRR; the T at which each H drops below 0.9 recovery; wall time.
"""
import json, math, os, time
import numpy as np

OUT = os.path.expanduser("~/gpurun-out-0930/cal2.result.json")
TS = [16, 64, 256, 1024, 4096]
HS = [256, 1024]
SEEDS = 5
Phi = lambda z: 0.5 * (1 + math.erf(z / math.sqrt(2)))  # noqa: E731


def cconv(a, b):
    return np.fft.irfft(np.fft.rfft(a, axis=-1) * np.fft.rfft(b, axis=-1), n=a.shape[-1], axis=-1)


def involution(a):
    return np.concatenate([a[..., :1], a[..., :0:-1]], axis=-1)


def run(H, T, seed):
    g = np.random.default_rng(seed)
    K = g.normal(0, 1 / math.sqrt(H), (T, H))
    V = g.normal(0, 1 / math.sqrt(H), (T, H))
    beta = cconv(K, V).sum(0)
    Vh = cconv(involution(K), beta[None, :])            # T x H recovered values
    Vn = V / np.linalg.norm(V, axis=1, keepdims=True)
    Vhn = Vh / np.maximum(np.linalg.norm(Vh, axis=1, keepdims=True), 1e-12)
    S = Vhn @ Vn.T                                      # cosine T x T
    return float((S.argmax(1) == np.arange(T)).mean())


def pred(a, D, C, n):
    return Phi(a * math.sqrt(D / n)) ** (C - 1)


def main():
    t0 = time.time()
    pts = []
    for H in HS:
        for T in TS:
            accs = [run(H, T, s) for s in range(SEEDS)]
            m = float(np.mean(accs))
            pts.append({"H": H, "T": T, "acc": m, "sd": float(np.std(accs)), "field_a1": pred(1.0, H, T, T)})
            print(f"H {H:5d} T {T:5d}: recovery {m:.4f} +- {np.std(accs):.4f} | field (a=1) {pred(1.0, H, T, T):.4f}", flush=True)
    inf = [p for p in pts if 0.02 < p["acc"] < 0.9999]
    grid = np.arange(0.5, 2.0, 0.001)
    errs = [math.sqrt(np.mean([(pred(a, p["H"], p["T"], p["T"]) - p["acc"]) ** 2 for p in inf])) for a in grid]
    a = float(grid[int(np.argmin(errs))]); rmse = float(min(errs))
    drop = {}
    for H in HS:
        below = [p["T"] for p in pts if p["H"] == H and p["acc"] < 0.9]
        drop[str(H)] = min(below) if below else None
    verdict = "PASS" if (rmse <= 0.02 and 0.9 <= a <= 1.4) else ("FAIL" if rmse > 0.05 else "FAIL(a out of range)" if rmse <= 0.02 else "INCONCLUSIVE")
    res = {"points": pts, "informative": len(inf), "a_hrr": a, "rmse": rmse, "T_below_0.9": drop,
           "verdict": {"PASS": verdict == "PASS", "label": verdict}, "seconds": time.time() - t0,
           "D": "H", "C": "T", "n": "T", "storage": "float64 HRR"}
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"CAL-2 a_HRR = {a:.3f} rmse {rmse:.4f} over {len(inf)} informative points; T below 0.9: {drop}; verdict {verdict} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
