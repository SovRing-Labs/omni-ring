#!/usr/bin/env python3
"""SIEVE-0 (PIFS-RES plan (unpublished); pre-registered): algebraic power-sieve decoding of product-encoded residues."""
import json, os, sys, time
import numpy as np
sys.path.insert(0, ".")
from omni_ring.crt import solve_crt  # noqa: E402

OUT = os.path.expanduser("~/gpurun-out-0930/sieve0.result.json")
PR = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137]
SETS = [tuple(PR[i:i + 3]) for i in range(0, len(PR) - 2, 2)]


def decode_sieve(v, cbs, ms):
    res = []
    for j, m in enumerate(ms):
        p = int(np.prod([mm for k, mm in enumerate(ms) if k != j]))
        w = v ** p                                              # kills every ring but j
        r = int(np.argmax((np.conj(cbs[j]) @ w).real))          # residue of x*p mod m
        res.append((r * pow(p, -1, m)) % m)
    return solve_crt(res, ms)


def run(D, ms, rng, n=50, kappa=None, quant=None):
    rots = [rng.integers(0, m, D) * (2 * np.pi / m) for m in ms]
    cbs = [np.exp(1j * np.outer(np.arange(m), r)) for m, r in zip(ms, rots)]
    M = int(np.prod(ms)); hit = 0; ts = []
    for _ in range(n):
        x = int(rng.integers(0, M)); ang = sum(x * r for r in rots)
        if kappa is not None:
            ang = ang + rng.vonmises(0, kappa, D)
        if quant is not None:
            ang = np.round(ang / (2 * np.pi / quant)) * (2 * np.pi / quant)
        v = np.exp(1j * ang)
        t = time.time(); y = decode_sieve(v, cbs, ms); ts.append(time.time() - t); hit += int(y == x)
    return hit / n, 1000 * float(np.median(ts))


def main():
    rng = np.random.default_rng(0); out = {}
    for D in (512, 1024, 4096):
        out[str(D)] = {}
        for ms in SETS:
            M = int(np.prod(ms))
            acc, ms_ = run(D, ms, rng)
            row = {"clean": acc, "ms": ms_}
            if M in (105, 4199, 33263, 190747, 1030301, 2278811) or ms == SETS[-1]:
                for kap in (64.0, 16.0, 4.0, 1.0):
                    row[f"k{kap:g}"] = run(D, ms, rng, n=30, kappa=kap)[0]
                row["q16"] = run(D, ms, rng, n=30, quant=16)[0]; row["q256"] = run(D, ms, rng, n=30, quant=256)[0]
            out[str(D)][str(M)] = row
            print(f"D {D:5d} M {M:>10,} {ms}: clean {acc:.3f} {ms_:.3f} ms" + (f" | kappa64 {row.get('k64')} k16 {row.get('k16')} k4 {row.get('k4')} k1 {row.get('k1')} | q16 {row.get('q16')} q256 {row.get('q256')}" if 'k64' in row else ""), flush=True)
    top = str(int(np.prod(SETS[-1])))
    out["verdict"] = {"PASS": all(out[str(D)][top]["clean"] == 1.0 and out[str(D)][top]["ms"] < 1.0 for D in (512, 1024, 4096))}
    json.dump(out, open(OUT, "w"), indent=1); print("SIEVE-0", out["verdict"])


if __name__ == "__main__":
    main()
