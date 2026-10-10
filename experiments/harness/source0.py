#!/usr/bin/env python3
"""SOURCE-0 (PREREGISTRATIONS-OMNITRI.md CORE): an intent-driven persistent resonator loop vs the one-shot cued resonator."""
import json, math, os, sys
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, ".")
from omni_ring.codebook import generate_rotations, generate_codebooks
from omni_ring.resonator import decode_soft
from omni_ring.crt import solve_crt
from omni_ring.types import DEFAULT_SPEC

MODS = DEFAULT_SPEC.all_moduli                   # (3, 5, 7, 13, 17)
DATA = DEFAULT_SPEC.moduli
CB = generate_codebooks(generate_rotations(DEFAULT_SPEC, seed=42), DEFAULT_SPEC)
CLAMP = 3                                        # the 13-ring
D = DEFAULT_SPEC.dimension


def encode(x):
    v = np.ones(D, complex)
    for cb, m in zip(CB, MODS):
        v = v * cb[x % m]
    return v


def source(v, intent, ticks=40, est=None, t_lo=1.0, t_hi=20.0):
    """Persistent loop: intent re-enters each tick; state has momentum; activations sharpen (softmax, rising temp)."""
    n = len(CB)
    if est is None:
        est = [cb.mean(0) for cb in CB]
    for t in range(ticks):
        est[CLAMP] = CB[CLAMP][intent]                                   # the source's intent, every tick
        temp = t_lo + (t_hi - t_lo) * t / max(1, ticks - 1)
        for i in range(n):
            if i == CLAMP:
                continue
            u = v.copy()
            for j in range(n):
                if j != i:
                    u = u * np.conj(est[j])
            a = (np.conj(CB[i]) @ u).real / D
            w = np.exp(temp * (a - a.max())); w /= w.sum()
            e = w @ CB[i]
            e = e / np.maximum(np.abs(e), 1e-9)
            est[i] = 0.5 * est[i] + 0.5 * e
            est[i] = est[i] / np.maximum(np.abs(est[i]), 1e-9)
    res = [int(np.argmax((np.conj(cb) @ e).real)) for cb, e in zip(CB, est)]
    return res, est


def decoded_x(res):
    return solve_crt(list(res[:len(DATA)]), DATA)


def trial(args):
    M, k = args
    g = np.random.default_rng(10_000 * M + k)
    while True:
        xs = g.choice(DEFAULT_SPEC.data_capacity, M, replace=False)
        r13 = [x % 13 for x in xs]
        uniq = [x for x in xs if r13.count(x % 13) == 1]
        if len(uniq) >= 2:
            break
    a, b = int(uniq[0]), int(uniq[1])
    v = sum(encode(int(x)) for x in xs)
    base = decode_soft(v, CB, iterations=15, clamp=a % 13, clamp_idx=CLAMP)
    base_ok = decoded_x(base.residues) == a
    res, est = source(v, a % 13)
    src_ok = decoded_x(res) == a
    res2, _ = source(v, b % 13, est=est)                                 # intent switch, no restart
    return M, bool(base_ok), bool(src_ok), bool(decoded_x(res2) == b)


if __name__ == "__main__":
    Ms = (8, 12, 16)
    with Pool(4) as pool:
        out = pool.map(trial, [(M, k) for M in Ms for k in range(200)])
    res = {}
    for M in Ms:
        o = [x for x in out if x[0] == M]; n = len(o)
        b = sum(x[1] for x in o) / n; s = sum(x[2] for x in o) / n; sw = sum(x[3] for x in o) / n
        pp = (sum(x[1] for x in o) + sum(x[2] for x in o)) / (2 * n)
        z = (s - b) / math.sqrt(max(pp * (1 - pp) * 2 / n, 1e-12))
        p = 0.5 * math.erfc(z / math.sqrt(2))
        res[M] = {"n": n, "baseline": b, "source": s, "switch": sw, "p_one_sided": p}
        print(f"SOURCE-0 M={M:2d}: baseline {b:.3f} | source {s:.3f} (p={p:.3g}) | intent switch {sw:.3f}", flush=True)
    r = res[12]
    res["verdict"] = {"PASS": r["source"] >= r["baseline"] + 0.10 and r["p_one_sided"] < 0.05 and r["switch"] >= 0.80,
                      "KILL": all(res[M]["source"] <= res[M]["baseline"] for M in Ms)}
    print("SOURCE-0:", res["verdict"], flush=True)
    json.dump({str(k): v for k, v in res.items()}, open(os.path.expanduser("~/gpurun-out-0930/source0.result.json"), "w"), indent=1)
