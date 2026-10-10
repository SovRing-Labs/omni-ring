#!/usr/bin/env python3
"""CORE-FIELD-0: CF-0 (bump persists / regenerates), CF-1 (residue rings read the bump exactly), CF-2 (source moves it;
HOLD / RELEASE). Pre-registered in CORE-FIELD plan (unpublished) (2026-10-02) before this harness existed.
Surface: 48 x 48 hex-neighbourhood grid wrapped on a torus; one update rule per cell (local excitation, global inhibition,
Burak–Fiete 2009 form); bump centre read by circular mean; residue readout = CAL-1 arm A encoding of (x, y) per axis."""
import json, os, sys
import numpy as np
sys.path.insert(0, ".")
from omni_ring.crt import direct_probe  # noqa: E402

OUT = os.path.expanduser("~/gpurun-out-0930/cf2c.result.json")
N = 48; MODS = (3, 5, 7, 13, 17); K = 16; D = 1024
rng = np.random.default_rng(0)


def hex_neighbours():
    """6 neighbours on an offset-row hex grid, wrapped (torus)."""
    nb = np.zeros((N, N, 6, 2), dtype=int)
    for y in range(N):
        for x in range(N):
            odd = y % 2
            cand = [(x - 1, y), (x + 1, y), (x - 1 + odd, y - 1), (x + odd, y - 1), (x - 1 + odd, y + 1), (x + odd, y + 1)]
            nb[y, x] = [((cx % N), (cy % N)) for cx, cy in cand]
    return nb


NB = hex_neighbours()


def step(a, inp=None, w_ex=0.5, w_in=2.5, b0=0.05, noise=0.0, tau=0.1):
    """CF-0b: rectified rate model with saturation (Burak-Fiete form): local excitation, global inhibition, tonic drive."""
    s = np.zeros_like(a)
    for k in range(6):
        s += a[NB[:, :, k, 1], NB[:, :, k, 0]]
    drive = w_ex * s - w_in * a.sum() / (N * N) * N * N / (N * N) * 6 - w_in * a.mean() * 6 * 0 + b0 + (inp if inp is not None else 0.0)
    drive = w_ex * s - w_in * 6 * a.mean() + b0 + (inp if inp is not None else 0.0)
    if noise:
        drive = drive + noise * rng.normal(size=a.shape)
    return np.clip(a + tau * (-a + drive), 0.0, 1.0)


def centre(a):
    th_x = 2 * np.pi * np.arange(N) / N; th_y = 2 * np.pi * np.arange(N) / N
    wx = a.sum(0); wy = a.sum(1)
    cx = (np.angle((wx * np.exp(1j * th_x)).sum()) % (2 * np.pi)) * N / (2 * np.pi)
    cy = (np.angle((wy * np.exp(1j * th_y)).sum()) % (2 * np.pi)) * N / (2 * np.pi)
    return cx, cy


def bump_at(cx, cy, r=3.0):
    yy, xx = np.mgrid[0:N, 0:N]
    dx = (xx - cx + N / 2) % N - N / 2; dy = (yy - cy + N / 2) % N - N / 2
    return np.exp(-(dx ** 2 + dy ** 2) / (2 * r ** 2))


def wrap_dist(p, q):
    d = np.abs(np.array(p) - np.array(q)); d = np.minimum(d, N - d); return float(np.hypot(*d))


# residue readout (CAL-1 arm A): one codebook set per axis
CB = [[np.exp(1j * rng.integers(0, K, (m, D)) * (2 * np.pi / K)) for m in MODS] for _ in range(2)]
CBC = [[np.conj(c) for c in cbs] for cbs in CB]


def encode_axis(cbs, v):
    res = [v % m for m in MODS]; s = sum(cb[r] for cb, r in zip(cbs, res)); ph = np.round(np.angle(s) * K / (2 * np.pi)).astype(int) % K
    return np.exp(1j * ph * 2 * np.pi / K)


def read_axis(ax, v):
    _, x = direct_probe(encode_axis(CB[ax], v), CB[ax], MODS, CBC[ax]); return x



def point(push, tau, w_in):
    kw = dict(tau=tau, w_in=w_in)
    hits = 0
    for _ in range(20):
        a = bump_at(24, 24); target = np.array([24.0, 24.0])
        for t in range(30):
            a = step(a, **kw)
        for t in range(100):
            vx, vy = rng.normal(0, 0.3, 2); target += (vx, vy)
            cx, cy = centre(a); a = step(a, inp=push * bump_at(cx + vx, cy + vy), **kw)
        hits += int(wrap_dist(centre(a), target % N) <= 2.0)
    a = bump_at(10, 10)
    for t in range(30):
        a = step(a, **kw)
    held = 0
    for t in range(300):
        a = step(a, inp=push * bump_at(10, 10) + 0.4 * push * bump_at(30, 30), **kw); held += int(wrap_dist(centre(a), (10, 10)) <= 1.0)
    rel = None
    for t in range(200):
        a = step(a, inp=push * bump_at(30, 30), **kw)
        if wrap_dist(centre(a), (30, 30)) <= 2.0:
            rel = t; break
    return hits / 20, held / 300, rel


def main():
    out = []; best = None
    for push in (0.6, 1.0, 1.5, 2.0, 3.0):
        for tau in (0.05, 0.1, 0.2):
            for w_in in (1.5, 2.5, 4.0):
                mv, hd, rl = point(push, tau, w_in)
                ok = mv >= 0.95 and hd >= 0.99 and rl is not None and rl <= 20
                out.append({"push": push, "tau": tau, "w_in": w_in, "move": mv, "hold": hd, "release": rl, "pass": bool(ok)})
                print(f"push {push} tau {tau} w_in {w_in}: MOVE {mv:.2f} HOLD {hd:.3f} RELEASE {rl} {'PASS' if ok else ''}", flush=True)
                if ok and (best is None or mv + hd > best[0]):
                    best = (mv + hd, push, tau, w_in)
    json.dump({"grid": out, "best": best, "verdict": {"PASS": best is not None}}, open(OUT, "w"), indent=1)
    print("CF-2c", "PASS" if best else "FAIL", best)


if __name__ == "__main__":
    main()
