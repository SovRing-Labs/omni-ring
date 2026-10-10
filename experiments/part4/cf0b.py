#!/usr/bin/env python3
"""CORE-FIELD-0: CF-0 (bump persists / regenerates), CF-1 (residue rings read the bump exactly), CF-2 (source moves it;
HOLD / RELEASE). Pre-registered in CORE-FIELD plan (unpublished) (2026-10-02) before this harness existed.
Surface: 48 x 48 hex-neighbourhood grid wrapped on a torus; one update rule per cell (local excitation, global inhibition,
Burak–Fiete 2009 form); bump centre read by circular mean; residue readout = CAL-1 arm A encoding of (x, y) per axis."""
import json, os, sys
import numpy as np
sys.path.insert(0, ".")
from omni_ring.crt import direct_probe  # noqa: E402

OUT = os.path.expanduser("~/gpurun-out-0930/cf0b.result.json")
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


def main():
    out = {}
    # CF-0 persistence, drift, regeneration
    a = bump_at(10, 20); c0 = centre(a)
    for t in range(10_000):
        a = step(a, noise=0.05)
    c1 = centre(a); drift = wrap_dist(c0, c1); alive = a.max() > 0.2 and a.max() / max(a.mean(), 1e-9) > 5
    dmg = a.copy(); mask = rng.random(a.shape) < 0.3; dmg[mask] = 0
    for t in range(50):
        dmg = step(dmg)
    regen = wrap_dist(c1, centre(dmg))
    out["CF-0"] = {"alive_after_10k": bool(alive), "drift_cells_per_10k": drift, "regen_centre_err": regen, "peak_over_mean": float(a.max() / a.mean())}
    out["CF-0"]["PASS"] = bool(alive and drift <= 10 and regen <= 1.5)
    print("CF-0", out["CF-0"], flush=True)
    # CF-1 readout exactness
    ok = 0; flagged = 0
    for _ in range(2000):
        cx, cy = int(rng.integers(0, N)), int(rng.integers(0, N)); a = bump_at(cx, cy)
        for t in range(20):
            a = step(a, noise=0.05)
        ex, ey = centre(a); x, y = int(round(ex)) % N, int(round(ey)) % N
        ok += int(read_axis(0, x) == x and read_axis(1, y) == y and wrap_dist((cx, cy), (x, y)) <= 1.0)
    for _ in range(2000):   # single-ring corruption flagged
        v = int(rng.integers(0, 1365)); res = [v % m for m in MODS]; i = int(rng.integers(0, 5)); res[i] = (res[i] + int(rng.integers(1, MODS[i]))) % MODS[i]
        s = sum(cb[r] for cb, r in zip(CB[0], res)); ph = np.round(np.angle(s) * K / (2 * np.pi)).astype(int) % K
        _, xd = direct_probe(np.exp(1j * ph * 2 * np.pi / K), CB[0], MODS, CBC[0]); flagged += int(xd >= 1365 or xd != v)
    out["CF-1"] = {"read_exact": ok / 2000, "corruption_flagged": flagged / 2000, "PASS": bool(ok / 2000 >= 0.99 and flagged / 2000 >= 0.999)}
    print("CF-1", out["CF-1"], flush=True)
    # CF-2 MOVE / HOLD / RELEASE
    hits = 0
    for _ in range(50):
        a = bump_at(24, 24); target = np.array([24.0, 24.0])
        for t in range(100):
            vx, vy = rng.normal(0, 0.3, 2); target += (vx, vy)
            cx, cy = centre(a); a = step(a, inp=0.6 * bump_at(cx + vx, cy + vy))      # MOVE: bias toward the shifted spot
        hits += int(wrap_dist(centre(a), target % N) <= 2.0)
    a = bump_at(10, 10); held = 0
    for t in range(500):                                                              # HOLD under a competing push
        a = step(a, inp=1.0 * bump_at(10, 10) + 0.4 * bump_at(30, 30))
        held += int(wrap_dist(centre(a), (10, 10)) <= 1.0)
    rel = None
    for t in range(200):                                                              # RELEASE: competing push wins
        a = step(a, inp=0.6 * bump_at(30, 30))
        if wrap_dist(centre(a), (30, 30)) <= 2.0:
            rel = t; break
    out["CF-2"] = {"move_within_2": hits / 50, "hold_frac": held / 500, "release_ticks": rel, "PASS": bool(hits / 50 >= 0.95 and held / 500 >= 0.99 and rel is not None and rel <= 20)}
    print("CF-2", out["CF-2"], flush=True)
    json.dump(out, open(OUT, "w"), indent=1); print("CORE-FIELD CF-0b", {k: v["PASS"] for k, v in out.items()})


if __name__ == "__main__":
    main()
