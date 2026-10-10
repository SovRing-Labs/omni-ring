#!/usr/bin/env python3
"""SHAPE-3D-0 (TRUEFORM plan (unpublished)): 2-D vs 3-D vs simplex key solids for routing, TF-3 protocol."""
import json, os, time
import numpy as np
OUT = os.path.expanduser("~/gpurun-out-0930/shape3d0.result.json"); d = 256; NQ = 20000


def verts(name, K):
    if name == "RANDOM":
        return None
    if name == "SQUARE":   return np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], float)
    if name == "TETRA":    return np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)
    if name == "OCTAGON":  return np.stack([[np.cos(2 * np.pi * k / 8), np.sin(2 * np.pi * k / 8)] for k in range(8)])
    if name == "CUBE":     return np.array([[sx, sy, sz] for sx in (1, -1) for sy in (1, -1) for sz in (1, -1)], float)
    if name == "HEXAGON":  return np.stack([[np.cos(2 * np.pi * k / 6), np.sin(2 * np.pi * k / 6)] for k in range(6)])
    if name == "OCTA":     return np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)
    if name.startswith("SIMPLEX"):
        e = np.eye(K); c = e - e.mean(0); return c
    raise ValueError(name)


def keys(name, K, rng):
    V = verts(name, K)
    if V is None:
        X = rng.normal(size=(K, d))
    else:
        V = V / np.linalg.norm(V, axis=1, keepdims=True)
        Q, _ = np.linalg.qr(rng.normal(size=(d, V.shape[1])))      # random orthonormal embedding: Gram exact
        X = V @ Q.T
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def acc(X, sigma, rng):
    K = len(X); idx = rng.integers(0, K, NQ)
    q = X[idx] + sigma * rng.normal(size=(NQ, d))
    return float((np.argmax(q @ X.T, 1) == idx).mean())


def main():
    rng = np.random.default_rng(0); out = {}
    for K, arms in ((4, ["SQUARE", "TETRA", "RANDOM"]), (6, ["HEXAGON", "OCTA", "SIMPLEX6", "RANDOM"]), (8, ["OCTAGON", "CUBE", "SIMPLEX8", "RANDOM"])):
        Xs = {a: keys(a, K, rng) for a in arms}
        sig = None
        for s in np.arange(0.05, 3.0, 0.02):
            if acc(Xs["RANDOM"], s, rng) <= 0.80:
                sig = float(s); break
        row = {"sigma_star": sig, "gram_min_angle_deg": {a: float(np.degrees(np.arccos(np.clip(np.max((Xs[a] @ Xs[a].T)[~np.eye(K, dtype=bool)]), -1, 1)))) for a in arms}}
        for a in arms:
            row[a] = float(np.mean([acc(Xs[a], sig, np.random.default_rng(7 + r)) for r in range(3)]))
        out[str(K)] = row
        print(f"K {K} sigma* {sig:.3f}: " + "  ".join(f"{a} {row[a]:.3f} (min angle {row['gram_min_angle_deg'][a]:.0f}°)" for a in arms), flush=True)
    v = {"tetra_ge_square": out["4"]["TETRA"] >= out["4"]["SQUARE"] + 0.02, "octa_ge_hexagon": out["6"]["OCTA"] >= out["6"]["HEXAGON"] + 0.02,
         "cube_ge_octagon": out["8"]["CUBE"] >= out["8"]["OCTAGON"] + 0.02, "simplex8_ge_cube": out["8"]["SIMPLEX8"] >= out["8"]["CUBE"], "simplex6_ge_octa": out["6"]["SIMPLEX6"] >= out["6"]["OCTA"]}
    v["PASS_J_theory"] = v["tetra_ge_square"] and v["octa_ge_hexagon"] and v["cube_ge_octagon"]
    out["verdict"] = v; json.dump(out, open(OUT, "w"), indent=1); print("SHAPE-3D-0", v)


if __name__ == "__main__":
    main()
