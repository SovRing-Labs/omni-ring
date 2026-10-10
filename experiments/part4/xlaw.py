#!/usr/bin/env python3
"""Capacity law across experiment families (reads control-center/experiments.db).
Model (superposition crosstalk): with n items superposed in D dims, each wrong candidate's score after unbinding is noise
with variance ~ n/(2D); picking the right item among C candidates: acc = Phi(a * sqrt(D/n))^(C-1). a = our efficiency
(1 = ideal). Fit a per family and globally; report n* (99% recovery) = a^2 * D / z99(C)^2."""
import json, math, os, sqlite3
import numpy as np

DB = "<your-data>/control-center/experiments.db"
con = sqlite3.connect(DB)
Phi = lambda z: 0.5 * (1 + math.erf(z / math.sqrt(2)))  # noqa: E731


def pts(exp, pattern, D, C, label):
    out = []
    for path, v in con.execute("select path, num from results where exp=? and path like ? and num is not null", (exp, pattern)):
        n = int([s for s in path.split("/") if s.isdigit()][0])
        out.append((label, D, C, n, v))
    return out


DATA = (pts("SIMPLEX-0", "/%/random/recovery", 5115, 16, "SIMPLEX-0 random (int8, K16)")
        + pts("TF-0", "/%/A/recovery", 5120, 16, "TF-0 A (int8, K16, D5120)")
        + pts("TF-0", "/%/B/recovery", 2560, 16, "TF-0 B (int8, K240, D2560)")
        + pts("TF-2", "/%/RANDOM/recovery", 5118, 16, "TF-2 random roles (int8, K16)")
        + pts("TF-6", "/%/PURE", 5120, 16, "TF-6 pure (int8, K16)")
        + pts("TF-1", "/RANDOM/%/exact", 5120, 6194, "TF-1 random pos (float, C6194)")
        + pts("TF-1", "/WIND/%/exact", 5120, 6194, "TF-1 wind pos (float, C6194)"))


def pred(a, D, C, n):
    return Phi(a * math.sqrt(D / n)) ** (C - 1)


def fit(rows):
    use = [r for r in rows if 0.02 < r[4] < 0.9999]          # saturated points carry no information about a
    grid = np.linspace(0.5, 1.6, 1101)
    if not use:
        return None, None
    err = [sum((pred(a, D, C, n) - acc) ** 2 for _, D, C, n, acc in use) for a in grid]
    a = float(grid[int(np.argmin(err))])
    return a, math.sqrt(min(err) / len(use))


if __name__ == "__main__":
    res = {"families": {}}
    for lab in sorted({r[0] for r in DATA}):
        rows = [r for r in DATA if r[0] == lab]
        a, rmse = fit(rows)
        res["families"][lab] = {"a": a, "rmse": rmse, "points": [(n, round(acc, 4), round(pred(a, D, C, n), 4) if a else None)
                                                                  for _, D, C, n, acc in sorted(rows, key=lambda r: r[3])]}
        print(f"{lab:36s} a = {a}  rmse {rmse}  (n, measured, predicted): {res['families'][lab]['points']}")
    a, rmse = fit(DATA)
    res["global"] = {"a": a, "rmse": rmse}
    print(f"GLOBAL a = {a:.3f}, rmse {rmse:.4f} over {len([r for r in DATA if 0.02 < r[4] < 0.9999])} informative points")
    z = lambda C: next(t for t in np.linspace(0, 8, 8001) if Phi(t) ** (C - 1) >= 0.99)  # noqa: E731
    for C in (16, 1000, 6194, 100000):
        res.setdefault("n99_per_1000_dims", {})[C] = round(a * a * 1000 / z(C) ** 2, 1)
    print("items holdable at 99% per 1,000 dims (law, global a):", res["n99_per_1000_dims"])
    json.dump(res, open(os.path.expanduser("~/gpurun-out-0930/xlaw.result.json"), "w"), indent=1)
