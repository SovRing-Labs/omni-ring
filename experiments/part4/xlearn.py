#!/usr/bin/env python3
"""XLEARN-0 — what the last days' data says (reads the result JSONs directly + experiments.db). Four questions, each a fit:
Q1 capacity vs D per decoder (SOFT / STD-SUPER / partition arms / sieve) against Kymn's quadratic — is ours a different law?
Q2 the codebook lottery: per-seed spread vs M and D — does it shrink with D (crowd noise) or is it draw-dependent?
Q3 shape: routing accuracy vs the keys' minimum pairwise angle across K (SHAPE-3D-0) — one curve?
Q4 cost: ms vs M for sieve / part / peel — the exponents (sieve ~ M^0, part ~ M^a, peel ~ K·M^b).
Writes ~/gpurun-out-0930/xlearn.result.json and prints the fits. No verdicts: this is the learning pass."""
import glob, json, os, sqlite3
import numpy as np

O = os.path.expanduser("~/gpurun-out-0930")
J = lambda f: json.load(open(os.path.join(O, f)))  # noqa: E731
out = {}


def fit_pow(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float); m = (x > 0) & (y > 0)
    if m.sum() < 2: return None
    b, a = np.polyfit(np.log(x[m]), np.log(y[m]), 1); return {"exponent": float(b), "coef": float(np.exp(a))}


# Q1 capacity per decoder vs D
q1 = {}
try:
    r = J("reson1b.result.json")
    for arm in r["arms"]:
        q1[arm] = {D: r["arms"][arm][D]["C"] for D in r["arms"][arm]}
except Exception as e: q1["reson1b_err"] = str(e)
field = {D: 0.13 * int(D) ** 2 + 24 * int(D) - 3700 for D in ("512", "1024", "4096")}
q1["field_kymn"] = field
try:
    s = J("sieve0.result.json")
    q1["SIEVE"] = {D: max(int(M) for M, row in s[D].items() if isinstance(row, dict) and row.get("clean") == 1.0) for D in ("512", "1024", "4096")}
except Exception as e: q1["sieve_err"] = str(e)
for arm in ("CONTIG", "ADAPT", "TRI", "HEX", "STRIDE"):
    try:
        lines = open(os.path.join(O, f"pifs_res0.smoke.{arm}.txt")).read().splitlines()
        q1[f"PIFS-{arm}"] = {l.split()[2].rstrip(":"): int(l.split("C ")[1].split()[0].replace(",", "")) for l in lines if ": C " in l}
    except Exception: pass
out["Q1_capacity_vs_D"] = q1
# Q1 reading: exponent of C vs D where available
for arm, cur in list(q1.items()):
    if isinstance(cur, dict) and all(k in cur for k in ("512", "1024")) and all(isinstance(cur[k], (int, float)) for k in cur):
        Ds = sorted(int(k) for k in cur); Cs = [cur[str(d)] for d in Ds]
        f = fit_pow(Ds, Cs); q1[f"{arm}_fit"] = f

# Q2 lottery
q2 = {}
try:
    r = J("reson1b.result.json")
    for arm in r["arms"]:
        rows = []
        for D, v in r["arms"][arm].items():
            for c in v["curve"]:
                if "acc" in c: rows.append((int(D), c["M"], c["acc"]))
        q2[arm] = {"n": len(rows), "frac_exactly_0.5": float(np.mean([abs(a - 0.5) < 0.02 for _, _, a in rows]))}
except Exception as e: q2["err"] = str(e)
try:
    p = J("pifs_res1.result.json")
    q2["PIFS-RES-1_spread"] = {a: {D: p["arms"][a][D]["seed_spread"] for D in p["arms"][a]} for a in p["arms"]}
except Exception: q2["PIFS-RES-1_spread"] = "pending"
out["Q2_lottery"] = q2

# Q3 shape: accuracy vs min angle
q3 = {}
try:
    s = J("shape3d0.result.json"); pts = []
    for K, row in s.items():
        if not K.isdigit(): continue
        for arm, ang in row["gram_min_angle_deg"].items():
            pts.append((ang, row[arm], int(K), arm))
    angs = np.array([p[0] for p in pts]); accs = np.array([p[1] for p in pts])
    b, a = np.polyfit(angs, accs, 1)
    q3 = {"points": pts, "linear_fit_acc_vs_angle": {"slope_per_deg": float(b), "intercept": float(a)},
          "corr": float(np.corrcoef(angs, accs)[0, 1])}
except Exception as e: q3["err"] = str(e)
out["Q3_shape_angle_law"] = q3

# Q4 cost exponents
q4 = {}
try:
    s = J("sieve0.result.json")
    for D in ("512", "1024", "4096"):
        Ms = [int(M) for M in s[D] if M.isdigit()]; ts = [s[D][str(M)]["ms"] for M in Ms]
        q4[f"SIEVE_D{D}"] = fit_pow(Ms, ts)
except Exception as e: q4["sieve_err"] = str(e)
try:
    for arm in ("CONTIG",):
        lines = open(os.path.join(O, f"pifs_res0.smoke.{arm}.txt")).read().splitlines()
        Ms, ts = [], []
        for l in lines:
            if " M " in l and " ms" in l and "D   512" in l:
                Ms.append(int(l.split("M")[1].split(":")[0].replace(",", ""))); ts.append(float(l.split()[-2]))
        q4["PART_bruteforce_D512"] = fit_pow(Ms, ts)
except Exception as e: q4["part_err"] = str(e)
try:
    p2 = J("peel2c.result.json"); q4["PEEL_ms_per_K"] = {K: v["ms"] for K, v in p2.items()}
except Exception: pass
out["Q4_cost"] = q4

json.dump(out, open(os.path.join(O, "xlearn.result.json"), "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in out.items()}, indent=1, default=str)[:6000])
