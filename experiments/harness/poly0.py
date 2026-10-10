#!/usr/bin/env python3
"""POLY-0 — sessions as polygons of triangles (p, tool, arg); int8 vs trit solid; re-time/re-role transforms; recursion.
Pre-registered: experiments/PREREGISTRATIONS-OMNITRI.md (POLY-0 + wave-1 operationalization + POLY-0 detail)."""
import datetime, json, os, sys, zlib
import numpy as np
sys.path.insert(0, "<your-data>/shards")
from episodes import load, DBS

D = 5120
PH = np.exp(2j * np.pi * np.arange(16) / 16).astype(np.complex64)
g = np.random.default_rng(12)
steps, meta = load(DBS["current"])
sess = {s: v for s, v in steps.items() if len(v) >= 16}
split = lambda tok: tok.split(":", 1) if ":" in tok else (tok, "(none)")  # noqa: E731
tools = sorted({split(t)[0] for v in sess.values() for t, _ in v}); args = sorted({split(t)[1] for v in sess.values() for t, _ in v})
ti, ai = {t: i for i, t in enumerate(tools)}, {a: i for i, a in enumerate(args)}
print(f"sessions >=16 steps: {len(sess)}, tools {len(tools)}, args {len(args)}", flush=True)
TOOL = g.integers(0, 16, (len(tools), D)); ARG = g.integers(0, 16, (len(args), D))
theta = g.uniform(0, 2 * np.pi, D)
P = [g.permutation(D) for _ in range(3)]
pos = lambda p: np.rint(np.outer(np.atleast_1d(p), theta) / (2 * np.pi / 16)).astype(np.int64) % 16  # noqa: E731
roleP = lambda p: pos(p)[:, P[0]]  # noqa: E731
roleT = TOOL[:, P[1]]; roleA = ARG[:, P[2]]
cleanT = (np.cos(2 * np.pi * roleT / 16).astype(np.float32), np.sin(2 * np.pi * roleT / 16).astype(np.float32))


CA = (np.cos(2 * np.pi * roleA / 16).astype(np.float32), np.sin(2 * np.pi * roleA / 16).astype(np.float32))  # once, not per call


def cleanA(u):
    return (u.real.astype(np.float32) @ CA[0].T + u.imag.astype(np.float32) @ CA[1].T) / D


def tri_ids(v, N):
    return np.array([(p, ti[split(t)[0]], ai[split(t)[1]]) for p, (t, _) in enumerate(v[:N])])


def polygon(T):
    return PH[(roleP(T[:, 0]) + roleT[T[:, 1]] + roleA[T[:, 2]]) % 16].sum(0)


def int8(a):
    s = max(np.abs(a.real).max(), np.abs(a.imag).max()) / 127 or 1
    return ((np.rint(a.real / s) + 1j * np.rint(a.imag / s)) * s).astype(np.complex64)


def trit(a):
    f = lambda x: (np.sign(x) * (np.abs(x) >= 0.5 * np.sqrt(np.mean(x ** 2)))).astype(np.float32)  # noqa: E731
    return (f(a.real) + 1j * f(a.imag)).astype(np.complex64)


def recover(acc, T, shift=0, delta=None):
    """Hidden-corner recovery for tool, arg, position (strict). Queries at p+shift; delta = re-role vector (phases)."""
    d = np.zeros(D, np.int64) if delta is None else delta
    Q = T[:64]; kp = roleP(Q[:, 0] + shift)
    u = acc * np.conj(PH[(kp + roleA[Q[:, 2]] + d) % 16]); cr, ci = cleanT
    okT = (((u.real @ cr.T + u.imag @ ci.T)).argmax(1) == Q[:, 1]).mean()
    u = acc * np.conj(PH[(kp + roleT[Q[:, 1]] + d) % 16]); okA = (cleanA(u).argmax(1) == Q[:, 2]).mean()
    N = len(T); PP = roleP(np.arange(N) + shift); okP = 0
    cp, spp = np.cos(2 * np.pi * PP / 16).astype(np.float32), np.sin(2 * np.pi * PP / 16).astype(np.float32)
    U = acc * np.conj(PH[(roleT[Q[:, 1]] + roleA[Q[:, 2]] + d) % 16])           # all 64 queries in one matmul
    SC = U.real.astype(np.float32) @ cp.T + U.imag.astype(np.float32) @ spp.T
    for q, sc in zip(Q, SC):
        A = set(np.nonzero((T[:, 1] == q[1]) & (T[:, 2] == q[2]))[0].tolist())
        okP += int(set(np.argsort(-sc)[:len(A)].tolist()) == A)
    return {"tool": float(okT), "arg": float(okA), "pos": okP / len(Q)}


res = {"curve": {}}; raw_b = []
for N in (16, 32, 64, 128, 256, 400):
    pool = [s for s, v in sess.items() if len(v) >= N]
    pick = [pool[i] for i in g.choice(len(pool), min(40, len(pool)), replace=False)] if pool else []
    if not pick:
        continue
    agg = {f: {"tool": [], "arg": [], "pos": []} for f in ("int8", "trit")}
    for s in pick:
        T = tri_ids(sess[s], N); acc = polygon(T)
        raw_b.append(np.mean([len(f"{p} {t}".encode()) for p, (t, _) in enumerate(sess[s][:N])]))
        for f, fn in (("int8", int8), ("trit", trit)):
            r = recover(fn(acc), T)
            for k in r:
                agg[f][k].append(r[k])
    row = {f: {k: float(np.mean(v)) for k, v in agg[f].items()} for f in agg}
    row["n_polygons"] = len(pick); res["curve"][N] = row
    print(f"POLY N={N:3d} ({len(pick)} polys)  int8 tool/arg/pos {row['int8']['tool']:.3f} {row['int8']['arg']:.3f} {row['int8']['pos']:.3f}"
          f" | trit {row['trit']['tool']:.3f} {row['trit']['arg']:.3f} {row['trit']['pos']:.3f}", flush=True)
C = res["curve"]; mn = lambda N, f: min(C[N][f].values()) if N in C else 0  # noqa: E731
raw = float(np.mean(raw_b)); res["bytes"] = {}
for N in C:
    for f, b in (("int8", 2 * D + 4), ("trit", 2048)):
        rec = np.mean(list(C[N][f].values()))
        res["bytes"][f"{f}_N{N}"] = b / (N * rec) if rec else None
res["bytes"]["raw_text_per_fact"] = raw
print("bytes/recovered fact:", {k: round(v, 1) for k, v in res["bytes"].items() if v}, flush=True)

# (b) transforms at N = 64 on the int8 polygon
pool = [s for s, v in sess.items() if len(v) >= 64]; pick = [pool[i] for i in g.choice(len(pool), min(40, len(pool)), replace=False)]
base, rt, rr = [], [], []
for s in pick:
    T = tri_ids(sess[s], 64); acc = int8(polygon(T)); r0 = recover(acc, T)
    moved = int8(acc * np.exp(1j * 37 * theta[P[0]]).astype(np.complex64)); r1 = recover(moved, T, shift=37)
    dl = g.integers(0, 16, D); r2 = recover(int8(acc * PH[dl]), T, delta=dl)
    base.append(np.mean(list(r0.values()))); rt.append(np.mean(list(r1.values()))); rr.append(np.mean(list(r2.values())))
res["transform"] = {"base": float(np.mean(base)), "retime": float(np.mean(rt)), "rerole": float(np.mean(rr))}
print("transform N=64:", res["transform"], flush=True)

# (c) recursion: polygon atoms as corners of higher triangles
names = pick = [s for s, v in sess.items() if len(v) >= 64]
accs = {s: int8(polygon(tri_ids(sess[s], 64))) for s in names}
PA = np.stack([np.rint(np.angle(accs[s]) / (2 * np.pi / 16)).astype(np.int64) % 16 for s in names])
rels = ["title", "day", "cost"]; RL = g.integers(0, 16, (3, D))
out_of = lambda s: [meta[s]["title"], datetime.date.fromtimestamp(meta[s]["t0"] / 1000).isoformat(),  # noqa: E731
                    f"cost{min(int(np.log2(1 + meta[s]['cost'] * 100)), 12)}"]
outs = sorted({o for s in names for o in out_of(s)}); oi = {o: i for i, o in enumerate(outs)}
OUTV = g.integers(0, 16, (len(outs), D)); Q2 = [g.permutation(D) for _ in range(3)]
HT = np.array([(k, r, oi[o]) for k, s in enumerate(names) for r, o in enumerate(out_of(s))])
rP, rR, rO = PA[:, Q2[0]], RL[:, Q2[1]], OUTV[:, Q2[2]]
B = int(np.ceil(len(HT) / 32)); h = (HT[:, 1] * 1_000_003 + HT[:, 2] * 7919) % B
hb = np.stack([int8(PH[(rP[HT[h == b, 0]] + rR[HT[h == b, 1]] + rO[HT[h == b, 2]]) % 16].sum(0)) for b in range(B)])
titles = [o for o in outs if sum(out_of(s)[0] == o for s in names) == 1]
qt = [(k, s) for k, s in enumerate(names) if out_of(s)[0] in set(titles)]
cp = (np.cos(2 * np.pi * rP / 16).astype(np.float32), np.sin(2 * np.pi * rP / 16).astype(np.float32))
hit = opened = 0
for k, s in qt:
    o = oi[out_of(s)[0]]; b = (0 * 1_000_003 + o * 7919) % B
    u = hb[b] * np.conj(PH[(rR[0] + rO[o]) % 16]); got = int(np.argmax(cp[0] @ u.real + cp[1] @ u.imag))
    hit += int(got == k)
    if got == k:
        r = recover(accs[names[got]], tri_ids(sess[names[got]], 64)); opened += np.mean(list(r.values()))
res["recursion"] = {"polygons": len(names), "higher_triangles": len(HT), "title_queries": len(qt),
                    "polygon_top1": hit / len(qt), "opened_recovery": opened / max(hit, 1)}
print("recursion:", res["recursion"], flush=True)
res["verdict"] = {"a": mn(64, "int8") >= 0.99 and mn(32, "trit") >= 0.99, "b": abs(res["transform"]["retime"] - res["transform"]["base"]) <= 0.005
                  and abs(res["transform"]["rerole"] - res["transform"]["base"]) <= 0.005,
                  "c": res["recursion"]["polygon_top1"] >= 0.99, "KILL": mn(16, "trit") < 0.99}
print("POLY-0:", res["verdict"], flush=True)
json.dump(res, open(os.path.expanduser("~/gpurun-out-0930/poly0.result.json"), "w"), indent=1)
