#!/usr/bin/env python3
"""QUAT-1 (PREREGISTRATIONS-OMNITRI.md): the 4-D core quantised to the binary tetrahedral group 2T (24 unit quaternions)."""
import itertools, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
g = np.random.default_rng(5)
N, NR = 500, 10
parent, kids = {}, {0: {}}
for i in range(1, N):
    while True:
        p = int(g.integers(0, i)); free = [r for r in range(NR) if r not in kids[p]]
        if free:
            break
    r = int(g.choice(free)); kids[p][r] = i; kids[i] = {}; parent[i] = (p, r)


def path_of(z, h):
    path, y = [], z
    while y in parent and len(path) < h:
        p, r = parent[y]; path.append(r); y = p
    return (y, path[::-1]) if len(path) == h else (None, None)


def qmul(a, b):
    """Hamilton product of quaternion arrays (..., 4)."""
    w1, x1, y1, z1 = np.moveaxis(a, -1, 0); w2, x2, y2, z2 = np.moveaxis(b, -1, 0)
    return np.stack([w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2], -1)



# the 24 elements of 2T: +-1, +-i, +-j, +-k and (+-1 +-i +-j +-k)/2
G = [np.array(v, float) for v in ([1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1])]
G = G + [-g for g in G] + [np.array(s) / 2 for s in itertools.product((1, -1), repeat=4)]
G = np.array(G)
assert len(G) == 24
MUL = np.zeros((24, 24), np.uint8)
for a in range(24):
    for b in range(24):
        p = qmul(G[a], G[b]); MUL[a, b] = int(np.argmin(np.abs(G - p).sum(1)))
assert all(np.allclose(qmul(G[a], G[b]), G[MUL[a, b]]) for a in range(24) for b in range(24)), "2T not closed"
DOT = (G @ G.T).astype(np.float32)                                   # similarity table
commuting = sum(MUL[a, b] == MUL[b, a] for a in range(24) for b in range(24)) / 576
print(f"2T: 24 elements, closed; fraction of commuting pairs {commuting:.3f}", flush=True)

D = 5120
rng = np.random.default_rng(11)
V = rng.integers(0, 24, (NR, D)).astype(np.uint8)
X = np.zeros((N, D), np.uint8); X[0] = rng.integers(0, 24, D)
cols = np.arange(D)
for i in range(1, N):
    p, r = parent[i]; X[i] = MUL[X[p], V[r]]
onehotT = np.zeros((N, D, 24), np.float32)                           # scores = mean_d DOT[q_d, X[n,d]]


def query(y, path):
    q = X[y]
    for r in path:
        q = MUL[q, V[r]]
    return DOT[q[None, :], X].mean(1)                                # (N,)


rows = {}
for h in range(1, 11):
    ok = tot = coll = 0
    for z in range(N):
        y, path = path_of(z, h)
        if y is None:
            continue
        s = query(y, path); ok += int(s.argmax() == z); tot += 1; coll += int((s >= 0.999).sum() > 1)
    rows[h] = {"n": tot, "exact": ok / tot, "collisions": coll}
print("QUAT-1 2T (5,120 B/vector) exact/collisions by hops: " +
      " ".join(f"{h}:{r['exact']:.3f}/{r['collisions']}" for h, r in rows.items()), flush=True)
a, b = X[1], V[2]
t = time.perf_counter(); [MUL[a, b] for _ in range(2000)]; tb = (time.perf_counter() - t) / 2000 * 1e6
pa, pb = rng.integers(0, 16, D).astype(np.uint8), rng.integers(0, 16, D).astype(np.uint8)
t = time.perf_counter(); [(pa + pb) & 15 for _ in range(2000)]; tt = (time.perf_counter() - t) / 2000 * 1e6
res = {"commuting_pairs": float(commuting), "by_hops": rows, "bind_us": {"2T_table": tb, "torus_uint8": tt},
       "bytes_per_vector": D}
res["verdict"] = {"PASS": all(r["exact"] >= 0.995 and r["collisions"] == 0 for r in rows.values())}
print(f"bind us (5,120 coords, numpy): 2T table {tb:.1f}, torus uint8 {tt:.1f} | QUAT-1: {res['verdict']}", flush=True)
json.dump(res, open(os.path.expanduser("~/gpurun-out-0930/quat1.result.json"), "w"), indent=1, default=str)
