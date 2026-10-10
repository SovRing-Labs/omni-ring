#!/usr/bin/env python3
"""QUAT-0 (PREREGISTRATIONS-OMNITRI.md CORE): does a non-commuting 4-D core (unit quaternions, S^3) remove the torus's
commutative path collisions? Same translational tree world as TRI-1's phase-addition arm."""
import json, os, time
import numpy as np

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


def unitq(shape, rng):
    q = rng.normal(size=shape + (4,)); return q / np.linalg.norm(q, axis=-1, keepdims=True)


def run_torus(D=5120):
    rng = np.random.default_rng(11)
    V = rng.integers(0, 16, (NR, D)); X = np.zeros((N, D), np.int64); X[0] = rng.integers(0, 16, D)
    for i in range(1, N):
        p, r = parent[i]; X[i] = (X[p] + V[r]) % 16
    C, S = np.cos(np.pi / 8 * X).astype(np.float32), np.sin(np.pi / 8 * X).astype(np.float32)
    def query(y, path):
        q = (X[y] + V[path].sum(0)) % 16
        return (np.cos(np.pi / 8 * q) @ C.T + np.sin(np.pi / 8 * q) @ S.T) / D
    return query, D * 4 // 8                                            # 4-bit phases -> bytes


def run_quat(D):
    rng = np.random.default_rng(11)
    V = unitq((NR, D), rng); X = np.zeros((N, D, 4)); X[0] = unitq((D,), rng)
    for i in range(1, N):
        p, r = parent[i]; X[i] = qmul(X[p], V[r])                       # right multiplication: order matters
    flat = X.reshape(N, D * 4).astype(np.float32)
    def query(y, path):
        q = X[y]
        for r in path:
            q = qmul(q, V[r])
        return (flat @ q.reshape(-1).astype(np.float32)) / D
    return query, D * 4 * 4                                             # float32 x 4 -> bytes


res = {}
for name, (query, nbytes) in (("torus_5120", run_torus()), ("quat_1280", run_quat(1280)), ("quat_5120", run_quat(5120))):
    rows = {}
    for h in range(1, 11):
        ok = tot = coll = 0
        for z in range(N):
            y, path = path_of(z, h)
            if y is None:
                continue
            s = query(y, path)
            ok += int(s.argmax() == z); tot += 1; coll += int((s >= 0.999).sum() > 1)
        rows[h] = {"n": tot, "exact": ok / tot if tot else None, "collisions": coll}
    res[name] = {"bytes_per_vector": nbytes, "by_hops": rows}
    print(f"QUAT-0 {name:10s} ({nbytes:,} B/vector) exact/collisions by hops: " +
          " ".join(f"{h}:{r['exact']:.3f}/{r['collisions']}" for h, r in rows.items() if r['n']), flush=True)
rng = np.random.default_rng(0)
a, b = unitq((5120,), rng), unitq((5120,), rng)
t = time.perf_counter(); [qmul(a, b) for _ in range(200)]; tq = (time.perf_counter() - t) / 200 * 1e6
pa, pb = rng.integers(0, 16, 5120), rng.integers(0, 16, 5120)
t = time.perf_counter(); [(pa + pb) % 16 for _ in range(200)]; tt = (time.perf_counter() - t) / 200 * 1e6
res["bind_us_5120"] = {"torus": tt, "quat": tq}
q = res["quat_5120"]["by_hops"]; tr = res["torus_5120"]["by_hops"]
res["verdict"] = {"PASS": all(q[h]["exact"] >= 0.995 and q[h]["collisions"] == 0 for h in q if q[h]["n"]) and
                  any(tr[h]["collisions"] > 0 for h in tr if tr[h]["n"]),
                  "KILL": sum(q[h]["collisions"] for h in q) >= sum(tr[h]["collisions"] for h in tr)}
print("bind us (numpy, 5,120 coords): torus %.1f quat %.1f" % (tt, tq), "| QUAT-0:", res["verdict"], flush=True)
json.dump({k: v for k, v in res.items()}, open(os.path.expanduser("~/gpurun-out-0930/quat0.result.json"), "w"), indent=1, default=str)
