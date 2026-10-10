#!/usr/bin/env python3
"""QUAT-2 (PREREGISTRATIONS-OMNITRI.md wave 4): the triangle memory on the 2T quaternion core (the prism's core)."""
import itertools, json, os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
from multiprocessing import Pool
import numpy as np

G = [np.array(v, float) for v in ([1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1])]
G = np.array(G + [-g for g in G] + [np.array(s) / 2 for s in itertools.product((1, -1), repeat=4)])


def qmul(a, b):
    w1, x1, y1, z1 = np.moveaxis(a, -1, 0); w2, x2, y2, z2 = np.moveaxis(b, -1, 0)
    return np.stack([w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2], -1)


MUL = np.array([[int(np.argmin(np.abs(G - qmul(G[a], G[b])).sum(1))) for b in range(24)] for a in range(24)], np.uint8)
CONJ = np.array([int(np.argmin(np.abs(G - G[a] * np.array([1, -1, -1, -1])).sum(1))) for a in range(24)], np.uint8)
D, NE, NR = 5120, 1000, 50
g0 = np.random.default_rng(41)
E = g0.integers(0, 24, (NE, D)).astype(np.uint8); R = g0.integers(0, 24, (NR, D)).astype(np.uint8)
P = [g0.permutation(D) for _ in range(3)]
role = [E[:, P[0]], R[:, P[1]], E[:, P[2]]]
TAB = [G[r].reshape(r.shape[0], D * 4).astype(np.float32) for r in role]   # clean-up tables per corner


def tri(s, r, o):
    return MUL[MUL[role[0][s], role[1][r]], role[2][o]]                        # (n, D) element indices


def int8q(acc):
    s = np.abs(acc).max() / 127
    return (np.rint(acc / s) * s).astype(np.float32)


def bucket(args):
    N, b, form = args
    g = np.random.default_rng(1000 * N + b)
    T = np.stack([g.integers(0, NE, N), g.integers(0, NR, N), g.integers(0, NE, N)], 1)
    acc = G[tri(T[:, 0], T[:, 1], T[:, 2])].sum(0)                          # (D, 4)
    acc = int8q(acc) if form == "int8" else acc.astype(np.float32)
    Q = T[:64]; s_, r_, o_ = Q[:, 0], Q[:, 1], Q[:, 2]
    uO = qmul(G[CONJ[MUL[role[0][s_], role[1][r_]]]], acc[None])           # o = (s r)^-1 t       (64, D, 4)
    uS = qmul(acc[None], G[CONJ[MUL[role[1][r_], role[2][o_]]]])           # s = t (r o)^-1
    uR = qmul(qmul(G[CONJ[role[0][s_]]], acc[None]), G[CONJ[role[2][o_]]]) # r = s^-1 t o^-1
    ok = [int(((uS.reshape(len(Q), -1).astype(np.float32) @ TAB[0].T).argmax(1) == s_).sum()),
          int(((uR.reshape(len(Q), -1).astype(np.float32) @ TAB[1].T).argmax(1) == r_).sum()),
          int(((uO.reshape(len(Q), -1).astype(np.float32) @ TAB[2].T).argmax(1) == o_).sum())]
    return N, form, ok, len(Q)


if __name__ == "__main__":
    Ns = (16, 32, 64, 128, 192, 256, 384, 512)
    with Pool(4) as pool:
        out = []
        for x in pool.imap_unordered(bucket, [(N, b, f) for N in Ns for f in ("float", "int8") for b in range(12)]):
            out.append(x)
            if len(out) % 24 == 0:
                print(f"progress {len(out)}/192", flush=True)
    res = {}
    for N in Ns:
        row = {}
        for f in ("float", "int8"):
            o = [x for x in out if x[0] == N and x[1] == f]; tot = sum(x[3] for x in o)
            row[f] = {c: sum(x[2][i] for x in o) / tot for i, c in enumerate("SRO")}
        res[N] = row
        print(f"QUAT-2 N={N:3d}  float S/R/O {row['float']['S']:.4f} {row['float']['R']:.4f} {row['float']['O']:.4f} | "
              f"int8 {row['int8']['S']:.4f} {row['int8']['R']:.4f} {row['int8']['O']:.4f}", flush=True)
    cap = max([N for N in Ns if min(res[N]["int8"].values()) >= 0.99] or [0])
    res["cap_int8"] = cap; res["bytes_per_bucket"] = {"2T_int8": 4 * D, "torus_int8": 2 * D}
    res["verdict"] = {"PASS": cap >= 256, "KILL": cap < 128}
    print("QUAT-2: cap_int8", cap, res["verdict"], flush=True)
    json.dump({str(k): v for k, v in res.items()}, open(os.path.expanduser("~/gpurun-out-0930/quat2.result.json"), "w"), indent=1)
