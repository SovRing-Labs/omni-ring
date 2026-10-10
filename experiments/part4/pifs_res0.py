#!/usr/bin/env python3
"""PIFS-RES-0 (PIFS-RES plan (unpublished); pre-registered before this file): partitioned resonator with geometric partitions.
Partition the D dims into blocks; per block, jointly search the 3 rings' residues (block-local sift over the product
codebook when it fits, else per ring); collage = weighted vote over blocks; then resonate (decode_soft) from the collage.
Arms: NONE (plain SOFT) · CONTIG · STRIDE · HEX · TRI · ADAPT. Ladder/seeds = RESON-1b. See the plan for bars."""
import json, os, sys, time
import numpy as np

sys.path.insert(0, ".")
from omni_ring.crt import solve_crt  # noqa: E402
from omni_ring.resonator import decode_soft  # noqa: E402

OUT = os.path.expanduser("~/gpurun-out-0930/pifs_res0.result.json")
SMOKE = os.environ.get("SMOKE") == "1"
PR = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151]
SETS = [tuple(PR[i:i + 3]) for i in range(0, len(PR) - 2, 2)]
MMAX = int(os.environ.get("MMAX", 10**9))
SETS = [s for s in SETS if int(np.prod(s)) <= MMAX]
DS = [512, 1024] if SMOKE else [512, 1024, 4096]
NSTATE, SEEDS = (6, 2) if SMOKE else (30, 2)
ARMS = os.environ.get("ARMS", "NONE,CONTIG,STRIDE,HEX,TRI,ADAPT").split(",")
B = int(os.environ.get("B", 32))


def field_C(D):
    return 0.13 * D * D + 24 * D - 3700


def partition(kind, D, b=B):
    if kind == "CONTIG":
        return [np.arange(D)[k * D // b:(k + 1) * D // b] for k in range(b)]
    if kind == "STRIDE":
        return [np.arange(k, D, b) for k in range(b)]
    if kind == "TRI":   # residue-class buckets of the dimension index: (d mod 3, d mod 5) -> 15 classes x 2 halves
        idx = np.arange(D); cls = (idx % 3) * 5 + (idx % 5); half = (idx // (D // 2))
        return [idx[(cls == c) & (half == h)] for c in range(15) for h in range(2) if ((cls == c) & (half == h)).any()]
    if kind == "HEX":   # dims on a W x H grid; hex cells (axial coords) of radius r -> cells as blocks
        W = int(np.sqrt(D)); H = D // W; r = max(1, int(np.sqrt(D / b / 3)))
        idx = np.arange(W * H); x = idx % W; y = idx // W
        q = np.floor(x / (1.5 * r)).astype(int); rr = np.floor((y - (q % 2) * 0.87 * r) / (1.73 * r)).astype(int)
        cell = q * 1000 + rr
        return [idx[cell == c] for c in np.unique(cell)]
    raise ValueError(kind)


def block_votes(v, cbs, blocks):
    """Joint local search per block over the product of the 3 rings' codewords (M_b = prod m), returns per-ring vote weights."""
    n = len(cbs); ms = [len(c) for c in cbs]
    votes = [np.zeros(m) for m in ms]
    # product codebook on this block, built lazily per block (3 rings: m1*m2*m3 rows, at most ~ 3.3M at the top of the ladder)
    M = int(np.prod(ms))
    if M > 400_000:                                   # too big to enumerate: per-ring votes with the others unknown (weak)
        for blk in blocks:
            for i in range(n):
                s = (np.conj(cbs[i][:, blk]) @ v[blk]).real
                votes[i][int(np.argmax(s))] += 1.0
        return votes, False
    g = np.stack(np.meshgrid(*[np.arange(m) for m in ms], indexing="ij"), -1).reshape(-1, n)   # M x n residues
    for blk in blocks:
        prod = np.ones((M, len(blk)), dtype=complex)
        for i in range(n):
            prod *= cbs[i][g[:, i]][:, blk]
        s = (np.conj(prod) @ v[blk]).real
        top = int(np.argmax(s)); margin = float((np.sort(s)[-1] - np.sort(s)[-2]) / max(abs(s).max(), 1e-9))
        for i in range(n):
            votes[i][g[top, i]] += 1.0 + margin
    return votes, True


def decode_pifs(v, cbs, kind, D):
    blocks = partition(kind, D) if kind != "ADAPT" else partition("CONTIG", D, 16)
    votes, joint = block_votes(v, cbs, blocks)
    if kind == "ADAPT":                                # split low-margin blocks in 4 (quadtree) and re-vote them
        est = [int(np.argmax(w)) for w in votes]
        spread = [np.sort(w)[-1] - np.sort(w)[-2] for w in votes]
        if min(spread) < 2:
            blocks = [sub for blk in blocks for sub in np.array_split(blk, 4)]
            votes, joint = block_votes(v, cbs, blocks)
    init = [cbs[i][int(np.argmax(w))].copy() for i, w in enumerate(votes)]
    # resonate from the collage estimate (the existing soft loop, initialised at the collage)
    n = len(cbs); cbc = [np.conj(c) for c in cbs]; est = init
    for _ in range(20):
        for i in range(n):
            u = v.copy()
            for j in range(n):
                if j != i:
                    u *= np.conj(est[j])
            a = np.maximum((cbc[i] @ u).real / len(v), 0); e = a @ cbs[i]; est[i] = e / np.maximum(np.abs(e), 1e-9)
    return tuple(int(np.argmax((cbc[i] @ est[i]).real)) for i in range(n)), len(blocks)


def main():
    t0 = time.time(); out = {"ladder": [int(np.prod(s)) for s in SETS], "B": B, "arms": {}}
    for arm in ARMS:
        out["arms"][arm] = {}
        for D in DS:
            curve, low = [], 0
            for ms in SETS:
                M = int(np.prod(ms)); per_seed = []; nb = 0; tt = []
                for s in range(SEEDS):
                    rng = np.random.default_rng(100 * s + D)
                    rots = [rng.integers(0, m, D) * (2 * np.pi / m) for m in ms]
                    cbs = [np.exp(1j * np.outer(np.arange(m), r)) for m, r in zip(ms, rots)]
                    hit = 0
                    for _ in range(NSTATE):
                        x = int(rng.integers(0, M)); v = np.exp(1j * sum(x * r for r in rots)); t1 = time.time()
                        if arm == "NONE":
                            res = tuple(decode_soft(v, cbs, iterations=200, tolerance=1e-4).residues)
                        else:
                            res, nb = decode_pifs(v, cbs, arm, D)
                        tt.append(time.time() - t1); hit += int(solve_crt(list(res), ms) == x)
                    per_seed.append(hit / NSTATE)
                acc = float(np.mean(per_seed)); curve.append({"M": M, "acc": acc, "per_seed": per_seed, "blocks": nb, "ms": 1000 * float(np.median(tt))})
                print(f"{arm:6s} D {D:5d} M {M:>9,}: acc {acc:.3f} seeds {per_seed} blocks {nb} {1000 * np.median(tt):.0f} ms", flush=True)
                low = low + 1 if acc < 0.5 else 0
                if low >= 2:
                    break
            good = [c["M"] for c in curve if c["acc"] >= 0.95]; C = max(good) if good else 0
            spread = float(np.mean([abs(c["per_seed"][0] - c["per_seed"][1]) for c in curve]))
            out["arms"][arm][str(D)] = {"curve": curve, "C": C, "C_field": field_C(D), "ratio": C / field_C(D), "seed_spread": spread}
            print(f"{arm:6s} D {D}: C {C:,} (field {field_C(D):,.0f}, ratio {C / field_C(D):.3f}) seed spread {spread:.2f}", flush=True)
    v = {}
    if "NONE" in out["arms"]:
        for D in DS:
            none = out["arms"]["NONE"][str(D)]
            best = max((a for a in out["arms"] if a != "NONE"), key=lambda a: out["arms"][a][str(D)]["C"], default=None)
            v[str(D)] = {"best_arm": best, "best_C": out["arms"][best][str(D)]["C"] if best else None, "none_C": none["C"],
                         "geom_ge_contig": (max(out["arms"].get("HEX", {}).get(str(D), {}).get("C", 0), out["arms"].get("TRI", {}).get(str(D), {}).get("C", 0)) >= out["arms"].get("CONTIG", {}).get(str(D), {}).get("C", 0))}
    out["verdict"] = v; out["seconds"] = time.time() - t0; out["smoke"] = SMOKE
    json.dump(out, open(OUT, "w"), indent=1)
    print("PIFS-RES-0", v, f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
