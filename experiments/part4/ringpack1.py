#!/usr/bin/env python3
"""RINGPACK-1 (pre-registered in CORTEX-2 plan (unpublished) after RINGPACK-0's harness error, BEFORE running): which-slot recall on a
bus packed to the brim, with marks made UNIQUE per slot by binding each coordinate to the slot's key:
  slot_k = sum_{x in members_k} encode(x) (.) key_k      (key_k = random unit phase vector per slot; bind = Hadamard product)
  query  = encode(x) (.) key_k  for a known (k, x)        -> sift against all S slot vectors (int8 dot, the bus read)
S = 10,000 slots, M = 8 / 16 / 24 / 32 / 48 / 64 / 96 / 128 / 192 / 256 marks per slot, 2,000 queries per M.
FIELD PREDICTS: per-slot crowd noise sd ~ sqrt(M / 2D) (A4 / L1); the max over 10k other slots ~ 4 sd, so recall@1 ~ 1.0
up to M ~ 100 at D = 5,120 and degrades past M ~ 200. PASS: recall@1 >= 0.99 at M = 24. Reported: recall@1/@5 per M, the
first M with recall@1 < 0.9, ms per query, clone MB. Clone file: /tmp/omniring-clone1-M{M}.npy.
"""
import json, os, sys, time
import numpy as np

sys.path.insert(0, ".")
from omni_ring import DEFAULT_SPEC, generate_rotations  # noqa: E402
from omni_ring.chain import encode  # noqa: E402

OUT = os.path.expanduser("~/gpurun-out-0930/ringpack1.result.json")
S = int(os.environ.get("S", 10_000))
MS = [8, 16, 24, 32, 48, 64, 96, 128, 192, 256]
NQ = 2000
D = DEFAULT_SPEC.dimension
N = DEFAULT_SPEC.data_capacity


def main():
    rot = generate_rotations(DEFAULT_SPEC, seed=0)
    code = np.stack([encode(x, rot) for x in range(N)]).astype(np.complex64)
    g = np.random.default_rng(0)
    keys = np.exp(1j * g.uniform(0, 2 * np.pi, (S, D))).astype(np.complex64)
    res = {"S": S, "D": D, "N": N}
    for M in MS:
        t0 = time.time()
        members = np.stack([g.choice(N, M, replace=False) for _ in range(S)])
        slots = np.zeros((S, 2, D), dtype=np.int8)
        for a in range(0, S, 500):
            v = code[members[a:a + 500]].sum(1) * keys[a:a + 500]
            s = np.maximum(np.abs(v.real).max(1), np.abs(v.imag).max(1))[:, None] / 127
            slots[a:a + 500, 0] = np.rint(v.real / s); slots[a:a + 500, 1] = np.rint(v.imag / s)
        np.save(f"/tmp/omniring-clone1-M{M}.npy", slots)
        re = slots[:, 0].astype(np.float32); im = slots[:, 1].astype(np.float32)
        hit1 = hit5 = 0; tq = []
        for _ in range(NQ):
            si = int(g.integers(0, S)); x = int(members[si][g.integers(0, M)])
            q = code[x] * keys[si]; t1 = time.time()
            sc = re @ q.real + im @ q.imag
            tq.append(time.time() - t1)
            top = np.argsort(-sc)[:5]
            hit1 += int(top[0] == si); hit5 += int(si in top)
        res[str(M)] = {"recall1": hit1 / NQ, "recall5": hit5 / NQ, "ms_per_query": 1000 * float(np.median(tq)), "gen_s": time.time() - t0, "clone_mb": slots.nbytes / 1e6}
        print(f"M {M:>3}: recall@1 {hit1 / NQ:.4f} @5 {hit5 / NQ:.4f}  {1000 * np.median(tq):.1f} ms/query  (field sd {np.sqrt(M / (2 * D)):.3f})", flush=True)
        del slots, re, im
    below = [M for M in MS if res[str(M)]["recall1"] < 0.9]
    res["verdict"] = {"PASS": res["24"]["recall1"] >= 0.99, "first_M_below_0.9": below[0] if below else None}
    json.dump(res, open(OUT, "w"), indent=1)
    print("RINGPACK-1", "PASS" if res["verdict"]["PASS"] else "FAIL", res["verdict"])


if __name__ == "__main__":
    main()
