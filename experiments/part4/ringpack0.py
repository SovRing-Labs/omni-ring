#!/usr/bin/env python3
"""RINGPACK-0 (J 2026-10-02: "make a test omniring / clone of the dev version, pack it to the brim and test it on recall").
Pre-registered before running. A CLONE of the omniring store (file-backed, not the live /dev/shm bus): S slots, each one
phase vector (D 5,120, the default 5-ring spec 3·5·7·13 + 17 check) holding a SUPERPOSITION of M marks (coordinates in
[0, 1,365)), stored int8 re/im like the bus. Packed to the brim = M swept 8 / 16 / 24 / 32 / 48 / 64 marks per slot at
S = 10,000 slots (the biggest clone that fits beside the running loop: 10k x 5,120 x 2 int8 = 102 MB per M level).
Two recalls:
  R1 per-slot readout (the omniring's own PEEL): read_all on a 100-slot sample per M -> true marks recovered / slot, fakes.
  R2 which-slot recall (a cued query across the whole packed clone): for 2,000 (slot, mark) pairs, encode the mark and sift
     it against all 10,000 slot vectors (int8 dot, the bus's read) -> recall@1 of the owning slot, and @5.
FIELD PREDICTS: R1 at M=24: 14.8-16.7 true/slot, fakes <= 3.6% (OPEEL-READ, RING3); R2: crowd noise sd ~ sqrt(M/2D) per
slot (A4/L1), max over 10k slots ~ 4 sd -> recall@1 ~1.0 up to M ~ 100, degrading past ~ M 200 (Plate; our L1 knee).
PASS: R1 at M=24 >= 14 true/slot and fakes <= 5%; R2 recall@1 >= 0.99 at M = 24. Reported: R1/R2 at every M; the M where
R2 drops below 0.9; ms per which-slot query; clone size. KILL: R1 < 10 at M=24 or R2 < 0.9 at M=24.
Clone file: /tmp/omniring-clone-M{M}.npy (int8, S x 2 x D).
"""
import json, os, sys, time
import numpy as np

sys.path.insert(0, ".")
from omni_ring import DEFAULT_SPEC, generate_rotations  # noqa: E402
from omni_ring.chain import encode  # noqa: E402
from omni_ring.peel import read_all  # noqa: E402

OUT = os.path.expanduser("~/gpurun-out-0930/ringpack0.result.json")
S = int(os.environ.get("S", 10_000))
MS = [8, 16, 24, 32, 48, 64]
NPEEL = 100
NQ = 2000
D = DEFAULT_SPEC.dimension
N = DEFAULT_SPEC.data_capacity


def main():
    rot = generate_rotations(DEFAULT_SPEC, seed=0)        # seed 0: a clean-decoding codebook draw (peel.py disclosure)
    code = np.stack([encode(x, rot) for x in range(N)]).astype(np.complex64)   # 1,365 x D exact codewords
    g = np.random.default_rng(0)
    res = {"S": S, "D": D, "N": N}
    for M in MS:
        t0 = time.time()
        members = np.stack([g.choice(N, M, replace=False) for _ in range(S)])  # S x M
        slots = np.zeros((S, 2, D), dtype=np.int8)
        for a in range(0, S, 500):
            v = code[members[a:a + 500]].sum(1)                                  # 500 x D complex
            s = np.maximum(np.abs(v.real).max(1), np.abs(v.imag).max(1))[:, None] / 127
            slots[a:a + 500, 0] = np.rint(v.real / s); slots[a:a + 500, 1] = np.rint(v.imag / s)
        path = f"/tmp/omniring-clone-M{M}.npy"; np.save(path, slots)
        tgen = time.time() - t0
        # R1: peel on a sample of slots
        true = fake = 0; tp = time.time()
        for k in range(NPEEL):
            v = (slots[k, 0].astype(np.float32) + 1j * slots[k, 1].astype(np.float32)).astype(np.complex64)
            got = read_all(v, rot, DEFAULT_SPEC)
            mem = set(int(x) for x in members[k])
            true += len([x for x in got if x in mem]); fake += len([x for x in got if x not in mem])
        tpeel = (time.time() - tp) / NPEEL
        # R2: which-slot recall across the whole clone (int8 dot = the bus read)
        re = slots[:, 0].astype(np.float32); im = slots[:, 1].astype(np.float32)
        hit1 = hit5 = 0; tq = []
        for _ in range(NQ):
            si = int(g.integers(0, S)); x = int(members[si][g.integers(0, M)])
            q = code[x]; t1 = time.time()
            sc = re @ q.real + im @ q.imag                                       # S scores
            tq.append(time.time() - t1)
            top = np.argsort(-sc)[:5]
            hit1 += int(top[0] == si); hit5 += int(si in top)
        r = {"gen_s": tgen, "clone_mb": slots.nbytes / 1e6, "R1_true_per_slot": true / NPEEL, "R1_fakes_frac": fake / max(1, true + fake),
             "R1_ms_per_slot": 1000 * tpeel, "R2_recall1": hit1 / NQ, "R2_recall5": hit5 / NQ, "R2_ms_per_query": 1000 * float(np.median(tq))}
        res[str(M)] = r
        print(f"M {M:>3}: R1 peel {true / NPEEL:.2f} true/slot fakes {fake / max(1, true + fake):.3f} ({1000 * tpeel:.0f} ms/slot) | "
              f"R2 which-slot recall@1 {hit1 / NQ:.4f} @5 {hit5 / NQ:.4f} ({1000 * np.median(tq):.1f} ms/query) | clone {slots.nbytes / 1e6:.0f} MB", flush=True)
        del slots, re, im
    r24 = res["24"]
    passed = r24["R1_true_per_slot"] >= 14 and r24["R1_fakes_frac"] <= 0.05 and r24["R2_recall1"] >= 0.99
    below = [M for M in MS if res[str(M)]["R2_recall1"] < 0.9]
    res["verdict"] = {"PASS": passed, "R2_first_M_below_0.9": below[0] if below else None}
    json.dump(res, open(OUT, "w"), indent=1)
    print("RINGPACK-0", "PASS" if passed else "FAIL", res["verdict"])


if __name__ == "__main__":
    main()
