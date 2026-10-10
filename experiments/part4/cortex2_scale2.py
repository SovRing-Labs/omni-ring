#!/usr/bin/env python3
"""CORTEX-2 SCALE-1 (J 2026-10-02: "largest vector matrix we can try, then an array of vectors").
Pre-registered before running. RAM budget: ~6 GB available with the step-3 loop running.
  A. Largest single lattice in RAM: unpacked 400,000 x 5,120 (2.05 GB) with the VM (MODE 0, AVX2) and the JIT kernel;
     then PACKED 4-bit 1,000,000 x 5,120 (2.56 GB) with the p4 kernel (sift_batch_p4), checked bit-for-bit against the
     unpacked kernel on a 20,000-row slice.
  B. Array of vectors = a batch of queries against the lattice in one call (sift_batch, TILE-1 tiled kernel): batches of
     1 / 16 / 64 / 256 queries on the 400k lattice; ms per query vs single-query.
FIELD PREDICTS: memory-bound scans: 2.05 GB at ~8 GB/s ≈ 250 ms/query; packed halves the bytes (≈ 1.3 GB → ~160 ms);
batching amortises the scan — per-query time falls toward compute-bound (TILE-1 measured 1.18-1.33x; expect ≥ 4x at 64+).
PASS (reported bars): top-1 correct 16/16 golden at every size; packed == unpacked bit-for-bit on the slice; per-query ms
reported. KILL: wrong top-1 or packed/unpacked mismatch.
"""
import json, os, sys, time
import numpy as np

sys.path.insert(0, ".")
from omni_ring import seed, jit  # noqa: E402
from omni_ring.vm import VM, ProgramCodebook, iid  # noqa: E402

OUT = os.path.expanduser("~/gpurun-out-0930/cortex2_scale2.result.json")
N_UNPACKED = int(os.environ.get("N_UNPACKED", 400_000))
N_PACKED = int(os.environ.get("N_PACKED", 1_000_000))


def gen(n, d, g, gold):
    lat = np.empty((n, d), dtype=np.uint8)
    for a in range(0, n, 50_000):
        lat[a:a + 50_000] = g.integers(0, 16, (min(50_000, n - a), d), dtype=np.uint8)
    lat[:16] = gold
    return lat


def main():
    res = {}
    gold = np.stack([seed.codeword(f"golden-{i}", 0) for i in range(16)])
    g = np.random.default_rng(1)
    cb = ProgramCodebook(seed.D)
    # ---- A1 unpacked 400k ----
    t0 = time.time(); lat = gen(N_UNPACKED, seed.D, g, gold); print(f"gen {N_UNPACKED:,}: {time.time() - t0:.1f}s {lat.nbytes / 1e9:.2f} GB", flush=True)
    vm = VM(lat, cb)
    vm.load_program(cb.write([iid("MODE", 0), iid("LOAD", 0), iid("SIFT", 3), iid("THRESHOLD", 2)]))
    ok = 0; ts = []
    for i in range(16):
        vm.set(0, gold[i]); t1 = time.time(); m = vm.tick(); ts.append(time.time() - t1); ok += int(m.topk[:1] == [i])
    tj = []; mism = 0
    for i in range(4):
        t1 = time.time(); a = jit.sift_jit(gold[i], lat); tj.append(time.time() - t1); mism += int(not np.array_equal(a, vm.scores if i == 15 else seed.sift_phase(gold[i], lat)))
    res["A1_unpacked"] = {"n": N_UNPACKED, "gb": lat.nbytes / 1e9, "top1": ok, "vm_ms_p50": 1000 * float(np.median(ts)), "jit_ms_p50": 1000 * float(np.median(tj)), "jit_mismatches": mism}
    print(f"A1 unpacked {N_UNPACKED:,}: top-1 {ok}/16  VM {1000 * np.median(ts):.0f} ms  JIT {1000 * np.median(tj):.0f} ms  mism {mism}", flush=True)
    # ---- B batch (array of vectors) on the 400k lattice ----
    res["B_batch"] = {}
    for bq in (1, 16, 64, 256):
        qs = np.stack([gold[i % 16] for i in range(bq)]) if bq > 1 else gold[:1]
        seed.sift_batch(qs[:1], lat)  # warm
        t1 = time.time(); sc = seed.sift_batch(qs, lat); dt = time.time() - t1
        top = sc.argmax(1); okb = int((top == np.arange(bq) % 16).sum())
        res["B_batch"][str(bq)] = {"ms_per_query": 1000 * dt / bq, "ms_total": 1000 * dt, "top1": okb}
        print(f"B batch {bq:>3}: {1000 * dt / bq:.1f} ms/query ({1000 * dt:.0f} ms total) top-1 {okb}/{bq}", flush=True)
    slice_un = lat[:20_000].copy()
    del lat, vm
    # ---- A2 packed 1M ----
    t0 = time.time()
    pk = np.empty((N_PACKED, seed.D // 2), dtype=np.uint8)
    for a in range(0, N_PACKED, 50_000):
        pk[a:a + 50_000] = seed.pack4(g.integers(0, 16, (min(50_000, N_PACKED - a), seed.D), dtype=np.uint8))
    pk[:20_000] = seed.pack4(slice_un)
    print(f"gen packed {N_PACKED:,}: {time.time() - t0:.1f}s {pk.nbytes / 1e9:.2f} GB", flush=True)
    okp = 0; tp = []
    for i in range(16):
        t1 = time.time(); sc = seed.sift_batch_p4(gold[i][None, :], pk)[0]; tp.append(time.time() - t1); okp += int(int(sc.argmax()) == i)
    mism_p = 0
    for i in range(16):
        a = seed.sift_batch_p4(gold[i][None, :], pk[:20_000])[0]; b = seed.sift_phase(gold[i], slice_un)
        mism_p += int(not np.array_equal(a, b))
    res["A2_packed"] = {"n": N_PACKED, "gb": pk.nbytes / 1e9, "top1": okp, "ms_p50": 1000 * float(np.median(tp)), "packed_vs_unpacked_mismatches": mism_p}
    print(f"A2 packed {N_PACKED:,}: top-1 {okp}/16  {1000 * np.median(tp):.0f} ms/query  packed==unpacked mism {mism_p}/16", flush=True)
    res["verdict"] = {"PASS": ok == 16 and okp == 16 and mism == 0 and mism_p == 0}
    json.dump(res, open(OUT, "w"), indent=1)
    print("SCALE-1", "PASS" if res["verdict"]["PASS"] else "KILL")


if __name__ == "__main__":
    main()
