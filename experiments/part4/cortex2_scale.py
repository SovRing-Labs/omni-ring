#!/usr/bin/env python3
"""CORTEX-2 SCALE-0 (J 2026-10-02: "run it on an extremely packed vector array, thousands, tens of thousands, 100k").
Pre-registered before running: lattices n in {1,000, 10,000, 100,000} x d 5,120 (100k = 512 MB uint8 phases), 16 golden
rows + random marks. Per n: VM in MODE 0 (exact, AVX2 kernel), MODE 1 (cascade), MODE 2 (soft), and the JIT-rendered kernel.
FIELD PREDICTS: a full scan is memory-bound, time ~ linear in n (512 MB at ~10 GB/s ≈ 50 ms/query at 100k on one socket);
cascade ≈ 3-4x less work; JIT single-thread ≈ 5x slower than the OpenMP kernel (G2).
PASS (reported bars): top-1 correct 16/16 golden queries at every n and mode; JIT == AVX2 bit-for-bit at every n (16 golden
+ 8 random queries); ms/query reported per n and mode. KILL: any JIT mismatch or a wrong top-1 in MODE 0.
Note: the CORTEX-2 step-3 loop (vm_loop.py) was running concurrently during this test.
"""
import json, os, sys, time
import numpy as np

sys.path.insert(0, ".")
from omni_ring import seed, jit  # noqa: E402
from omni_ring.vm import VM, ProgramCodebook, iid  # noqa: E402

OUT = os.path.expanduser("~/gpurun-out-0930/cortex2_scale.result.json")
NS = [1_000, 10_000, 100_000]


def main():
    labels = [f"golden-{i}" for i in range(16)]
    gold = np.stack([seed.codeword(s, 0) for s in labels])
    cb = ProgramCodebook(seed.D)
    res = {}
    g = np.random.default_rng(0)
    for n in NS:
        t0 = time.time()
        lat = g.integers(0, 16, (n, seed.D), dtype=np.uint8)
        lat[:16] = gold
        tgen = time.time() - t0
        vm = VM(lat, cb)
        r = {"gen_s": tgen, "bytes": int(lat.nbytes)}
        for mode in (0, 1, 2):
            vm.load_program(cb.write([iid("MODE", mode), iid("LOAD", 0), iid("SIFT", 3), iid("THRESHOLD", 2)]))
            ok = 0; ts = []
            for i in range(16):
                vm.set(0, gold[i]); t1 = time.time(); m = vm.tick(); ts.append(time.time() - t1)
                ok += int(m.topk[:1] == [i] and not m.refused)
            r[f"mode{mode}"] = {"top1": ok, "ms_p50": 1000 * float(np.median(ts)), "ms_max": 1000 * float(np.max(ts))}
            print(f"n {n:>7,} MODE {mode}: top-1 {ok}/16  {1000 * np.median(ts):.1f} ms/query", flush=True)
        mism = 0; tj = []; ta = []
        qs = [gold[i] for i in range(16)] + [g.integers(0, 16, seed.D).astype(np.uint8) for _ in range(8)]
        for q in qs:
            t1 = time.time(); a = jit.sift_jit(q, lat); tj.append(time.time() - t1)
            t1 = time.time(); b = seed.sift_phase(q, lat); ta.append(time.time() - t1)
            mism += int(not np.array_equal(a, b))
        r["jit"] = {"mismatches": mism, "of": len(qs), "ms_p50": 1000 * float(np.median(tj))}
        r["avx2_ms_p50"] = 1000 * float(np.median(ta))
        print(f"n {n:>7,} JIT: mismatches {mism}/{len(qs)}  jit {1000 * np.median(tj):.1f} ms  avx2 {1000 * np.median(ta):.1f} ms", flush=True)
        res[str(n)] = r
        del lat, vm
    ok_all = all(res[k]["mode0"]["top1"] == 16 and res[k]["jit"]["mismatches"] == 0 for k in res)
    res["verdict"] = {"PASS": ok_all}
    json.dump(res, open(OUT, "w"), indent=1)
    print("SCALE-0", "PASS" if ok_all else "KILL")


if __name__ == "__main__":
    main()
