#!/usr/bin/env python3
"""TILE-1 (PREREGISTRATIONS-OMNITRI.md wave 4): the seam-matched hexahedral tile vs today's (32, 64), on the 45 MB head."""
import ctypes, itertools, json, os, sys, time
import numpy as np
sys.path.insert(0, ".")
import omni_ring.seed as S
from omni_ring.heads import RingHead

lib = S._clib()
gomp = ctypes.CDLL("libgomp.so.1")
h = RingHead.load("<your-data>/data/ring-heads/d5.verb_log")
lat = np.ascontiguousarray(h.marks, np.uint8)
q = np.ascontiguousarray(S.encode(np.load("<your-data>/data/ring-heads/d5.verb_log/eval_X.npy").astype(np.float32)[:256], h.proj, h.centre), np.uint8)
n, d = lat.shape


def run(tq, tm):
    out = np.empty((len(q), n), np.int32)
    lib.omniring_sift_batch_tiled(lat, n, d, q, len(q), out, tq, tm, 1)
    return out


def best(tq, tm, reps=3):
    run(tq, tm); b = 1e9
    for _ in range(reps):
        t = time.perf_counter(); run(tq, tm); b = min(b, time.perf_counter() - t)
    return b


ref = run(32, 64)
res = {"lattice": [n, d], "queries": len(q)}
for threads in (1, 4):
    gomp.omp_set_num_threads(threads)
    grid = {}
    for tq, tm in itertools.product((4, 8, 16, 32, 64), (8, 16, 32, 64, 128, 256)):
        assert np.array_equal(run(tq, tm), ref), f"not bit-identical at {(tq, tm)}"
        grid[f"{tq}x{tm}"] = best(tq, tm)
    cur, der = grid["32x64"], grid["8x32"]
    bk = min(grid, key=grid.get)
    res[f"{threads}t"] = {"ms_per_query": {k: v * 1000 / len(q) for k, v in grid.items()},
                          "current_32x64": cur * 1000 / len(q), "derived_8x32": der * 1000 / len(q),
                          "grid_best": bk, "grid_best_ms": grid[bk] * 1000 / len(q),
                          "derived_speedup": cur / der, "best_speedup": cur / grid[bk]}
    r = res[f"{threads}t"]
    print(f"TILE-1 {threads}t: current 32x64 {r['current_32x64']:.3f} ms/q | derived 8x32 {r['derived_8x32']:.3f} "
          f"({r['derived_speedup']:.2f}x) | grid best {bk} {r['grid_best_ms']:.3f} ({r['best_speedup']:.2f}x)", flush=True)
res["verdict"] = {"bit_identical_all": True,
                  "PASS": res["1t"]["derived_speedup"] >= 1.10 and res["4t"]["derived_speedup"] >= 1.10}
print("TILE-1:", res["verdict"], flush=True)
json.dump(res, open(os.path.expanduser("~/gpurun-out-0930/tile1.result.json"), "w"), indent=1)
