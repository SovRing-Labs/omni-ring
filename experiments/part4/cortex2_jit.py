#!/usr/bin/env python3
"""CORTEX-2 step 2 gate G2 (pre-registered in CORTEX-2 plan (unpublished) before this harness existed):
the generated (JIT-rendered) sift kernel must equal the hand-written AVX2 kernel bit-for-bit on the SuperSeed golden set
and on a head-sized lattice (8,738 x 5,120); compile <= 50 ms; steady-state ms/query within 1.5x of the AVX2 kernel
(reported either way). KILL: any score mismatch."""
import json, os, sys, time
import numpy as np

sys.path.insert(0, ".")
from omni_ring import seed  # noqa: E402
from omni_ring import jit  # noqa: E402
from omni_ring.vm import VM, ProgramCodebook, iid  # noqa: E402

OUT = os.path.expanduser("~/gpurun-out-0930/cortex2_jit.result.json")


def main():
    labels = [f"golden-{i}" for i in range(16)] + ["", "ümlaut", "E:Paris", "R:P36"]
    gold = np.stack([seed.codeword(s, 0) for s in labels])
    g = np.random.default_rng(0)
    big = g.integers(0, 16, (8738, seed.D)).astype(np.uint8)        # head-sized lattice (d5.verb shape)
    big[:20] = gold
    f, tcomp = jit.compile_sift(seed.D)
    print(f"compile {tcomp * 1000:.1f} ms  kernel {jit.kernel_hash(seed.D)}  cpu features avx2", flush=True)
    mism = 0
    for i in range(16):
        a = jit.sift_jit(gold[i], gold); b = seed.sift_phase(gold[i], gold)
        mism += int(not np.array_equal(a, b))
    for i in range(16):
        a = jit.sift_jit(gold[i], big); b = seed.sift_phase(gold[i], big)
        mism += int(not np.array_equal(a, b))
    # random queries too
    for _ in range(50):
        q = g.integers(0, 16, seed.D).astype(np.uint8)
        a = jit.sift_jit(q, big); b = seed.sift_phase(q, big)
        mism += int(not np.array_equal(a, b))
    # timing
    q = gold[0]
    for fn in (jit.sift_jit, seed.sift_phase):
        fn(q, big)
    t0 = time.time(); [jit.sift_jit(q, big) for _ in range(50)]; tj = (time.time() - t0) / 50 * 1000
    t0 = time.time(); [seed.sift_phase(q, big) for _ in range(50)]; ta = (time.time() - t0) / 50 * 1000
    t0 = time.time(); [seed.sift_phase(q, big, backend="numpy") for _ in range(5)]; tn = (time.time() - t0) / 5 * 1000
    # the VM running on the generated kernel
    cb = ProgramCodebook(seed.D)
    vm = VM(gold, cb)
    vm.backend = "jit"
    seed_sift = seed.sift_phase
    seed.sift_phase = lambda qq, lat, backend="auto": jit.sift_jit(qq, lat) if backend == "jit" else seed_sift(qq, lat, backend)
    vm.load_program(cb.write([iid("LOAD", 0), iid("SIFT", 7), iid("THRESHOLD", 4)]))
    vm_ok = 0
    for i in range(16):
        vm.set(0, gold[i]); m = vm.tick(); vm_ok += int(m.topk[0] == i and np.array_equal(vm.scores, seed_sift(gold[i], gold)))
    seed.sift_phase = seed_sift
    res = {"compile_ms": tcomp * 1000, "mismatches": mism, "of": 82, "ms_jit": tj, "ms_avx2": ta, "ms_numpy": tn,
           "ratio_jit_over_avx2": tj / ta, "vm_on_jit_ok": vm_ok, "kernel_hash": jit.kernel_hash(seed.D),
           "verdict": {"PASS": mism == 0 and tcomp <= 0.05 and vm_ok == 16, "bitexact": mism == 0,
                       "compile_le_50ms": tcomp <= 0.05, "speed_within_1.5x": tj <= 1.5 * ta}}
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"G2: mismatches {mism}/82 | jit {tj:.2f} ms vs avx2 {ta:.2f} ms (numpy {tn:.1f}) ratio {tj / ta:.2f} | VM on jit {vm_ok}/16 | {res['verdict']}")


if __name__ == "__main__":
    main()
