# omni-ring changelog

## 0.2.0 (2026-10-10)
**Behaviour change: read this first.** The default codebook seed is now **0** (`codebook.SAFE_SEED`); it was **42**.
Every generator that takes `seed=` (`generate_rotations`, `generate_quantized_codebook`, `generate_direct_codebooks`,
`verify_unguided`, `retrieval.GatedMemoryEngine`) now inherits it. Why: unguided decode at D = 5,120 depends on the
seed. Seeds 0, 1 and 2 recover a clean single mark 15 of 15 times; seeds 3, 21 and 42 recover it 1 of 15 times, and
42 was the old default. All shipped decode paths are cued, so nothing broke, but any code that omitted `seed=` will now
draw different codebooks. If you saved marks written under the old default, pass `seed=42` explicitly to read them
back. `tests/test_codebook.py` guards the new default.

## Added in 0.2.0

- **SuperSeed v1 and v1.1:**
  - portable codewords (`seed.py`);
  - a shard format reader and writer (`shards.py`);
  - ring heads (`heads.py`);
  - belt and gearbox scheduling;
  - frames; gap (tie-inclusive top-k); logic, loop, study, style and telemetry modules.
- **Designed codebooks** (`solids.py`): simplex, orthogonal, cross-polytope, hypercube, and polygons via harmonic frames.
- **Kernels:**
  - a cube-tiled batch sift (`omniring_sift_tiled.c`) with an L2-fitted tile per CPU (`seed.tile_shape`, 1.18–1.33×,
    bit-identical), plus an SSE fallback kernel;
  - an optional AVX-512 kernel (`build_kernel_avx512.sh`, `omniring_avx512.c`), loaded only on CPUs that report the
    needed features and bit-identical to AVX2; `OMNI_RING_NO_AVX512=1` turns it off.
- **The sifter as code** (Part 4):
  - `vm.py`: a vector VM with 15 opcodes and keyed check slots;
  - `jit.py`: an LLVM-rendered sift kernel, bit-exact; needs `llvmlite`, which is optional, and falls back to the C
    kernel without it;
  - `vm_loop.py`: an always-on tick loop;
  - `route.py`: the decode ladder SIEVE / PART / PEEL / CHECK;
  - `core_learn.py`: the learning core;
  - `peel.read_all_kernel`: the crowded-slot read, 24 of 24 to 64 of 64 marks per slot in the saved results.
- **Store** (`store.py`): residue-addressed put and get with no scan, superposed group slots, and a 4-bit text lattice
  with a prefix cascade.
- **Experimental** (no disclosure yet; interfaces may change):
  - `osring/`: ingesters that write a machine's files, shell history, packages and logs into one store, plus a C
    gram-hash kernel (`omniring_gramhash.c`);
  - `shadow/`: a read-only shadow of an external field database; it returns empty results when that database is
    absent.
- **Disclosures** (prior art): Parts 3 and 4, with all harnesses, result files and pre-registrations in `experiments/`.
  Failures are reported as registered.

## Changed

- **VM idle ticks no longer re-sift an unchanged query.** An exact score memo cut idle-tick time from 0.58 ms at 1,000
  marks and 6.1 ms at 10,000 to about 0.025 ms at both, with identical output over a random 120-tick stream.
- **`seed.sift_batch_p4` routes four or more queries to the register-blocked JIT kernel**: coarse pass 4.9–8.3× faster,
  top-k identical. `OMNIRING_P4_JIT=0` keeps the C kernel, and any JIT failure falls back to it.

## Tests

190 passed, 5 skipped on an AVX2 laptop (`./build_kernel.sh && python3 -m pytest -q`). The skips need AVX-512 or
optional tools.
