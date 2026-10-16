> **Status: under development — pre-alpha.** Interfaces, numbers and file layout can change without notice.

# omni-ring — status

Last checked: 2026-10-16, Linux laptop (Intel i5-8365U, 8 threads), on a clean copy of this repo. 11 of 12 suite tests work today; 11 of 12 can run without special hardware. Each line below names its test id in `suite.json` and the evidence file it rests on.

## Works today

- **O1** The five numerical acceptance gates (separation, cycle, resonator, clamped recall, check ring) — PASS 2026-10-16 (`5 passed, 1 deselected in 19.13s`); evidence: `README.md`, `tests/test_gates.py`.
- **O2** Latency budgets: direct solve, resonator, AVX2 per-vector, ternary GEMV, ring-bank scan — PASS 2026-10-16 (`4 passed in 1.05s`); evidence: `README.md`, `tests/test_gates.py`, `tests/test_avx2.py`, `tests/test_ternary_gemv.py`, `tests/test_ring_bank.py`.
- **O3** Core ring maths and the golden SuperSeed files (codebook, bind/unbind, CRT, resonator, seed v1 and v1.1) — PASS 2026-10-16 (`33 passed in 14.42s`); evidence: `tests/golden/seed_v1.json`, `tests/golden/seed_v1_1.json`, `tests/test_seed.py`, `tests/test_crt.py`, `README.md`.
- **O4** AVX2 C kernel, polymorphic slot, gram hash and sift/solids agree with the NumPy reference — PASS 2026-10-16 (`42 passed, 2 deselected in 11.04s`); evidence: `README.md`, `c_src/omniring_avx2.c`, `c_src/omniring_polymorphic.c`, `c_src/omniring_sift.c`, `tests/test_polymorphic.py`, `tests/test_sift.py`.
- **O5** AVX-512 sift and popcount kernel (VNNI / VPOPCNTDQ) bit-identical to AVX2 — PASS 2026-10-16 (`2 passed, 3 skipped in 1.15s`); evidence: `README.md`, `tests/test_avx512.py`, `c_src/omniring_avx512.c`, `build_kernel_avx512.sh`.
- **O6** PEEL cued multi-sweep readout recovers marks from crowded slots — PASS 2026-10-16 (`19 passed in 89.36s (0:01:29)`); evidence: `tests/test_peel.py`, `TECHNICAL-DISCLOSURE.md`.
- **O7** Program VM and five-line tick loop: tampered or public programs are refused — PASS 2026-10-16 (`13 passed in 67.64s (0:01:07)`); evidence: `tests/test_vm.py`, `tests/test_loop.py`.
- **O8** Storage layer: shared-memory bridge, query bus, store, mark store, shards, retrieval, telemetry — PASS 2026-10-16 (`48 passed in 18.90s`); evidence: `SHARD-FORMAT.md`, `DATA-AND-SHARDS.md`, `tests/test_shards.py`, `tests/test_mark_store.py`.
- **O9** Learning core, gearbox, belt, ring transformer, ring bank, frames, gap style and logic — PASS 2026-10-16 (`24 passed, 1 deselected, 1 warning in 9.08s`); evidence: `tests/test_ring_transformer.py`, `tests/test_gearbox.py`, `tests/test_core_learn.py`, `TECHNICAL-DISCLOSURE-PART-3.md`.
- **O10** Hypervector wire on 24,495 real tool steps (HV-0 result file) — PASS 2026-10-16 (`HV-0 PASS 24495 4506 2560`); evidence: `experiments/results/hv0.result.json`, `experiments/harness/hv0.py`, `TECHNICAL-DISCLOSURE-PART-3.md`.
- **O11** Shared-memory IPC round trip (IPC-0): resident C worker vs spawned process — PASS 2026-10-16 (`IPC-0 PASS median_us 0.892 spawn_over_shm 36237`); evidence: `experiments/results/ipc0.result.json`, `experiments/harness/ipc0.py`, `experiments/PREREGISTRATIONS-OMNITRI.md`.

## Not yet tested

- **O12** Twin-drift gate between the package and the authors' project tree — external: needs something the test laptop does not have; needs: the authors' private project tree at ~/projects/omniring-llm

## Known failures and honest negatives

- Quick test `./build_kernel.sh && python3 -m pytest -q` on a fresh copy, load 40.67: **FAIL** — last line `1 failed, 189 passed, 5 skipped in 214.90s (0:03:34)`.
- Speed budgets (O2) depend on load: on 2026-10-16 O2 failed twice at load average ~30 and passed at load 4 (`4 passed`); the census quick-test failure below is the same ring-bank scan budget (935 µs vs 800 µs at load 40).
- `tests/test_twin_drift.py` compares against a second checkout that exists only on the author's machine; it skips elsewhere.

## Corrections

- **C02** — SIEVE-0: Part 4 already states the 1 ms bar was missed (correct as published).
- **C04** — RESON-1b "KILL" against Kymn 2025 (Part 4): the comparison is invalid — the paper's C(D) curve fixes K = 2 moduli, the run used K = 3; the K = 2 re-run is owed. Erratum pending.

## How to re-check

- One file, any python3 >= 3.8: download [`sovring-suite.pyz`](https://github.com/SovRing-Labs/SovRing/releases/latest/download/sovring-suite.pyz), then `python3 sovring-suite.pyz --run omni-ring` (headless) or `python3 sovring-suite.pyz` (local page on 127.0.0.1).
- From this checkout: `bash install.sh` (checks python, prepares a venv, runs the suite headless, then opens the page; `--no-gui` to skip the page).
- Quick test only: `./build_kernel.sh && python3 -m pytest -q tests/test_gates.py tests/test_crt.py tests/test_seed.py`.

## Defensive publication

- `TECHNICAL-DISCLOSURE.md` — on public GitHub since 2026-09-29
- `TECHNICAL-DISCLOSURE-PART-2.md` — on public GitHub since 2026-10-01
- `TECHNICAL-DISCLOSURE-PART-3.md` — on public GitHub since 2026-10-10
- `TECHNICAL-DISCLOSURE-PART-4.md` — on public GitHub since 2026-10-10
- DOI: [10.5281/zenodo.23282029](https://doi.org/10.5281/zenodo.23282029) (Zenodo, see `CITATION.cff`)
- Licence: PolyForm Small Business 1.0.0 / PolyForm Noncommercial 1.0.0 (LICENSING.md); PolyForm Free Trial for evaluation
