# AGENTS.md — omni-ring

For coding assistants (and people) setting this repository up. Plain steps; every command runs from the repo root.

**Status: pre-alpha, under development.** Interfaces, numbers and file layout can change without notice. Read `STATUS.md` first: what works today (each line names its test and evidence file), what is not yet tested, the known failures and the corrections.

## What it is
Residue hyperdimensional computing engine: coprime-ring (CRT) memory with a check ring, resonator decode, two-faced ternary/phase hypervector slot, AVX2 C kernel, shards and a shared-memory bus.

## Install and check (3 commands)
```bash
git clone --depth 1 https://github.com/SovRing-Labs/omni-ring && cd omni-ring
python3 -m venv .venv && . .venv/bin/activate && python -m pip install -q numpy pytest
./build_kernel.sh && python3 -m pytest -q   # expect: passed
```
Needs: python3>=3.10, numpy, pytest, gcc, cpu:avx2.
Whole suite instead: `bash install.sh --no-gui` (python check → venv → every runnable `suite.json` test, headless; without `--no-gui` it then opens the local results page on 127.0.0.1).
Measured: 233.8s on a fresh clone (Intel i5-8365U laptop, 2026-10-16, exit 1).
Note: tests/test_ring_bank.py::test_ring_bank_cross_attention_scan is a speed bar (< 800 us per scan with the AVX2 kernel built); on a fresh clone at load 40 on 8 threads (2026-10-16) it measured 935 us and failed while the other 189 passed; on an idle machine not yet measured.

## What the tests mean
`suite.json` lists every claim this repository makes with its verify command, the expected output, what passing means, how the number was obtained and how to use it in your own work (12 tests, in `suite.json` at the repository root). `STATUS.md` says what works today, what is not yet tested and the known failures; the umbrella's `STATUS.md` has one row per repository.

## Where the evidence is
- `DOSSIER.md` — history, metrics and limits
- `TECHNICAL-DISCLOSURE.md` — the method, as dated prior art
- `CITATION.cff` — DOI 10.5281/zenodo.23282029

## Honest limits (from DOSSIER.md)
A cue is mandatory for resonance; unguided decoding is seed-dependent and abstains when it fails. Resonant capacity ≈ 12 marks per slot at one cue (the exhaustive sift reads more). Superposition is finite: triangle buckets are safe to 256 facts; beyond that, truth must live in exact rows. Exact multi-hop lookups are ~250x faster in an indexed database (Part 2 §9).

## Licence
PolyForm Small Business 1.0.0 OR PolyForm Noncommercial 1.0.0 (free trial for evaluation). Details: `LICENSING.md`.

## Guided setup through MCP
The SovRing umbrella ships an MCP server that explains, checks readiness, clones and runs these tests: https://github.com/SovRing-Labs/SovRing (README → "Paste this to your assistant"). Zero-install docs endpoint: https://gitmcp.io/SovRing-Labs/omni-ring
