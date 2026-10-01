# OMNIRING — Coprime Resonator Hyperdimensional Engine

> *"We are resonators; everything we build then becomes another resonator to build off from. Develop us well."*

`omni-ring` is a production-grade, zero-external-dependency (NumPy + standard library + AVX2 C-kernel) implementation of the **OMNIRING Coprime Resonator Hyperdimensional Engine** and the **Two-Faced Polymorphic Bus** with shared-memory (`/dev/shm`) IPC.

---

## Disclosures (published as prior art)

- [`TECHNICAL-DISCLOSURE.md`](TECHNICAL-DISCLOSURE.md) — Part 1: coprime residue ring memory, check ring, cued resonator, peeling, facets, mark store.
- [`TECHNICAL-DISCLOSURE-PART-2.md`](TECHNICAL-DISCLOSURE-PART-2.md) — Part 2: the SuperSeed portable spec, geometry of the sift (cubes, pyramids, spheres, lattices, golden-ratio frequencies, programs as vectors), integrated-GPU execution, the triangle memory (omnitri), and the table / full-text / ring / matrix substrate — with all measured results, including failures.

Code for Part 2 follows in weekly updates.

---

## Architecture Overview

OMNIRING implements **Residue Hyperdimensional Computing (Residue HDC)** with a **Two-Faced Polymorphic Bus**:

1. **Two-Faced Polymorphic Slot (2,560 Bytes):**
   - **Ternary Face ($D = 10,240$, 2 bits/dim):** 160 uint64s `sign` + 160 uint64s `active`. Used by the **Reflex Layer** and **Omnitool** for instantaneous Hamming/popcount matching, binary masking, and associative routing.
   - **Phase Ring Face ($D = 5,120$, 4 bits/dim):** Packed $K = 16$ nibbles ($2$ dimensions per byte = $2,560$ bytes). Used by the **Resonator Engine** for lossless unbinding, timing-chain tracking, and continuous soft settling.
   - **IQ Constellation Transcoding:** 10,240 trits pair up into 5,120 complex $(I, Q) \in \{-1, 0, +1\}^2$ phasors, mapping directly to even phase angles $\{0, 2, 4, 6, 8, 10, 12, 14\}$ on the 16-phase circle with 100% reversible fidelity.

2. **Coprime Ring Decomposition:**
   - Base data rings: $m \in \{3, 5, 7, 13\}$ $\implies$ Data capacity $N_{\text{info}} = 3 \cdot 5 \cdot 7 \cdot 13 = 1,365$ orthogonal states.
   - Redundant check ring: $m_{\text{check}} = 17$ $\implies$ Total state space $M = 23,205$ states.

3. **Additively-Scaled Probing & CRT:**
   - Resolves $23,205$ states with just $3 + 5 + 7 + 13 + 17 = 45$ dot-product probes via Chinese Remainder Theorem (CRT) reconstruction in $< 0.35\text{ ms}$.

4. **Continuous Soft Resonator:**
   - Power-iteration soft settling to untangle superposed/crowded memory slots without falling into local attractor traps or the global barycenter.
   - Clamping known factors (e.g. 13-ring) enables high recovery ($\ge 60\%$) even in dense 12-mark superpositions.

5. **100% Single-Ring Error Detection:**
   - Using the 17-check ring ($17 > \max(3,5,7,13)$), out-of-bounds CRT detection catches all $54,600$ possible single-ring corruptions ($0$ false negatives).

6. **Hardware AVX2 Kernel:**
   - Uses `_mm256_shuffle_epi8` (`vpshufb`) for 1-cycle table lookups into a 16-byte cosine table.
   - Achieves **$465\text{ ns}$** per vector similarity in C, and **$671\text{ ns}$** in Python batch.

7. **Zero-Copy /dev/shm IPC with Seqlock Protection:**
   - Full compatibility with the fleet's `/dev/shm/vsa_matrix_bus` and `/dev/shm/vsa_matrix_seq` (1,000 slots $\times$ 2,560 bytes).

---

## Directory Structure

```
packages/omni-ring/
├── c_src/
│   ├── omniring_avx2.h          # C header for AVX2 SIMD routines
│   ├── omniring_avx2.c          # AVX2 vpshufb 1-cycle lookup & vector arithmetic
│   ├── omniring_polymorphic.h   # C header for polymorphic slot & transcoders
│   └── omniring_polymorphic.c   # AVX2/popcount ternary & phase transcoding
├── omni_ring/
│   ├── __init__.py              # Unified exports & thread governor
│   ├── types.py                 # RingSpec, PhaseVector, ResidueTuple, ResonatorResult
│   ├── codebook.py              # K=16 phase codebooks and rotation generators
│   ├── chain.py                 # Timing chain, Hadamard binding/unbinding, advance
│   ├── resonator.py             # Multi-ring continuous soft-settling power iteration
│   ├── crt.py                   # Chinese Remainder Theorem & 45-probe direct solver
│   ├── check.py                 # 17-ring anomaly detector (100% error catch)
│   ├── shm_bridge.py            # /dev/shm memory-mapped zero-copy IPC bridge
│   ├── avx2.py                  # Ctypes bridge for compiled AVX2 shared object
│   ├── polymorphic.py           # TernaryVector, PhaseVectorPacked, PolymorphicSlot/Bus
│   └── libomniring_avx2.so      # Compiled -O3 -mavx2 shared library
├── tests/
│   ├── test_codebook.py         # Codebook generation & unit modulus tests
│   ├── test_chain.py            # Binding/unbinding inverse & homomorphism tests
│   ├── test_resonator.py        # Resonator convergence & clamped recall tests
│   ├── test_crt.py              # CRT solver & exhaustive 54,600 error check tests
│   ├── test_shm_bridge.py       # IPC read/write roundtrip tests
│   ├── test_avx2.py             # AVX2 nibble-packed bind/unbind & latency tests
│   ├── test_polymorphic.py      # Polymorphic slot, transcoding, & seqlock tests
│   └── test_gates.py            # All 6 required numerical acceptance gates
├── pyproject.toml               # Standalone pip-installable configuration
└── README.md                    # Comprehensive documentation
```

---

## Build

```bash
./build_kernel.sh          # compiles omni_ring/libomniring_avx2.so (x86-64 + AVX2, gcc/clang with OpenMP)
pip install -e .
```

Without the kernel the package imports and runs NumPy fallbacks; kernel-only tests fail.

Licence: PolyForm Small Business 1.0.0 **or** PolyForm Noncommercial 1.0.0, your choice (`LICENSING.md`). Your data and shards: `DATA-AND-SHARDS.md`; shard format (CC BY 4.0): `SHARD-FORMAT.md`. Provenance and cited prior art: `PROVENANCE.md`.

## Verification & Benchmarks

Run all 33 tests:
```bash
cd packages/omni-ring
pytest tests/ -v
```

### Verified Gate Results

| Test / Gate | Target / Requirement | Measured Output | Status |
|---|---|---|---|
| **1. Separation Gate** | $\max\|\text{cross-corr}\| < 0.04$ ($N=1,000$ entries) | **0.0383** | **PASS** |
| **2. Cycle Gate** | 1st repeat $t=1,365$ (4 rings); $t=23,205$ (5 rings); $\text{enc}(x)\otimes\text{tick} = \text{enc}(x+1)$ | Exact cycle match, $0$ phase drift | **PASS** |
| **3. Resonator Gate** | $100\%$ ($105/105$) clean 3-5-7 convergence $\le 20$ iters | **105/105** ($100\%$) | **PASS** |
| **4. Clamped Recall Gate** | $\ge 60\%$ recall at 12 marks (vs $\le 20\%$ unguided) | **68% clamped** vs **12% unguided** | **PASS** |
| **5. Check Ring Gate** | $0$ undetected errors over $54,600$ single-ring corruptions | **0 / 54,600 missed** ($100\%$ detection) | **PASS** |
| **6a. Direct Solve Latency** | $< 1.0\text{ ms}$ on CPU for 45 probes + CRT | **0.342 ms** ($3\times$ faster) | **PASS** |
| **6b. Resonator Latency** | $< 8.0\text{ ms}$ on CPU (15-iteration solve) | **4.239 ms** ($1.9\times$ faster) | **PASS** |
| **6c. SHM IPC Latency** | $< 50\text{ }\mu\text{s}$ roundtrip write + read | **4.410 }\mu\text{s}$ ($11\times$ faster) | **PASS** |
| **7. AVX2 Kernel Latency** | $< 1.25\text{ }\mu\text{s}$ per 5,120-dim codevector | **0.465 }\mu\text{s}$ (C) / **0.671 }\mu\text{s}$ (Python batch) | **PASS** |
| **8. Transcoder Reversibility** | $100\%$ lossless roundtrip for active constellations | **100% (5,120/5,120 pairs)** | **PASS** |
| **9. Seqlock Concurrency** | Zero torn reads across shared memory | **Verified** | **PASS** |
