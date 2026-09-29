# OMNIRING Research Buglog & Evolutionary Feature Register

**Pillar:** The Nervous System (Cognitive Core of the Sovereign Neural OS)  
**Package:** `packages/omni-ring`  
**North Star:** *"Democratise AI — capable, reliable, verifiable AI on the computers people already own, shaped by any person or community with their own data. No trillion-token budgets. No captivity."*  
**Creed:** *"We are resonators; everything we build then becomes another resonator to build off from. Develop us well."*

---

## 1. Executive Summary & Coordination Policy

This document is the living engineering ledger for the **OMNIRING Coprime Resonator Hyperdimensional Engine**. It records every anomaly, bug, root cause, applied remediation, and queued architectural feature across both the core math substrate and its polymorphic bus integration.

**Active Dev Coordination Note:**  
The Gemini 3.8 Flash Core Dev owns engine development (`omni_ring/`, `c_src/`, `tests/`). The Commander coordinates, sets contracts, safeguards the live core, verifies gates, and bridges OMNIRING into the portals ecosystem without racing or stepping on active file edits.

---

## 2. Bug Registry & Resolution Matrix

| Bug ID | Component | Description / Symptom | Root Cause | Status |
|---|---|---|---|---|
| **BUG-001** | `chain.py` / `omniring_bridge.py` | 5-ring composite tick desyncs CRT coordinate after 1,365 cycles. | Advancing both 4 data rings and check ring with a single data tick operator. The 4 data rings repeat at $t=1,365$, but the 17-ring repeats at $1,365 \pmod{17} = 5 \neq 0$. Full cycle is $23,205$. | **FIXED** (Separate `_tick` and `_tick_check` operators). |
| **BUG-002** | `chain.py` / `crt.py` | `TypeError: 'RingSpec' object is not iterable` during `get_tick`. | `get_tick(rotations)` and `encode(x, rotations)` expect a list of phase vectors, not the `RingSpec` instance. | **FIXED** (Type annotations clarified; wrapper shims provided). |
| **BUG-003** | `test_avx2.py` | `AssertionError: AVX2 latency exceeded budget (3.238 µs >= 1.25 µs)`. | Python loop overhead calling ctypes 10,000 times for tiny 45-vector batches. The measured time was Python call frame overhead, not AVX2 SIMD compute. | **FIXED** (Benchmarked over 500-vector batch with `min(trials)`: true hardware latency is **0.641 µs**). |
| **BUG-004** | `retrieval.py` / `test_retrieval.py` | `test_17_check_gate_rejection` failed: asserted `REJECTED`, got `VERIFIED`. | 5,120-D hyperdimensional representations are so robust that flipping 3,800 dims by $+5$ left enough SNR in the remaining 1,320 dims to still decode cleanly. No residue error actually occurred! | **FIXED** (Applied exact $\pi$-inversion $+8$ on 4,200 dims, inducing clean residue slip caught by CRT check). |
| **BUG-005** | `shm_bridge.py` | Multi-process bus corruption or stale writer lock lingering after process crash. | Missing advisory lockfile (`flock`) on shared memory file creation. | **FIXED** (Implemented `Single-Writer Law` with exclusive `.lock` file and clean teardown in `omniring_bridge.py`). |
| **BUG-006** | `ring-math.ts` (Client) | TypeScript compiler errors: `Cannot find name '_' / 'x'`, missing `ChainVerdict` export. | Untyped lambda parameters and unexported interface in TypeScript port. | **FIXED** (Exported interfaces, typed parameters; `tsc --noEmit` exits 0). |
| **BUG-007** | `OmniringDial.svelte` | Svelte 5 template compilation error. | Legacy syntax `{#each MODULI as \|m, i\|}` invalid in Svelte 5. | **FIXED** (Updated to standard Svelte 5 `{#each MODULI as m, i}`). |
| **BUG-008** | Environment | `pip install -e` rejected by PEP 668 externally managed environment. | System Python prevents global pip installs without virtualenv. | **FIXED** (Created `~/.local/lib/python3.12/site-packages/omni-ring.pth` user-site linkage; zero system pollution). |

---

## 3. Detailed Bug Dossiers

### BUG-001: The 5-Ring vs 4-Ring Cycle Periodicity Trap
- **Symptoms:** When stepping a composite phase vector with `advance(v, tick, 1)`, CRT coordinate recovery worked perfectly for steps $0 \dots 60$, but after $1,365$ steps, the coordinate did not loop cleanly back to 0.
- **Deep Mathematical Analysis:**
  The 4 data rings $(3, 5, 7, 13)$ have lowest common multiple:
  $$\text{lcm}(3, 5, 7, 13) = 3 \times 5 \times 7 \times 13 = 1,365$$
  However, the 17-check ring is coprime to all four:
  $$\text{lcm}(3, 5, 7, 13, 17) = 1,365 \times 17 = 23,205$$
  If a single tick operator is derived from the product of all 5 rings, the data rings advance in lockstep with the check ring, but after 1,365 steps, the check ring is at phase $1,365 \pmod{17} = 5$. A naive 4-ring CRT solver expects the check ring to also be at 0, triggering a false-positive `ResidueDesyncError`.
- **Resolution:**
  Maintain dual rotation bases in the daemon:
  1. `_v_data = chain.advance(_v_data, _tick_data, 1)` (mod 1,365).
  2. `_v_check = chain.advance(_v_check, _tick_check, 1)` (mod 17).
  Or compute the ground-truth state coordinate $x = (x + 1) \pmod{1,365}$ and encode directly into the composite vector.

### BUG-004: Hyperdimensional Noise Immunity Masking Fault Injection
- **Symptoms:** Unit test attempting to verify that the 17-check ring rejects corrupted vectors failed because the vector was accepted as `VERIFIED`.
- **Deep Analysis:**
  In a 5,120-dimensional space quantized to $K=16$ phases, the projection dot product is:
  $$\text{sim}(u, v) = \frac{1}{D} \sum_{d=0}^{D-1} \cos(\theta_u[d] - \theta_v[d])$$
  Corrupting 3,800 dimensions with an offset of $+5$ ($+1.96$ radians) introduced noise, but the remaining $1,320$ dimensions ($26\%$ of the vector) were completely pristine. Since cross-correlation between orthogonal codevectors is $< 0.04$, the signal from the $1,320$ clean dimensions was still $\approx 0.26$, easily dominating the noise floor and correctly identifying the residue!
- **Resolution:**
  To genuinely test the 17-check ring's rejection capability, fault injection must overcome the HDC margin by applying exact phase inversion ($\pi$ radians, or $+8$ in $K=16$) across $\ge 4,200$ dimensions, or directly injecting an illegal residue tuple into `check_anomaly`.

---

## 4. Needed Features & Evolutionary Roadmap (Nervous System)

### FEATURE-001: Hardware AVX-512 SIMD Kernel
- **Context:** Hosts with AVX-512 (e.g. Intel Xeon, Skylake-X, Ice Lake, Tiger Lake, Sapphire Rapids, AMD Zen 4/5) support 512-bit ZMM registers.
- **Architecture:**
  - With `__m512i`, execute 64 byte-shuffles in parallel per instruction via `_mm512_shuffle_epi8`.
  - Process an entire 5,120-dimensional vector in only $40$ SIMD instructions.
  - Anticipated latency: **$\le 150$ nanoseconds** per vector (over $3\times$ speedup over current AVX2 kernel).

### FEATURE-002: GPU Tensor Core / Vulkan Compute Kernels for Superposition Swarms
- **Context:** For large associative memory search (100,000+ slots in `/dev/shm`), CPU SIMD reaches memory bandwidth saturation.
- **Architecture:**
  - Implement a raw Vulkan Compute shader (`omniring.comp`) and CUDA kernel.
  - Pack 4-bit nibbles into `int4` / `int8` tensor cores.
  - Execute massive parallel soft-resonator power iterations across thousands of simultaneous query vectors.

### FEATURE-003: Sliding-Window Fractional Power Encoding (FPE) for Smooth Analog Resonance
- **Context:** Residue rings provide exact discrete coordinates ($t=41$ and $t=42$ have zero cosine similarity). However, human circadian context, temperature, and semantic concepts require fuzzy continuity.
- **Architecture:**
  - Implement Fractional Power Encoding: $v(t) = v_0^{\odot t}$ where the phase rotation angle scales smoothly with real-valued $t \in \mathbb{R}$.
  - Allows the portals to query: *"What happened around 2:00 PM yesterday?"* with graceful cosine degradation based on temporal distance.

### FEATURE-004: Clamped Multi-Factor Superposition inside the Fast-Bus
- **Context:** Currently, each slot in `ShmBridge` stores a single dominant state vector.
- **Architecture:**
  - Extend the bus format to store superpositions of up to 16 concurrent concepts:
    $$S = \sum_{i=1}^{M} A_i \otimes B_i \otimes C_i$$
  - Implement hardware-accelerated continuous soft resonator power iteration directly inside the C library, returning all $M$ factor triples with their respective eigengap confidences.

### FEATURE-005: WebAssembly (Wasm) + SIMD Browser Engine
- **Context:** Portals currently receive residue tuples over SSE and run pure TypeScript math in Web Workers.
- **Architecture:**
  - Compile `c_src/omniring_avx2.c` to WebAssembly using Emscripten with `-msimd128`.
  - Use 128-bit Wasm SIMD intrinsics (`wasm_i8x16_shuffle`) inside the browser.
  - Achieves near-native SIMD resonator recall directly inside client PWAs with zero network hops.

### FEATURE-006: Bi-directional Query/Resonance Bus
- **Context:** Portals are currently read-only observers of the Fast-Bus.
- **Architecture:**
  - Implement a lockless ring-buffer query channel in `/dev/shm/omniring_query.bin`.
  - Portals write a query cue vector; the background resonator daemon resolves the bound factors and deposits the settled coordinate back into the response slot in $< 500$ microseconds.

### FEATURE-007: Seqlock Concurrency Protocol for Zero-Copy Multi-Reader Safety
- **Context:** High-frequency writers updating `/dev/shm` while slow readers read 20,480-byte phase buffers can encounter tearing.
- **Architecture:**
  - Integrate the Linux kernel seqlock pattern:
    - Writer increments sequence counter to odd before writing.
    - Writer writes data, issues memory fence (`_mm_sfence`).
    - Writer increments sequence counter to even.
    - Readers read counter before and after: if counter is odd or changed, retry.
    - 100% lockless, zero syscalls, zero tearing.
