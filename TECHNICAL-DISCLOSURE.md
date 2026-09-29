# OMNIRING — Technical Disclosure

**Title:** Coprime residue ring memory with a redundant check ring, cue-driven soft resonator readout, explaining-away
peeling, facet ("diamond") systems with a checksum centre, count-weighted decaying mark store, and distributed shard
querying — on commodity CPUs.
**First public disclosure:** the date of the first public commit of this file. **Licence of the code:** PolyForm Small
Business 1.0.0 or PolyForm Noncommercial 1.0.0, user's choice (`LICENSING.md`). **Intent:** this document is published so that everything it describes is prior art. The
authors claim no patent on it and ask that none be granted to anyone on what is disclosed here.
**Authors:** see `LICENSE.md` Required Notice. Cited prior work: `PROVENANCE.md`.

This is an enabling description: a practitioner can rebuild the system from this text alone. Every number in section 12
was measured on a 2019 laptop CPU (Intel i5-8365U, AVX2, no GPU) with the code in this repository or its research harness.

---

## 1. Summary

A memory that stores small structured records ("marks") in fixed-size high-dimensional vectors and reads them back by
*resonance* rather than by lookup tables or gradient-trained weights.

* Each mark is an integer coordinate `x` in `[0, N)` with `N = 3 · 5 · 7 · 13 = 1,365`, written as the elementwise
  product (binding) of one codevector per **coprime ring** (moduli 3, 5, 7, 13) plus one per **check ring** (modulus 17).
* Several marks are superposed (summed) into one slot vector.
* A query supplies a **cue** (the known position of at least one ring). A **soft resonator** settles all free rings together
  and returns the ring positions; the **Chinese Remainder Theorem** (CRT) turns them into a coordinate; the **check ring**
  verifies it (any single-ring slip lands outside `[0, N)`), and the system **abstains** instead of answering when it fails.
* Crowded slots are read one mark at a time by **peeling**: accept a verified mark, subtract its codevector, settle again.
* Richer records bind several **facet systems** (e.g. what / who / where / when) with a **checksum centre ring**.
* Marks accumulate in a **mark store** with a rehearsal counter and **decay**, so recent and repeated evidence dominates.
* Across devices, each machine holds a **shard**; a cue is broadcast; only answers that pass the check ring may win.

## 2. Representation

* **Phase face.** A vector is `D = 5,120` unit-modulus complex numbers. Each phase may be stored at `K = 16` levels
  (4 bits), so a vector packs into 2,560 bytes (one bus slot). Binding = elementwise product (phase addition mod `K`);
  unbinding = product with the conjugate; similarity = mean real part of `a · conj(b)`.
* **Ternary face.** The same 2,560 bytes can be read as 10,240 ternary dimensions `{−1, 0, +1}` in two bitplanes
  (sign, active) for popcount similarity. A total transcode maps each ternary pair to one phase nibble:
  the 8 active pairs map bijectively to the 8 even phases `{0,2,…,14}`; the neutral pair `(0,0)` maps to phase 0
  (the additive identity) — a null ternary block is a null phase block. No pair is left undefined.
* Quantising phases to `K = 16` cost nothing measurable in recall (section 12).

## 3. Coprime counting rings and the timing chain

* For each modulus `m`, draw a per-dimension angle `θ_m[d] = j · 2π/m` with `j` uniform in `{0,…,m−1}`. Position `k` of
  ring `m` is `exp(i · k · θ_m)`. Rings: 3, 5, 7, 13 (information) and 17 (check). Pairwise coprime moduli make the
  combined position repeat only after the product of the moduli, so every coordinate below `N` has a unique signature.
* `encode(x) = Π_m exp(i · (x mod m) · θ_m)` over all rings.
* **Timing chain.** One tick multiplies every ring forward one position at once: `encode(x) ⊙ tick = encode(x+1)`, with
  `tick = Π_m exp(i θ_m)`. The combined reading is a clock with period `N` (information rings) or `N·17` (all rings).
  The clock is a cue, not a payload: it stamps *when* a mark was written.
* **Check ring.** Modulus 17 ≥ the largest information modulus. For legal coordinates (`x < N`) the check residue is fully
  determined by the others. Any single-ring error moves the CRT reconstruction outside `[0, N)`, so it is detected:
  verified exhaustively over 54,600 single-ring corruptions (100% detection). About 94% of the combined 5-ring space is
  deliberately illegal; a wrong settle almost always lands there.
* Coprimality is what prevents early realignment (degeneracy): rings with a shared factor (e.g. 6 and 9) would realign at
  their LCM and make distinct coordinates look identical.

## 4. Soft resonator readout (the "resonance")

Given a slot vector `v` and codebooks `C_i` (rows = ring positions):

1. Each free ring starts as the mean of its codebook; a **clamped** ring (the cue) is fixed at its known position.
2. For each free ring `i` in turn: `u = v ⊙ Π_{j≠i} conj(est_j)` (unbind the other rings' current estimates);
   scores `s = Re(C_i^* u) / D`; weights `w = max(s, 0)` (negative evidence cannot propagate — a mild lateral inhibition);
   `est_i = normalise_to_unit_modulus(w · C_i)`.
3. Repeat until the argmax positions are stable; report positions, iterations, and the confidence gap
   (best minus runner-up score). The gap predicts correctness (section 12).
4. CRT → coordinate; check ring → **VERIFIED**, **REJECTED** (illegal), or **ABSTAIN** (not converged).

**A cue is required.** With rotation codebooks the whole chain collapses to a single rotation, and factorising it with no
cue is marginal at `D = 5,120`: depending on the random draw, unguided decoding of a clean single mark succeeds 15/15 or
1/15. With any one ring clamped it succeeded at every draw tested. Unguided failures abstain rather than answer wrongly.
A helper (`codebook.verify_unguided`) measures a draw before any unguided use.

**Why it is not brute force.** The resonator compares against `3 + 5 + 7 + 13 (+17)` codevectors per pass, never the
`1,365` (or `23,205`) combinations: cost grows with the *sum* of ring sizes, not their product.

## 5. Direct (keyed) storage for single marks

When a slot holds exactly one mark, store it as the **sum** (not product) of one codevector per ring, each ring bound to
its own random unit **key** (without keys, residue codebooks share position 0 = the all-ones vector and collide). Read
each ring by argmax over its keyed codebook (45 probes for 3+5+7+13+17) and apply CRT + check: exact, about 11× faster than
resonating. Two marks can share a slot with a **1-bit tag**: bind mark A with a random unit tag `T`, mark B with `conj(T)`;
unbind the tag to read each.

## 6. Peeling (explaining away) for crowded slots

1. For each value `c` of the cue ring (e.g. the 13 ring): settle with that clamp.
2. If the settle is legal under the check ring, accept `x` and subtract `encode(x)` from the slot.
3. After one sweep, sweep again over the residual.

Subtraction removes the loudest marks so the quieter ones become readable; the check ring decides what may be subtracted.

## 6b. Exhaustive lattice sift and line firings

* **Exhaustive sift.** When the legal lattice is small enough to enumerate (here `N = 1,365`), a slot is read in one pass by
  comparing it with every legal codeword (`Re(W^* v)/D`, one matrix product) and keeping those above a threshold (0.5).
  Codewords of distinct coordinates are nearly orthogonal (cross-similarity sd ≈ 1/√(2D)), so a slot holds hundreds of marks
  read this way; the ≈ 12-mark figure in §12 is the limit of *cued resonance*, not of the slot.
* **Line firings.** Between exhaustive sift and a single cued settle lies a family of *lines*: each line is a resonator with a
  chosen subset of rings clamped to one value each (a lattice line of the torus formed by the rings). All lines are settled in
  one batched firing and filtered by the check ring. More clamped rings → more lines, fewer free factors per line, higher
  recall. Rule: enumerate what the budget allows, resonate the rest.
* **Keep the magnitude for crowded slots.** A slot quantised to phase only stops separating members from non-members near
  192 marks; kept as int8 real/imag parts (10 KB per slot at D = 5,120) it reads 192/192 and 383/384. Codewords stay 4-bit phases.
* **Integer sift kernel.** Lattice rows hold one 4-bit phase index per dimension; the slot is int8 real/imag. Per 32 dimensions:
  two byte-shuffle table lookups (127·cos, 127·sin of the phase index), sign transfer onto the query, unsigned×signed
  multiply-add into 16-bit pairs, widening add into 32-bit accumulators; rows are scored in parallel threads. Identical to the
  floating reference computed in the same integer arithmetic.
* **Star arrays.** Records bind several small facet systems (the points) plus a checksum ring (the centre). Each point keeps
  its own slot of facet values and sifts it; the centre tests only the product of the points' survivors against the bound
  record slot and the checksum. Cores are chosen per checkpoint by lattice size and fill: sift (small), line firing
  (medium), cued resonance (too large to enumerate), warm-start tracking (streams). The lattice may be precomputed and held in
  memory (or memory-mapped), making a read one sequential stream serving any number of queries.
* **Warm start (tracking).** A resonator that starts each settle from its previous settled state keeps a cued mark through
  growing clutter and releases it when the mark is removed.

## 7. Dispersal, slot selection and facet ("diamond") systems

* Recall inside one slot falls with load (about 12 marks per slot at one cue). Spreading marks over many lightly loaded
  slots keeps in-slot recall high (≈94% given the right slot), but choosing *which* slot answers is the bottleneck when
  every slot looks alike.
* **Facet systems.** A record binds one value from each of several small categorical codebooks (e.g. four "corners"),
  plus a **centre checksum ring** whose position is `(sum of corner indices) mod 17`. Known corners are cues; the centre
  flags wrong settles. Corner size is the lever: small corners (≈15 values) read reliably; large corners (≈105) or corners
  split into many sub-rings (more free factors) do not.
* The same checksum idea applies at every scale: rings within a system, systems within a record, shards within a fleet.

## 8. Continuous dials (fractional power encoding)

A continuous quantity `t` (time, angle) is encoded as `exp(i · ω · t)` with per-dimension frequencies `ω` drawn uniformly in
`(−2π/L, 2π/L)` (similarity between nearby `t` falls off over about `L` ticks). A dial is best used as a **cue** (settle
"around when"), not as a factor to be discovered in a crowded slot. A symbol × dial record (e.g. the same symbol at many
rotations) is read by unbinding the symbol and scanning the dial; a coarse pass using only low-frequency dimensions on a
coarse grid, followed by a fine local scan, is an equivalent and cheaper variant.

## 9. Mark store: reinforcement, rehearsal counter, decay

* Per slot: an `int16` accumulator (saturating at ±32,767), a `uint32` rehearsal counter, and a last-write time.
* `add(slot, mark, weight)` accumulates the mark's ternary vector; the counter is a **sidecar**, never part of the vector —
  phases quantised on write cannot carry a count.
* `decay(factor)` multiplies accumulators and rounds toward zero: a mark reinforced once disappears before a mark reinforced
  five times. Nothing decides what to forget; the magnitude is the evidence. `prune` clears empty slots.
* **Lane resonance.** One slot per worker lane; each settle adds an "ok" or "fail" prototype; resonance =
  `(sim(read(slot), P_ok) − sim(read(slot), P_fail))` gives a decaying, per-lane health signal for routing.

## 10. Reading the bus from a small model

A small (e.g. ternary) transformer reads the bus through cross-attention over a bank of named slots (goal, temporal,
entity context, system state, action attractor, immune filter, working registers). The full slot is reduced to the model
width by a fixed, seeded ±1 projection — never by truncation, which discards most of each holographic vector.

## 11. Portals and distributed shards

* One writer per device publishes the ring state to shared memory under an exclusive lock and streams it to readers
  (HTTP / server-sent events). Apps are readers, never writers.
* Across machines, each device writes only its own shard. Because encoding is deterministic from a shared codebook seed,
  devices exchange **marks** (a few bytes), not vectors, and rebuild vectors locally. A cue is broadcast; each shard
  answers only with check-ring-legal results; the strongest verified answer wins.

## 12. Measured results (this repository and its harness)

| Property | Result |
|---|---|
| Single-ring slip detection by the 17 check ring | 100%, exhaustive over 54,600 cases |
| Clean single-mark round trip | 1,365 / 1,365 |
| Cued recall, 12 marks per slot, one ring clamped | 67% (chance 8%); two clamps: 93% |
| Capacity (largest load with ≥ 50% recall, one cue) | ≈ 12 marks per slot |
| Phase quantisation to 16 levels | no measurable recall cost |
| Confidence gap vs correctness | lowest quartile 20% correct, highest 96% |
| Check ring on wrong settles in crowded slots | flags 82%, 0 false alarms |
| Exhaustive sift of all 1,365 legal codewords, threshold 0.5 | 24, 48, 96, 192 marks/slot: all read, 0 fakes; 384: 382.6 (3 fakes); 1,000: 973.7 (≈ 17 ms numpy) |
| Line firing, 91 lines (13-ring × 7-ring clamped), 24 marks/slot | 20.45 true per slot, 1.4% fakes (13 lines: 8.55) |
| Phase-only vs int8-complex slot, exhaustive sift | phase-only separable to ~96 marks, fails at 192; int8 complex 192/192, 383.1/384 (4 fakes) |
| Integer AVX2 sift, i5-8365U, 8 threads | 1,365-row lattice 0.72 ms per firing (0.525 µs/row, 7 MB); 50,625-row lattice held in RAM 25.8 ms (259 MB, ≈ 10 GB/s) |
| Star array, 4 facets × 15 + checksum, points prune → centre joins | 6 records: same recall as full 50,625 sift with 83× fewer codewords; 24 records: 2× (facets saturate); cued (2 known facets) 1.000 vs 0.950 line sift vs 0.900 resonator |
| Warm-started cued settle, slot filling to 16 marks | recall 100% vs 72.8% cold; after removal the mark is still reported 1.3% |
| Keyed direct read, single-mark slots | 1,000 / 1,000, ≈ 11× faster than resonating |
| 1-bit tag, two marks per slot | 300 / 300 (267 / 300 untagged) |
| decode_soft silent-wrong rate, 13-ring cue, k ≤ 10 | 1 in 1,200 decodes (≤ 1 / 300 per load up to k = 100); unguided failures abstain |
| Peeling, 24 marks per slot | research harness: 14.8 true marks per slot vs 8.45 for one sweep, 3.6% accepted fakes; engine `peel.read_all` (two cued sweeps, 17-legal filter): 16.65 true, 0 / 333 fakes, ≈ 367 ms per read |
| Diamond, centre checksum on wrong settles | flags 93% |
| Diamond, 15-value corners, 2 known, 12 marks | 91% |
| Codebook neighbour similarity 0.80, read-time whitening | 52% → 75% |
| Bind / dot, 10,240-D ternary, AVX2 | ≈ 35 ns / ≈ 93 ns |
| Full scan of 1,000 slots | 145 µs |

## 13. Variations also disclosed

Any number of rings and any pairwise-coprime moduli; one or more redundant check rings (detecting more simultaneous
errors); non-coprime categorical rings for facets; any dimension `D`; phase quantisation at any `K`, or unquantised;
binary, bipolar, ternary or complex phasor codevectors; permutation-based binding instead of elementwise product; any
checksum function for the centre ring (sum, weighted sum, hash) and any modulus; multiple cues; cue values drawn from a
clock, a query encoder model, a category index or a keyword index; peeling with whole-codeword or projection
subtraction; exhaustive sift over an enumerable legal lattice; batched line firings with any subset of rings clamped;
warm-started (continuously running) resonators; annealed weight sharpening; read-time whitening of correlated codebooks; decay by multiplication, by
subtraction or by age; per-slot counters of any width; one slot per lane, per user, per topic or per time bucket;
execution on CPUs (AVX2/AVX-512/NEON), GPUs, FPGAs, in-memory or analog hardware; distribution over any number of machines
exchanging marks, cues or verified answers.
