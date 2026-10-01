# OMNIRING — Technical Disclosure, Part 2: geometric vector computing and the triangle memory

**Title:** A portable integer specification for phase-vector memories (SuperSeed); geometric execution of the sift
(cube tiling, coarse-to-fine pyramids, cluster spheres, lattice packings, quasiperiodic frequencies, programs as vectors,
heterogeneous CPU/iGPU execution); and a triangle memory ("omnitri") — relational facts as bound phase triangles with
three edge indexes, verification by a second path, ternary triage, decay, coherence ("harmony"), polygons, one-operation
transforms and nested recursion — integrated with a relational table, a full-text index and sparse matrix closure.
**Part 1:** `TECHNICAL-DISCLOSURE.md` (coprime residue ring memory). **First public disclosure:** the date of the first
public commit of this file. **Licence of the code:** as Part 1 (`LICENSING.md`). **Intent:** published so that everything
it describes is prior art. The authors claim no patent on it and ask that none be granted to anyone on what is disclosed
here. **Authors:** see `LICENSE.md` Required Notice. **Hardware for every number:** Intel i5-8365U laptop (4 cores / 8
threads, AVX2, 15 W, ~12 GB/s DRAM read) with integrated Intel UHD 620 (Mesa, OpenGL ES 3.1 compute), 2026-09-30/10-01.

This is an enabling description, written to let a practitioner rebuild every mechanism. Every experiment was
pre-registered (pass/kill bars written before the harness ran); failures are reported with the same weight as passes
(section 11). Private data (fleet logs, agent sessions, shards, trained marks) is not disclosed — only counts and scores.

---

## 1. Summary

1. **SuperSeed** — an integer-only specification so that phase-vector memories (codewords, projections, encodings, scores)
   are byte-identical across languages and chips, with a golden-hash conformance test and a tie-inclusive top-k rule.
2. **Geometry of the sift.** The same exact search reshaped by geometry: *cubes* (one fetched byte serves several
   queries), *pyramids* (coarse prefix first, exact apex), *spheres* (cluster-first), *lattices* (joint quantisation),
   *quasicrystals* (golden-ratio frequencies), and *programs* stored as vectors. Result: the same answers ~12x faster.
3. **Heterogeneous execution.** The sift as a resident compute-shader engine on an integrated GPU, and a measured routing
   table that sends each job to the engine that wins it.
4. **Omnitri (triangle memory).** A fact (subject, relation, object) is one bound vector; each triangle is filed under
   its three edges, so any two corners recover the third with one clean-up; a second path confirms (triangulation); a
   trit decides commit / escalate / contradiction (triage); decay keeps superpositions under capacity (truncation);
   resultant length measures consistency (harmony). Triangles bundle into **polygons** (one object per session or idea)
   that transform in one operation and nest as atoms of higher triangles.
5. **Substrate.** Truth lives in relational rows (the "square"); bulk inference is sparse matrix closure (the "cube");
   words enter by a full-text index, meaning by a ring pen; the triangle memory supplies what tables cannot (verification,
   coherence, transforms, nesting). A cache in front of the embedding model removes repeated perception.

## 2. SuperSeed v1 / v1.1 — the portable specification

* **Codewords.** `codeword(label, seed, d)`: FNV-1a-64 of the label, mixed with the seed, expanded by splitmix64; each
  output word yields 4-bit phases (K = 16). Integer operations only.
* **Projections.** Ternary matrices {-1, 0, +1} from the same hash stream; P(0) = 1/2, P(+-1) = 1/4.
* **Encoding.** `encode(x, proj)`: float32 input -> int8 by per-row absmax with round-half-to-even -> int32 ternary
  sums for the real and imaginary parts -> the 16-sector phase decided by integer comparisons against fixed tangent
  constants `floor(tan(theta) * 2^16)` at 11.25, 33.75, 56.25, 78.75 degrees (no atan2, no transcendental functions).
* **Score.** `sift(q, lattice)`: exact int8 cos/sin table indexed by phase difference; int32 sums. Identical in Python,
  C (AVX2), C (SSE, for pre-AVX2 CPUs), C++ and a GLSL compute shader (verified on 5,711 queries, 0 mismatches).
* **Packed layout v1.1.** 4 bits per phase in "block-64 split" order: within each 64-dim block, byte j holds phase
  64b + j (low nibble) and phase 64b + 32 + j (high nibble), so a 32-byte SIMD load yields 64 phases. Storage only;
  scores are bit-identical to the unpacked lattice.
* **Cascade.** Score every row on the first d0 dims (each output dim is an independent ternary projection, so a prefix is
  an unbiased, noisier estimate), keep the M best **plus every row tied at the M-th coarse score**, rescore on all dims.
* **Tie-inclusive top-k rule.** Every mark scoring >= the k-th best score votes; order = score descending, then mark index
  ascending; votes are added in that order. Breaking k-th-place ties by index biased heads that hold exact duplicate marks
  (measured: d4 0.745 -> 0.806, d3.outcome 0.595 -> 0.668 when ties are included).
* **Conformance.** `golden()` hashes fixed codewords, projections, encodings and scores; any dialect must reproduce them.

## 3. Ring heads (associative classifiers over marks)

A head stores one mark per labelled training example (the SuperSeed encoding of an embedding vector) plus its label.
Reading = sift the query against all marks, tie-inclusive top-k, score-weighted vote. Outputs: label, confidence (best
score / self score), **margin** ((winner - runner-up) / total vote) used to **abstain**, and **provenance** (`because`: the
marks that voted, with scores and source ids). Training = writing marks; no gradients. The embedding model must be the
same "pen" for marks and queries (re-embedding a head with its serving model: cosine 1.000000 to the served vectors).

## 4. Geometry of the sift

### 4.1 Cubes (tiling)
Batch sift reshaped from lines into cubes (queries x marks x dims). Inner cube = 4 queries x 1 mark: each looked-up
32-phase chunk of a mark is reused by 4 queries before eviction; tiles 32 queries x 64 marks; optional Hilbert-order tile
walk. Bit-identical. 256 queries: 5 MB head 2.1-2.2x (1 thread), 45 MB head (DRAM-resident) 2.4-2.6x (1 thread) and
2.4-2.5x (4 threads). 8-deep cubes no better than 4-deep (compute-bound, register pressure).

### 4.2 Pyramid (coarse to fine)
Three levels: all marks on 320 dims (tiled cubes) -> the 512 best + ties rescored on 1,280 dims (packed rows) -> the 64
best + ties on all 5,120 dims -> vote. 1 label mismatch in 5,711 queries vs the full sift; accuracy change 0.0000 on 8
heads; work 8.8-25.8% of a full sift; 1.692 -> 0.345 ms/query (4.9x) on the 45 MB head; ~12x vs the untiled kernel.
Effective rate 68-80 G dims/s full-search-equivalent, above the 44-53 G dims/s brute-force ceiling (it skips work).

### 4.3 Spheres (cluster-first, IVF on the torus)
k-means on the torus (assignment by phase cosine; centroid = per-dim circular mean re-quantised to 16 phases),
sqrt(n) clusters, probe the best nprobe, then the pyramid inside. Measured limit: at <= 9k marks per head, 1-2% of true
neighbours sit across cluster boundaries and flip labels (0.7-7.5%); the pyramid is already cheap at this scale. Disclosed
variant: boundary marks stored in their two nearest clusters (spill), balanced sizes, for >= 1M marks.

### 4.4 The torus vs the paraboloid
Same projection, two geometries: angle only (torus, 4 bits/dim, phase-cosine score) vs the exact complex value lifted to
(y, |y|^2) so Euclidean nearest = a dot product (paraboloid). Accuracy within +-3 points on 7/8 heads (lookup barely
depends on geometry); the paraboloid costs 2-4x the bytes and was faster under float BLAS; **binding recovery: torus 1.000
vs paraboloid 0.018** (4 role-filler pairs bundled, 1,000-atom clean-up). The torus earns its place by exact composition.

### 4.5 Lattice ladder
Joint quantisation of the exact projection at equal bits per real: cube Z (per real), hexagonal A2 (per complex plane),
D4 (24-cell lattice, 4 reals), E8 (8 reals, Conway-Sloane nearest point). Reconstruction cosine at 2.0 bits/real:
E8 0.969-0.991 > D4 0.958-0.959 > A2 0.956 > Z 0.954; retrieval within ~1.5 points across lattices. (The E8 lead is
overstated: the entropy estimate saturates for 8-dim blocks.)

### 4.6 Quasiperiodic (golden-ratio) frequencies
Fractional-power encodings of scalars with rational frequencies alias (exact repeats over a wide range). Frequencies from
a golden-ratio (Weyl / 1-D cut-and-project) sequence do not: at range R = 100, worst sidelobe 0.045 and 0% gross decode
failures under 20% phase noise, vs 1.000 and 91% for rational torus frequencies (random: 0.069 / 0%).

### 4.7 Programs as vectors
A program = sum over steps of bind(position tick^t, role, instruction). Position by torus ticks; long programs as chained
128-step chunks. 1,024 steps in 8 chunks (20 KB) decode exactly (1.000). Unbinding as integer phase subtraction mod 16 on
the stored bytes (no angles): 33 us per instruction. Capacity edge at 128 steps per chunk (0.999 on CPU and GPU alike).

### 4.8 Hardware limits measured (roofline)
RAM read 11.4 / 12.4 GB/s (1/4 threads); in-cache sift 12.6 / 43.6 G dims/s; tiled sift on a DRAM head 15.5 / 53.0 G
dims/s (cubes deliver ~4x more mark bytes per second than DRAM supplies). Every gain came from reusing fetched bytes or
skipping work — none from faster arithmetic.

## 5. Heterogeneous execution (integrated GPU)

* **Projector shader.** Headless EGL, GLES 3.1 compute: a 64-thread workgroup walks one mark's words together (coalesced),
  8 queries share each loaded word (the cube on the GPU), the score term is a 256-entry table `tab[q*16+p]` held in
  **shared (on-chip) memory**: 10-12.6 G dims/s, bit-identical (naive one-invocation-per-pair shader: 3-4x slower).
* **Resident engine.** Marks uploaded once and kept on the GPU; queries arrive over a pipe; 2-D dispatch; large heads
  dispatched in chunks (a single long dispatch tripped the driver's hang check and returned silently untouched buffers).
  14.1 G dims/s end to end, bit-identical.
* **Task split.** With the CPU embedding text and the GPU sifting, embedding throughput rose 51% (791 vs 525 chars/s).
* **Routing table.** Serving (embed + small sift): CPU for both (GPU sift raised p99 18%). Bulk sift alone: CPU cubes.
  Bulk sift while the CPU embeds: GPU engine. Float or ternary matrix work: CPU (fp32 GEMM: GPU 95 vs CPU 253 GFLOPS;
  ternary table-lookup GEMM on the GPU exact but ~18x slower). Rule: the GPU takes a job only when it is sift-bound AND
  the CPU is saturated by perception.

## 6. Omnitri — the triangle memory

### 6.1 Triangles and edges
Atoms are 16-phase codewords (entities, relations). Each corner role has its own fixed permutation of the D axes. A fact
is one vector `t = P_S(s) + P_R(r) + P_O(o) (mod 16)`; the inverse is subtraction. A **bucket** is the complex sum of the
phasors `exp(i 2pi t / 16)` of its triangles, stored as int8 real/imag with a per-bucket scale.
**Triangulation:** each triangle is filed three times, in buckets keyed by a hash of each edge — (s, r) to recover o,
(r, o) to recover s, (s, o) to recover r. A query with any two corners = one bucket, one unbind (multiply by the conjugate
of the two known corners), one clean-up against the missing corner's permuted codebook. **Confirmation by a second
path:** an answer o from the (s, r) bucket must return r from the (s, o) bucket.

### 6.2 Triage (the trit), truncation (decay), harmony
* **Trit:** -1 contradiction (the second-best corner also scores above tau_c: one edge, two answers); +1 commit (best
  score >= tau AND the second path confirms); 0 escalate to an exact system. Thresholds fitted on a calibration half.
* **Truncation:** capacity per bucket is finite; buckets are split on fill, chains are cut when a hop's margin falls, and
  unreinforced facts decay (`acc <- lambda * acc + new`, window ~ capacity / 2).
* **Harmony:** consistency of k observations of one edge = 1 - s2/s1 of the clean-up scores (equivalently, the resultant
  length of agreeing unbound fillers): one number separating agreeing from dissenting evidence.

### 6.3 Chaining (inference)
Hop by hop: each hop is one edge lookup plus clean-up, which resets noise every hop. Pure phase addition
(`C = A + r1 + r2`) is exact only for translation-like relations, and fails exactly where two relation orders commute
(a then b = b then a: the abelian-torus limit known from TorusE).

### 6.4 Polygons, transforms, recursion
* **Polygon:** the bundle of all triangles of one unit of meaning (here: an agent session, triangle = (step position,
  tool, argument)), positions by continuous-phase fractional power encoding quantised to 16 phases.
* **Transform:** binding distributes over bundling, so ONE operation on the polygon moves every triangle in it: re-time
  (multiply by `exp(i Delta theta)`, query at p + Delta) or re-role (add a phase vector delta, query with key + delta).
* **Recursion:** a polygon's quantised angle is itself an atom; it becomes a corner of higher triangles (polygon,
  relation, outcome); levels stack (facts -> polygons -> groups -> top), each lookup a clean-up that resets noise.
* **Trit solid:** each real and imaginary component thresholded to {-1, 0, +1} (zero band |x| < 0.5 rms).

### 6.5 Sifter core (continuous refinement)
Each bucket keeps its member list (12 bytes per triangle; vectors regenerate from codebooks). A background loop sifts each
member against the bucket; where a member's clean-up margin < 0.1, it is reinforced (`acc += eta * member`) and its
confuser pushed away (`acc -= eta * confuser`) — error-driven bundling without gradients. Disclosed with its measured
limits (section 9).

## 7. Substrate: square, cube, line, circle, triangle

* **Square (truth):** a relational table `triangles(id, s, r, o, polygon, t, provenance, trit)` with indexes; exact hops
  as indexed joins. Polygons = GROUP BY; solids = joins.
* **Cube (bulk inference):** each relation as a sparse boolean adjacency matrix; composed relations = sparse products,
  all chains for all entities at once (an iGPU or background job).
* **Line (words):** a full-text index (SQLite FTS5, BM25) from surface tokens to rows.
* **Circle (meaning):** a ring pen (an embedding model, or a character-trigram phase pen) resolving noisy or partial
  surfaces to atoms by sift.
* **Triangle:** the omnitri layer for verification, coherence, transforms and nesting; every committed answer can be
  confirmed by one indexed row lookup.
* **Router:** exact -> table; words -> full-text; gist -> ring -> triangle -> table confirmation.
* **Perception cache:** exact-text hash in front of the embedding model; near-duplicate reuse only if measured safe.

## 8. Measured results

| Experiment | Result |
|---|---|
| Portability (Python / C AVX2 / C SSE / C++ / GLSL) | bit-identical scores; C++ == Python on 5,711 queries |
| Cubes (TILE-0) | 2.1-2.6x, bit-identical |
| Pyramid (PYRAMID-0) | 4.9x (12x vs untiled), 1 / 5,711 label changes, accuracy change 0.0000 |
| Spheres (IVF-0) | KILL at <= 9k marks/head (boundary flips 0.7-7.5%) |
| Torus vs paraboloid (GEOM-0) | lookup within +-3 pts; binding 1.000 vs 0.018 |
| Lattice ladder (LAT-0) | E8 > D4 > A2 > Z reconstruction; retrieval within ~1.5 pts |
| Golden frequencies (QUASI-0) | R = 100: 0% failures vs 91% rational |
| Programs as vectors (PROG-1) | 1,024 steps exact in 20 KB; 33 us/instruction (integer unbind) |
| GPU resident sift (HOLO-3) | 14.1 G dims/s bit-identical; embedding +51% while the GPU sifts |
| GPU in the serving path (HOLO-4) | KILL: p99 +18% |
| Triangles, capacity (TRI-0) | >= 99% all corners to 256 triangles/bucket (int8); 100k facts strict 0.993-1.000; 1 clean-up per query |
| Chaining (TRI-1) | == forward chaining on 100%: functions 2-10 hops, branching 2-6 hops; phase addition 95-100% (all failures commutative collisions) |
| Triage at fill 64 (TRI-2) | +1 precision 1.000, coverage 0.992, absent edges committed 0, contradictions caught 1.000 |
| Triage at fill 256 (TRI-2b) | +1 precision 0.993, coverage 0.967, absent committed 0, contradictions 0.967; fill 384: coverage 0.30, false contradictions 0.42 |
| Decay (TRI-3) | recent-64 recovery 1.000 vs 0.141 without |
| Harmony (HARMONY-0) | AUC 1.000 (0.955 vs 0.503) |
| Polygons on real agent sessions (POLY-0; 300 sessions, 57 tools, 2,977 arguments) | int8 >= 99.7% to 256 triangles; trit 100% to 32, 99.4% at 64; re-time / re-role transforms lossless; polygon recursion 1.000 / opened 1.000 |
| Two-level nesting (REC-2) | 262,144 facts, end-to-end 0.9995 in 3 lookups |
| Sifter core (SIFT-CORE-0) | int8 capacity 1.5-2x (384: 0.942 -> 0.992; 512: 0.835 -> 0.983); safety bar failed (section 9) |
| Matrix closure vs SQL bulk joins (MATRIX-0) | exact; 23x (2,700 chains) and 102x (43,207 chains) faster |
| Perception cache (EMB-CACHE-0) | live event stream: 97,385 texts, 86 distinct — 99.9% exact repeats; near-duplicates not safe (3 / 60 label changes) |
| Substrate router (SQ-0) | see section 8.1 |
| Exact single-goal chains, ring vs SQL vs hash (DATALOG-FAIR) | ring 2-15 ms, SQLite 0.014-0.061 ms, hash 0.003-0.016 ms (exact lookups belong to indexes) |

### 8.1 SQ-0 (substrate router; 300 sessions, 20,761 rows, 2,808 arguments; 1,000 queries per type)
| Query type | SQL alone | FTS5 alone | Ring alone (trigram pen + polygon) | Routed | p95 routed |
|---|---|---|---|---|---|
| Exact surface | 1.000 | 0.895 | 0.928 | **1.000** | 0.04 ms |
| Lexical (words reordered / case) | 0.003 | 0.903 | 0.327 | **0.905** | 69 ms |
| Noisy (2 character substitutions) | 0.000 | 0.000 | 0.580 | **0.626** | 23 ms |
| Overall | 0.334 | 0.599 | 0.612 | **0.844** | |

Routing met or beat the best single tool on every type and overall (0.844 vs 0.612), but FAILED the pre-registered
latency bar (p95 <= 20 ms): the full-text join (not tuned) costs ~44 ms. Each tool fails a different query type; only
the combination covers all three.

## 9. Limits and negative results (disclosed as measured)

* Superposition beyond ~256 triangles per bucket is not safely usable: at full fill a single bucket's raw score cannot
  separate absent from present facts (absent top / present median 0.67 at 256, 1.00 at 512), and the sifter core makes
  that slightly worse even as it raises recall. Truth must live in rows; triangles index and verify.
* The trit solid is not dense storage: 32 B per recovered fact at 64 triangles vs ~27 B of raw text; refinement with the
  trit inside the loop gained 0.5 points at 256 (0.975 -> 0.980).
* Superposing 2-3 queries into one coarse pass (SUPER-Q-0): 1.6-2.5x faster but 0.4-15% label changes — KILL.
* Clean-up as an integer phase sift on the packed codebook (CLEAN-P4-0): same answers to 128 triangles but 0.6x the
  speed of a float matrix multiply on a 1,000-atom codebook.
* SIMD-within-a-register binding in an array language was slower than byte arithmetic (temporaries); in C the packed
  binding runs at 0.099 ns per phase (109x over 64-bit integers).
* Prefix (pyramid) clean-up on full buckets fails (0.20 agreement): prefixes need single items, not superpositions.
* Ring chaining is ~250x slower than an indexed SQL join on exact chains; its value is noisy or partial corners,
  coherence, transforms and nesting, not exact lookup speed.
* The integrated GPU lost every matrix workload on this laptop and helps only beside a CPU saturated by perception.

## 10. Variations and designs also disclosed

Any D, K, permutation scheme or binding operator (phase addition, elementwise product, permutation, circular
convolution); any number of corner roles (tetrahedra and n-simplices: subject, relation, object, time, context...);
edge indexes over any subset of corners; confirmation by any independent path; trits from margins, path agreement or
coherence; decay by multiplication, subtraction or age; harmony by resultant length, score ratio or entropy; polygons
over sessions, documents, concepts, time windows or users; transforms by any distributive operation; recursion to any
depth; sifter cores running on CPU, GPU, NPU or in-memory hardware, continuously or event-driven; cubes of any depth and
tile shape, Hilbert or other space-filling tile orders; pyramids of any number of levels; clusters with spill;
lattice quantisers (A_n, D_n, E8, Leech) for marks; quasiperiodic frequency sets from any irrational rotation or
cut-and-project scheme; programs as vectors with any position code; heterogeneous routing by measured tables.
Designs not yet measured, disclosed here: (a) fine-tuning a ternary embedding model's own weights on local text pairs
with a full-precision teacher to make it the system's pen; (b) a ternary "triangle head" on an encoder's hidden states
emitting (subject, relation, object) corner vectors bound into a triangle, so text maps directly to triangles; (c) a
versioned ring-memory attention layer inside an encoder (frozen per model version so stored marks stay comparable);
(d) a router that selects table / full-text / ring / triangle per query by the triage trit and measured cost.

## 11. Method
Every experiment was pre-registered in the research log before its harness ran (bars, data, operationalisation);
harness changes made before results were seen are recorded; one post-hoc analysis (integer unbind for programs) is
labelled post-hoc and not re-scored; a mis-calibrated bar (SIFT-CORE-0 safety) is reported as failed with a separate
diagnostic. Research harness file names: parab0, pyramid0, ivf0, lat0, quasi0, prog0/1, holo3, tri_wave, tri_wave2,
poly0, sift_core0, bindswar0, cleanp4_0, superq0, embcache0, matrix0, sq0.

## 12. Prior work (cited, not copied)
Kanerva (hyperdimensional computing; analogy "dollar of Mexico"); Plate (holographic reduced representations, fractional
power encoding); Frady, Kent, Olshausen, Sommer (resonator networks, 2020); Ebisu & Ichise, TorusE (AAAI 2018) and Sun
et al., RotatE (ICLR 2019) — knowledge-graph embeddings on the torus / as rotations, trained by gradient (the triangle
memory here is untrained, integer and online); Imani et al. and others on retraining HDC prototypes (error-driven
bundling); Wu et al., Memorizing Transformers (kNN memory attention); Microsoft BitNet b1.58, BitNet Distillation and
BitNet text embeddings (ternary encoders); Conway & Sloane (lattice quantisers); Weyl equidistribution and quasicrystal
cut-and-project sequences; Hilbert curves; Jegou et al. / FAISS (inverted-file indexes); image pyramids and cascades;
Charikar (SimHash); SQLite FTS5 (BM25).

## 13. Thank you

Nothing here was built from nothing. Every mechanism in this document stands on work that other people did first,
published openly, and explained well enough that a person with a laptop could learn from it. We cited them above; we
want to thank them here, plainly.

* **Pentti Kanerva**, for hyperdimensional computing and for showing that a single vector can hold a structured thought —
  the "dollar of Mexico" analogy is the square in our ladder.
* **Tony Plate**, for holographic reduced representations and fractional power encoding: binding, bundling and clean-up
  are his vocabulary before they were ours.
* **E. Paxon Frady, Spencer Kent, Bruno Olshausen and Friedrich Sommer**, for resonator networks — the settling loop that
  finds the missing corners.
* **Christopher Kymn and colleagues**, for residue hyperdimensional computing, the coprime rings Part 1 is built on.
* **Ila Fiete, Yoram Burak and Ted Brookings**, for grid-cell modular codes, the picture that made coprime rings feel
  natural.
* **Takuma Ebisu and Ryutaro Ichise** (TorusE) and **Zhiqing Sun, Zhi-Hong Deng, Jian-Yun Nie and Jian Tang** (RotatE),
  for putting knowledge on the torus and relations into rotations; their work told us where our triangles stand and
  where the abelian limit lies.
* **Mohsen Imani and colleagues**, for retraining hyperdimensional prototypes — the idea behind the sifter core.
* **Yuhuai Wu, Markus Rabe, DeLesley Hutchins and Christian Szegedy** (Memorizing Transformers), for memory attention
  over stored keys.
* **The BitNet researchers at Microsoft Research**, including **Shuming Ma, Hongyu Wang and Furu Wei**, and the authors
  of BitNet Distillation and BitNet text embeddings, for proving ternary models are real and for releasing open weights
  that run on the computers people already own — our embedder is theirs.
* **Nomic AI**, for an open embedding model with open data and training code, our teacher and benchmark.
* **John Conway and Neil Sloane**, for *Sphere Packings, Lattices and Groups* and the E8 nearest-point algorithm.
* **Hermann Weyl** (equidistribution) and **Nicolaas de Bruijn and Roger Penrose** (quasiperiodic order), whose ideas
  stopped our clocks from aliasing.
* **David Hilbert**, for the curve that orders our tiles.
* **Hervé Jégou, Matthijs Douze, Cordelia Schmid** and the **FAISS** team, for inverted-file indexes and honest
  benchmarks of approximate search.
* **Peter Burt and Edward Adelson** (image pyramids) and **Paul Viola and Michael Jones** (cascades), for coarse-to-fine.
* **Moses Charikar**, for SimHash.
* **Stephen Robertson and Karen Spärck Jones**, for BM25, and **D. Richard Hipp** and the **SQLite** developers, for the
  database and FTS5 that hold our truth in rows.
* **Jon Doyle**, for truth-maintenance systems, and **Tim Rocktäschel and Sebastian Riedel**, for soft unification —
  the logic spine's ancestors.
* The developers of **NumPy, SciPy, OpenBLAS, Mesa** and the **Linux** kernel, and the maintainers of the open GPU drivers,
  whose free tools turned a 2019 laptop into a laboratory.

Any error in how we used their ideas is ours, not theirs.
