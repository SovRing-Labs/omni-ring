# omni-ring — Technical Disclosure, Part 3: designed solids, 4-D cores, the source, seams, the hypervector wire

Defensive publication (prior art). Author: Juan Carlos De Santiago, with the Sovereign Raptor fleet (AI-assisted).
Dated by its release commit and Zenodo version DOI. Part 1 = `TECHNICAL-DISCLOSURE.md` (coprime rings, check ring,
resonator, peeling); Part 2 = `TECHNICAL-DISCLOSURE-PART-2.md` (SuperSeed, ring heads, sift geometry, omnitri,
substrate). This part covers work measured 2026-10-01. Every experiment was pre-registered (bars written before the run);
failures are reported as registered.

## 1. Summary
1. **Designed codebooks ("the matrices are the base, they make all other solids").** A codebook is specified by its Gram
   matrix: simplex (equiangular), orthogonal, cross-polytope, hypercube and polygon codebooks built from harmonic frames on
   the 16-phase torus — exact when n divides 16; tiling a frame across the dimension preserves the Gram matrix exactly.
2. **4-D cores.** Replacing the torus (abelian: bind = phase addition, commutes) with unit quaternions on S³ (bind =
   quaternion product, does not commute) removes the commutative-collision limit of path chaining. Quantising to the
   binary tetrahedral group 2T (24 unit quaternions = the vertices of the regular 24-cell) keeps it exact with a one-byte
   element and a 24×24 multiplication table. The finite subgroups of S³ — 2T (24), 2O (48), 2I (120, the 600-cell) — are
   the doubled symmetry groups of the Platonic solids; each is a candidate core with one-byte elements.
3. **The source: an intent-driven resonator loop whose program is a vector.** Each tick the loop decodes its own program
   vector (instruction slots bound to position atoms and bundled), verifies two independent checksum slots (coprime
   moduli), and executes the instructions (NOP, HOLD m, SHARPEN rate, RELEASE, FREEZE ring, FOCUS ring) on the resonator
   state; tampered programs are refused, the last good program kept (random program vectors are accepted at about 0.04%: see the correction in section 4). The program can be rewritten live.
4. **Seams.** L2-fitted sift tiles derived from the CPU's cache size; a shared-memory single-producer/single-consumer
   ring with a resident C worker for cross-language calls; a fork-server that keeps a Python tool executor warm.
5. **Real embeddings on the torus.** Real document embeddings (1,024-d) phase-encoded through a fixed random projection
   after whitening, bound to position atoms and superposed in one 10 KB polygon, are returned bit-exactly from the row
   store; whitening is required because real embedding sets occupy a narrow cone.
6. **The hypervector wire.** A request (tool, head, argument) encoded as a role-bound, phase-quantised, 4-bit-packed
   2,560-byte frame: any node can route, deduplicate and cluster requests without parsing.
7. **The geometric attention head (design).** Attention mapped onto the system: stored shards = K and V (swappable
   memory), a core = the model (forms Q from intent and state; chooses shards, threshold, fold-back), the attention firing
   = sift (QKᵀ on the torus) → threshold/top-k gap → bundle·V; N heads = N mounted shards merged by bundling.

## 2. Designed codebooks
- Simplex of n points in the 16-phase torus: harmonic frame rows `k·j mod 16` (k = 1..n-1 of the n-th roots),
  Gram = -1/(n-1) off-diagonal, exact when n | 16; orthogonal sets via Kronecker products of harmonic frames;
  cross-polytope = ±orthogonal; hypercube = sign patterns; polygon = cyclic rotations. Module `omni_ring/solids.py`
  (`simplex`, `orthogonal`, `cross_polytope`, `hypercube`, `polygon`, `gram`, `min_separation`), 9 tests.
- SIMPLEX-0 (relation codebook in triangle buckets): recovery random vs simplex 256: 0.9995 = 0.9995; 512: 0.9929 vs
  0.9932; 1,024: 0.896 vs 0.912 (+1.6); 2,048: 0.672 vs 0.697 (+2.5); 4,096: 0.447 vs 0.463. 5th-percentile margin +9%
  (256), +29% (512). A small, real gain that appears near capacity.
- DESIGNED-0 (whole triangle memory on exactly orthogonal designed codebooks): no gain over random (TRI-0 int8 capacity
  256 = 256; TRI-2b triage within noise). Disclosed as a negative result: at D = 5,120 random phase codebooks are already
  near-orthogonal; design pays only where codebooks are small or dimensions few.

## 3. 4-D cores
- QUAT-0: tree world (500 nodes, 10 relations), query = start bound with the path's relation vectors in order, one
  clean-up. Torus: exact 0.945–1.000 with 54/54/46/28/12/8/2 commutative collisions at hops 1–7. Unit quaternions (float):
  exact 1.000 at every hop 1–10, 0 collisions, at D = 1,280 and 5,120.
- QUAT-1: every coordinate one of the 24 elements of 2T, stored as a byte; bind = MUL[24×24] table lookup (integer,
  exact); similarity = mean of a 24×24 dot table. Exact 1.000 and 0 collisions at every hop 1–10; 5,120 B/vector
  (5 bits/coordinate packable → 3,200 B) vs 20,480 B for float quaternions. 29.2% of element pairs commute.
- QUAT-2: triangle buckets on the 2T core — triangle = P_S(s)·P_R(r)·P_O(o) per coordinate (order kept), bucket = int8 sum
  of quaternion components, unbind by multiplying with the conjugate on the correct side. int8 capacity (all corners ≥ 99%)
  384 vs the torus's 256; at 512: 99.2/99.9/99.0% vs ~87%. Bytes per bucket 20 KB vs 10 KB (about even per byte); the
  4-D core adds exact order at no capacity cost.
- Variations disclosed: 2O (48 elements) and 2I (120 elements, 600-cell) cores with byte elements; D4 / hypercube cores;
  mixed cores per field (abelian torus for unordered fields and time, non-abelian core for ordered paths); "the algebra
  of the memory matches the algebra of the data" as a selection rule, measured per structure type.

## 4. The source (program-vector resonator loop)
- SOURCE-0 (persistent, intent-fed resonator vs one-shot decode, Part 1 rings, D = 5,120): recall at M = 8/12/16 marks
  per slot 0.975/0.915/0.835 vs 0.940/0.880/0.790 (small, consistent gain; +10-pt bar not met); intent switch without
  restart 0.25–0.27 (failed: momentum holds the old attractor).
- SOURCE-2: the loop as an interpreter of its own program vector (4 instruction slots + checksum slot, int8-stored).
  Self-written RELEASE on intent change: switch success 0.985/0.935/0.830 (from 0.265/0.250/0.270). One 48-way checksum
  leaked tampered programs at the expected 1/48 rate (0.985 refused).
- SOURCE-3: two independent check slots, c1 = Σ(k+1)·id mod 48 and c2 = Σ p_k·id mod 47 (p = 3, 5, 7, 11): 1,000
  tampered + 1,000 random programs refused 100% (superseded for random vectors, below), 200 clean accepted 100%; FREEZE removes a ring from the update set on
  the next tick and NORMAL restores it, 100/100 (measured as update events); recall unchanged; 544 ticks/s in Python,
  decode 0.42 ms/tick.
- Variations: programs exchanged between processes as vectors and verified before execution; programs bundled with
  their data; instruction sets for sift/gap/fold (array-language statements) executed on the same kernels.
> Correction (2026-10-01, experiment TRUEFORM TF-4b): the 1,000-trial random-vector result was
> small-sample. Over 20,000 uniform random program vectors per arm, both random-atom and orbit slot positions accept
> ~0.04% (8 and 9 of 20,000; analytic (1/48)^2 = 0.043%). Single-instruction tampering is always caught (c2 mod 47 is
> prime) (partial: this holds for the pair of checks, by exhaustive test over every single-slot change; c2 alone misses a difference of 47 in every slot and c1 catches it). Two small check slots are an error-DETECTING code, not a security boundary; a much larger check space is needed
> before program integrity can be claimed against random or adversarial vectors.


## 5. Seams
- TILE-1: batch sift tile (TQ queries × TM marks) chosen so TM·d + TQ·2d ≤ L2 (read from the OS) → (8, 32) on a 256 KB
  L2: 1.18× (1 thread) and 1.33× (4 threads) over a fixed (32, 64); 60 tile shapes bit-identical. Shipped as
  `seed.tile_shape(d)` (override `OMNIRING_TILE`).
- IPC-0 (1 KB round trip, median): shared-memory SPSC ring + resident C worker 0.9 µs (p99 1.2); resident Python over a
  Unix socket 9.5 µs; spawning a Python process 32 ms; spawning the tool executor 170 ms.
- IPC-1: a fork-server for a Python tool executor (warm main(), one forked child per request; children must reset
  SIGCHLD to default to see exit codes): tool calls 198 → 38 ms (bash) and 179 → 23 ms (write); identical results 20/20.
- FEED-0: an exact-text digest cache in front of the embedder served 2,000 live notes with identical labels at 299×
  throughput (the stream had 2 distinct texts; cache only).

## 6. Real embeddings in torus polygons
- Data: 6,194 real document-chunk embeddings (1,024 floats each). Encode: phase = angle(R1·e + i·R2·e) quantised to 16
  (R1, R2 fixed Gaussian projections, D = 5,120); polygon = int8-quantised complex sum of M items, item k bound to
  position atom k; read k = unbind + sift against all 6,194 phase vectors → id → the stored 1,024 floats.
- Without whitening (NOMIC-POLY-0): exact 0.997/0.983/0.953/0.765/0.361 at M = 16/32/64/128/256 — the set sits in a
  narrow cone (mean pairwise cosine 0.82) and meaning-preserving phases crowd.
- With mean-centring + SVD whitening before the projection (NOMIC-POLY-1, fresh seed): M = 64 exact 0.9996, committed
  precision 1.000 (coverage 0.989); M = 128 exact 0.996, committed 0.9996 (1 wrong of ~2,505; the registered 100% bar
  missed by one); M = 256 exact 0.971. One 10,240 B polygon holds 64–128 embeddings (262–524 KB raw).
- Abstain rule: commit only if (top1 − top2)/top1 ≥ 0.05. Variation: random id atoms in the polygon with meaning kept in
  the row (1.000/1.000/0.994 at 64/128/256); 4-D core polygons; a second check residue per slot.

## 7. The hypervector wire (HV-0)
- 24,495 real agent tool steps → 4,506 distinct (tool, head, argument) requests. Frame = phase-quantised Σ_f R_f + a(v_f)
  (role atoms R_tool, R_head, R_arg; atoms from a portable hash generator), packed 4 bits per phase = 2,560 B.
- Decode a field = unbind R_f + sift against that field's codebook: tool / head / argument 1.000 / 1.000 / 1.000
  (codebooks 57 / 320 / 3,202 values); identical requests → identical frames; 0 collisions among 4,506 frames;
  similarity same tool+head 0.55 vs different tool 0.02. Encode 0.98 ms, decode 1.46 ms/field in numpy.
- Uses disclosed: routing without parsing; exact-duplicate caching by frame digest; clustering; writing requests as marks;
  frames attached to tool calls and to a work ledger (who / what / size / connected-to) for completion verification.

## 8. The geometric attention head (design, disclosed)
`Q ← encode(intent ⊗ state)`; for each mounted shard h: `S_h ← sift(Q, K_h)` (exact integer QKᵀ on the torus);
`A_h ← gap(S_h)` (threshold / tie-inclusive top-k / log-domain softmax); `O_h ← A_h · V_h` (weighted bundle);
`state ← fold(state, bundle_h(role_h ⊗ O_h))`. Shards are swappable memory (same core + new shard = same skill on new
knowledge); the core program is the skill (same shard + new program = new skill on the same knowledge). Every output is
traceable to the marks that passed the gap.

## 9. Limits and negative results
DESIGNED-0 no gain; SOURCE-0 below its bar; SOURCE-2 single checksum leaks at 1/48; NOMIC-POLY-0 failed without
whitening; NOMIC-POLY-1 128-item leg missed 100% by one item; IPC-1 latency bar (20 ms) missed (38 ms; remaining cost is
inside the executor); FEED-0 exercised only the cache; q8_0 quantisation of the text encoder was slower (0.78×) and less
exact (min cosine 0.9972) than f16 on the measured CPU; triangle-bucket triage unsafe beyond 256 per bucket on the torus.

Registered failures (as measured):
- SOURCE-2: live rewrite 0.43 against a registered 100 % (the failing leg was "unfreeze resumes updates", 43 / 100, because a converged ring
  sits at its fixed point; the freeze itself took effect next tick in 100 / 100); tampered / random programs refused 0.985 / 0.980 against
  100 %. SOURCE-3 is the response.
- The 4-D core is slower per bind than the torus: 2T table lookup 60.0 µs against 6.2 µs; float quaternion 410 µs against 105 µs.
- The derived tile (8, 32) is not the optimum: the best grid cell reached 1.41× against 1.18× / 1.33× for the derived tile.

## 10. Prior work (cited, not copied)
Kanerva (hyperdimensional computing); Plate (holographic reduced representations); Gayler (VSA); Frady, Kleyko, Sommer
(resonator networks; residue/phasor HDC); Kleyko et al. (VSA surveys); Conway & Smith (On Quaternions and Octonions —
finite subgroups of S³, the 24-cell and 600-cell); Coxeter (Regular Polytopes); Vaswani et al. (attention); TorusE
(knowledge-graph embedding on a torus — the commutative limit); Kwon et al. (vLLM, paged attention) and llama.cpp
(continuous batching) for the batching design; HPVM-HDC / HDC++ (hypervector language design).

## 11. Method
Pre-registered bars: `experiments/PREREGISTRATIONS-OMNITRI.md` (bars registered in other plan files are not published); harnesses
and result JSONs in `experiments/`; numpy + C on a 4-core laptop (i5-8365U, 256 KB L2/core). Failures are reported as
registered; diagnostics are labelled and never re-scored.
