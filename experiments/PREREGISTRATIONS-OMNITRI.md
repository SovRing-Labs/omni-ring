# OMNITRI — the triangle primitive (imagine/plan step; J 2026-09-30)
Pillar: Nervous System. Lineage: omni-ring (the line as a primitive: exact bulk sift), GEOM plan (unpublished) (torus earns its place
by binding 1.000 vs 0.018; QUASI-0 golden frequencies; PROG-1 capacity ceiling), CORTEX-LOGIC (Tracker = exact ground truth),
ring heads (margin abstain = the trit). J's properties: TRIANGULATION, TRIAGE, TRUNCATION; movement, resonance, harmony.
Prior art (search before judging): TorusE (Ebisu & Ichise, AAAI 2018 — KG embedding on the n-torus, relations as
translations) and RotatE (Sun et al., ICLR 2019 — relations as rotations in complex space). Both are TRAINED by gradient.
Our distinct claim: integer, exact, untrained, online (a fact is usable the moment it is written), same spec on CPU/SSE/iGPU.

## Design (status: DESIGN, nothing run)
Fact = triangle (s, r, o). Atoms: random 16-phase codebooks per ring (entities, relations); one permutation per corner role.
  t = P_S(s) + P_R(r) + P_O(o)  (mod 16)   — one bound vector, exact inverse by subtraction.
TRIANGULATION — every triangle has three edges; index each edge. Store t into three edge-keyed bundles:
  key(s,r) -> bundle holds o;  key(r,o) -> holds s;  key(s,o) -> holds r.   Bundle = complex sum (int accumulators) of
  unbound fillers under a hashed bucket of the edge. Query with any two corners = one edge = one bucket, one unbind, one
  clean-up sift over the missing corner's codebook. Confirmation = a second, independent path lands on the same corner.
TRIAGE — every closure gets a trit from clean-up margin (best - second) and path agreement:
  +1 commit (margin >= tau AND >= 2 agreeing paths, or 1 path at high margin), 0 escalate to Tracker/Prover, -1
  contradiction (one edge, two different corners each above tau). Same shape as the ring heads' margin abstain.
TRUNCATION — keep every superposition under its capacity: cap facts per bucket (split on fill), cut a chain when the hop
  margin falls below tau, decay unreinforced facts (the Loop's Reinforce/Decay), pyramid prefix (320 dims) for clean-up.
Chaining (inference) = hop by hop: each hop is one edge lookup + clean-up, so noise is reset every hop.
  Correction to 09-30 chat: pure phase-addition chaining (C = A + r1 + r2) is exact only for translation-like relations
  (TorusE's regime); the general case is hop + clean-up. Both measured below.
MOVEMENT = golden-ratio phase clock on a time corner (tetrahedron: s, r, o, t). RESONANCE = resonator settling when two
corners are unknown. HARMONY = resultant length of agreeing triangles (consistency as one number).
Home when it passes: omni_ring/tri.py (library), then the always-on iGPU engine (HOLO-5) runs the T3 loop in background:
triangulate new closures -> triage -> truncate; the CPU answers from committed closures.

## Pre-registrations (before running; scripts to be experiments/harness/tri0.py ...; data = LOGIC-0/0b fact sets + synthetic)
TRI-0 edge-index recovery. D = 5120. Facts 1k..100k synthetic (1k entities, 50 relations) + LOGIC-0 facts. Hide one corner.
  PASS: >= 99% recovery at bucket fill <= cap, for all three edge types; report cap where it drops below 99%; query cost
  <= 2 sifts of the codebook. KILL: cap < 16 facts/bucket (then it is a hash table with extra steps).
TRI-1 chaining vs Tracker. LOGIC-0 transitive/rule cases, 1-10 hops. Arms: hop+clean-up; pure phase addition.
  PASS: hop+clean-up derives the same facts as Tracker.chain on >= 99% of cases up to 5 hops, and is faster than
  forward chaining for single-goal queries. Phase-addition arm reported, expected exact only on translational relations.
TRI-2 triage calibration. +1 precision >= 99% vs Tracker truth; contradictions (-1) catch >= 95% of Tracker.contradictions();
  0 rate reported (escalation share). KILL: +1 precision < 97%.
TRI-3 truncation. Pyramid-prefix clean-up gives the same corner as full clean-up on >= 99.5%; decay keeps recovery >= 99%
  under a stream that would otherwise overfill buckets.
HARMONY-0 resultant length separates consistent vs contradictory fact sets: AUC >= 0.95 against Tracker labels.
Order: TRI-0 -> TRI-1 -> TRI-2 -> TRI-3 -> HARMONY-0 -> RES-TRI-0 (two unknown corners) -> HOLO-5 (engine).

## Verify
- design present: grep -c "TRI-0" experiments/PREREGISTRATIONS-OMNITRI.md
- after each run: results appended here as "- TRI-n RESULT: ..." with ~/gpurun-out-0930/tri*.result.json

## POLY — triangles into polygons, applied to the shards (J 2026-09-30: "every triangle becomes a building triangle for a
## polygon of information … compress a multitude of larger bits into one solid, manageable, transformable polygon")
Lineage: SHARDS plan (unpublished) (slots = int8 re/im superpositions, SCALE-1a ~100 facts per 5,120-d slot; Encoding-2 finding: a
partial cue has ~0 match with a plate, so the router is metadata) — the edge key fixes that: two known corners ARE a cue.
Design (status: DESIGN):
  POLYGON  = bundle of the triangles of one unit of meaning (a session, a document, a concept) = one slot vector.
  SOLID    = polygons glued along shared edges (shared corners) = one shard; the shard library = the assembly.
  RECURSION = a polygon's vector (cleaned against a polygon codebook) is itself a corner of higher triangles
             ("session P — fixed — bug B"), so each level scaffolds the next (the fractal; primitives standing alone).
  TRANSFORM = binding distributes over bundling: ONE integer add to the polygon moves every triangle in it at once
             (re-time = movement, re-role, map to another domain = the square/analogy), no unpacking.
  TRIT SOLID = threshold the polygon's accumulators to {-1, 0, +1} per component (weak = 0) — the storable, servable form.
Honest byte math (to be measured, not claimed): int8 slot 10 KB / ~100 triangles ~ 100 B per fact — about raw-text size;
  a trit solid ~ 20 B per fact. The gain is ONE operable object (O(1) transforms, O(1) edge queries), bytes second.
POLY-0 (pre-registered; data = episode-current sessions as polygons: triangle = (step index, tool, argument token), plus
  LOGIC-0 facts). Report recovery of a hidden corner vs triangles per polygon N (16..400), int8 vs trit.
  PASS: (a) int8 >= 99% at N >= 64 and trit >= 99% at N >= 32; (b) transform: after one-add re-time/re-role, recovery equal
  to the untransformed polygon (+-0.5 pt); (c) recursion: from a higher triangle, retrieve the right polygon top-1 >= 99%,
  then its hidden corner at the (a) rate. KILL: trit < 99% at N = 16 (then the trit solid is not a storage form).
  Report bytes per recovered fact for int8, trit, and the raw text.
Verify: python3 experiments/harness/poly0.py prints the verdict; result appended here as "- POLY-0 RESULT: ...".

## Wave-1 operationalization (declared 2026-09-30 BEFORE any harness ran; J "run the next wave of tests")
Script experiments/harness/tri_wave.py (parts tri0, tri1, tri2, tri3, harmony0) + experiments/harness/poly0.py. CPU, < 3 GB RAM.
Common: D = 5120; atoms uniform on 16 phases; roles S/R/O = three fixed random permutations of the D axes; a triangle
  t = P_S(s)+P_R(r)+P_O(o) mod 16; a bucket/polygon = complex sum of e^{i 2pi t/16}, stored int8 re/im with a per-bucket
  scale (the SHARDS slot format); float arm reported alongside. Query = multiply by the conjugate of the two known corners,
  clean up by phase-cosine against the missing corner's permuted codebook.
TRI-0: capacity curve — buckets of N = 16..512 random triangles (1k entities, 50 relations), hide each corner type; then
  100k facts hashed into buckets sized to the measured cap. One-to-many edges scored strictly: the top-|A| answers must be
  exactly the true set A.
TRI-1: synthetic rule world (Tracker's LOGIC-0 worlds are unary-level; this needs binary chains): 300 entities, 10 base
  relations each a random total function, derived c_1 = r_1 o r_2, c_h = c_{h-1} o r_{h+1} (h <= 9, so 2..10 hops), as Horn
  rules in Tracker. Only BASE facts go in the triangle store (edge (s,r) buckets). Query (x, c_h, ?) answered by hopping
  base relations with clean-up each hop; truth = Tracker.chain facts. Speed vs Tracker.chain (full) and Prover.prove
  (single goal). Branching variant (each relation 1-2 successors; answer sets compared exactly) reported. Phase-addition arm:
  a translational world (entity = root + sum of relation vectors along its path, 500-node random tree, 10 relations),
  query = x + v_1 + ... + v_h, one clean-up.
TRI-2: store 20k facts at fill 64/bucket. Queries: 5k present edges, 5k ABSENT edges (key never stored — must not commit),
  1k contradiction edges (two objects written on a functional edge). Second path = closing check: the answer o from the
  (s,r) bucket must return r from the (s,o) bucket. tau fitted on a calibration half, reported on the other half.
TRI-3: prefix clean-up (first 320 of the permuted dims -> top-8 -> full) vs full; decay: stream 2,000 triangles into one
  bucket with accumulator *= lambda per insert (lambda so the effective window ~ cap/2), recovery of the most recent 64.
HARMONY-0: ground truth is by construction, not Tracker — Tracker's contradictions() only sees explicit (p, not-p) and has
  no inequality, so it cannot label two different objects on a functional edge (deviation declared). k = 8 observations of
  one functional edge inside a bucket of 64 background triangles; contradictory = 1..4 dissenting observations.
  harmony = 1 - s2/s1 of the clean-up scores; AUC over 2,000 sets.
POLY-0: current opencode.db (read-only), sessions via the shard episode loader (unpublished) load(); triangle = (position p, tool, arg
  token) with positions as continuous-phase FPE (theta_d uniform) quantized to 16 phases; polygons binned by N; trit = each
  of re, im in {-1,0,+1} with zero band |x| < 0.5 * rms. Re-time transform = multiply the accumulator by e^{i Delta theta}
  and query at p+Delta. Recursion: polygon atom = 16-phase angle of its accumulator; 2,000 higher triangles
  (polygon, outcome-relation, outcome) bucketed at fill 32.
POLY-0 detail (declared before poly0.py ran): polygon N = first N steps of sessions with >= N steps (<= 40 sessions per N);
  arg clean-up against the FULL arg vocabulary of the loaded sessions (chunked); position corner scored strictly (top-|A|).
  Recursion relations: title (full-title atom), day, cost band; polygon retrieval scored on title edges whose title is unique
  (one-to-many day/cost edges reported as strict sets). Trit bytes = 2 trits/dim packed 5 per byte (2,048 B per polygon).

## SIFT-CORE-0 (J 2026-09-30: "what if those buckets also had a sifter core, like the omniring … constantly fires, sifting
## naturally") — pre-registered BEFORE running; experiments/harness/sift_core0.py; runs after POLY-0 finishes
Idea: each bucket keeps its member list (ids, 12 B/triangle; triangles regenerate from the codebooks) = its core. The core
fires continuously: sift every member against the bucket; where a member's clean-up margin (true - best other) < 0.1 on
any corner, reinforce it (acc += eta * member) and push away the confuser (acc -= eta * confusing triangle); re-render
int8 / trit. Error-driven bundling (perceptron-style refinement of a superposition; cf. retraining in HDC classifiers).
Buckets N = 256, 384, 512, 768, 1024 random triangles (1k entities, 50 relations), 8 buckets each, eta = 0.5, 10 passes.
Scored on ALL members, all three corners. Safety: 2,000 ABSENT edge queries per N — p99 of their top score must stay
below TRI-2's tau (0.3) after refinement (the core must not teach the bucket to hallucinate).
PASS: int8 >= 99% at N >= 384 (1.5x the plain cap 256) within 10 passes, absent p99 < 0.3. Trit capacity reported.
KILL: no N step gained over plain.
- SIFT-CORE-0 harness change: buckets run in parallel (4 procs), one seeded rng per bucket. CORRECTION: the old serial
  harness had printed its N=256 line (int8 0.9917 -> 0.9951, trit 0.9795 -> 0.9844, absent p99 0.671) just before it was
  killed; I had not read it when deciding. Both runs reported. Note: the absent bar (p99 < 0.3) was set from TRI-2's tau at
  fill 64; at fill >= 256 plain-bucket noise alone may exceed it — a plain-bucket absent p99 will be reported as a
  DIAGNOSTIC (not a re-score).

## Wave-1 RESULTS (2026-09-30; ~/gpurun-out-0930/tri_wave.result.json, poly0.result.json, sift_core0.result.json)
- TRI-0 PASS: int8 cap 256 triangles/bucket (>= 99% all corners; 384: 96%, 512: 87%); 100k facts at fill 256 strict
  recovery S 0.9943 / R 1.0000 / O 0.9933; one clean-up sift per query.
- TRI-1 PASS: hop + clean-up == Tracker.chain on 100% of cases, functions 2-10 hops, branching 2-6 hops (answer sets
  exact). Ring query ~7-8 ms vs Tracker.chain 1.06 s / 17.3 s (naive nested-loop joins — overstates the gain) and
  Prover 144 / 218 ms single goal. Phase-addition arm 95-100%: every failure a commutative collision (a then b == b then
  a) — the abelian-torus limit (TorusE regime); hop + clean-up is the general method. Harness fix: Tracker.assert_fact
  re-chains from all facts per call (O(n^2)); bulk-loaded base facts then one chain().
- TRI-2 PASS (fill 64): +1 precision 1.000, coverage 0.992, absent edges committed 0.000 (all escalated), contradictions
  caught 1.000, false contradictions 0.000 (tau 0.30 at the grid edge, tau_contra 0.42).
- TRI-3 SPLIT -> FAIL as registered: decay PASS (recent-64 recovery 1.000 vs 0.141 without), prefix clean-up FAIL (0.203
  same as full): 320 dims cannot beat the noise of 256 superposed facts; the pyramid needs single items (a core), not
  full buckets.
- HARMONY-0 PASS: AUC 1.000 (consistent 0.955 vs contradictory 0.503).
- POLY-0 PASS (a, b, c): real sessions (300, 57 tools, 2,977 args). int8 >= 99.7% to N = 256; trit 100% to 32, 99.4% at
  64, 95-99% at 128. Re-time and re-role transforms 1.000 = base. Recursion: 92 polygons, 84 unique-title queries,
  polygon top-1 1.000, opened 1.000. Bytes per recovered fact: trit 32 B at N=64, 16 B at 128; raw text 26.9 B — the trit
  solid is NOT meaningful compression for short facts; the gain is one operable object. Perf fix: clean-up tables
  precomputed + batched (19x faster, identical results).
- SIFT-CORE-0 FAIL as registered (capacity leg met, safety leg failed): int8 384: 0.942 -> 0.992, 512: 0.835 -> 0.983 in
  10 passes (~1.5-2x capacity); 768+ not rescued; trit no gain (refinement in float, then rounded — the core does not
  yet optimize the trits themselves). Absent p99 bar failed (0.69..3.1) — partly miscalibrated (absolute score grows
  with reinforcement). DIAGNOSTIC (relative: absent top / present median true): 256 0.670 -> 0.691, 384 0.855 -> 0.941,
  512 1.004 -> 1.056 — at full fill a single bucket's raw score cannot tell absent from present even before refinement,
  and refinement worsens it slightly. Safety at high fill must come from TRI-2's closing check (second path), untested
  above fill 64. Old serial harness N=256 line: 0.9917 -> 0.9951 (consistent).
NEXT (support needed): TRI-2b triage at fill 256/384 with the closing check, with and without the core; TRIT-CORE-0 (the core
optimizes trits directly); fair speed vs an indexed Datalog engine; recursion level 2 measured; real fleet facts.

## Wave-2 STAGED (J 2026-09-30: "manage and transform 3 items as though it were 1 … apply this to our cpu processing, on the
## conveyor belt, each lane per clock cycle takes 1 thing but processes it as though it were 3. stage the next tests")
Physics first (GEOM lineage, LIMIT-0): a lane cannot do 3x the arithmetic per cycle. "1 as 3" is real in three ways only —
(i) fewer BYTES moved per item (the cube: one fetched mark byte serves 4 queries, measured 2.4x), (ii) fewer PASSES (one pass
serves several items, then a cheap exact check — the pyramid pattern), (iii) more ITEMS per register (SWAR / SIMD packing).
Each test below is one of these. All pre-registered here BEFORE writing their harnesses; none run.

BELT-3 tests (the conveyor belt = ring_headd's batch path: embed -> sift -> vote)
- SUPER-Q-0 (fewer passes; the triangle's "3 as 1"): bundle 3 encoded queries into ONE phase vector (complex sum -> angle
  -> 16 phases), one coarse sift over all marks, take the top-M candidates, rescore each query exactly on the candidates
  only (sift_rows_p4), vote as read_marks. Heads: all 8; M in (64, 128, 256) per bundle; groups of 2 and 3 queries.
  PASS: label mismatch vs exact read_marks <= 0.5% and dacc >= -0.005 on every head (IVF-0's bar), AND sift-stage time per
  query >= 1.5x faster than exact on d5.verb_log at 1 thread. KILL: no M meets the mismatch bar with any speed-up.
  Harness: experiments/harness/superq0.py.
- BIND-SWAR-0 (more items per register): phases packed 2 per byte; bind/unbind = nibble-wise add/sub mod 16 on uint64
  words (16 phases per 64-bit op; masks 0x0F0F.. for even/odd nibbles, no cross-nibble carry). Arms: int64 numpy (as used
  in tri_wave.py), uint8 numpy, uint64 SWAR numpy, C (auto-vectorized AVX2) in c_src. PASS: bit-identical to (a+b) % 16 on
  10M random pairs, all arms; SWAR >= 4x uint8 and >= 16x int64 per phase; C >= 2x SWAR numpy. Harness: bindswar0.py.
- CLEAN-P4-0 (fewer bytes; the omniring kernel does the triangle clean-up): quantize the unbound query u to 16 phases
  (angle) and run clean-up as an omniring sift on the packed codebook (sift_batch_p4; 4-query cube applies) instead of the
  float cos/sin matmul. PASS: same corner as float clean-up >= 99.5% at fill <= 128 and report 256; >= 1.5x faster per
  query (1 thread). KILL: < 98% at fill 64 (magnitude loss too costly). Harness: cleanp4_0.py.
- BELT-TRI-0 (integration, only if SUPER-Q-0 and CLEAN-P4-0 pass): ring_headd batch path gains RING_SUPERQ=3; bench_belt.py
  rhythm arm, same text stream as BELT-CPP. PASS: batch texts/s >= +20%, p99 not worse, labels == exact on the eval sets.

Omnitri support tests (from wave-1 NEXT)
- TRI-2b: TRI-2 at fill 256 and 384, closing check on, arms plain vs SIFT-CORE (10 passes). PASS at 256 (plain): +1
  precision >= 99%, absent committed <= 0.5%, contradictions caught >= 95%; 384 reported per arm (decides whether the
  core's capacity is usable safely). Harness: tri_wave.py tri2b.
- TRIT-CORE-0: the core refines with the TRIT rendering inside the loop (evaluate on trit(acc), reinforce failures in the
  float acc, re-render). Random buckets as SIFT-CORE-0. PASS: trit >= 99% at N = 256 (plain 0.975) in <= 10 passes.
- REC-2: two nesting levels. 4,096 polygons x 64 synthetic triangles (262k facts); level 1 = higher triangles (polygon,
  group-rel, group) at fill 64; level 2 = groups as atoms in top triangles. End-to-end 3-lookup recovery PASS >= 99%;
  report lookups and ms per query.
- DATALOG-FAIR: same TRI-1 worlds; baseline = SQLite with indexes on (pred, a) answering (x, c_h, ?) by an h+1-way indexed
  join, and a hash-indexed Python chainer. Report ms per single-goal query for 2..10 hops. No pass bar on speed (it is a
  calibration): the ring's honest speed claim becomes whatever this shows; prediction written now: SQLite wins on exact
  functional chains (sub-ms), the ring's distinct value is noisy/partial corners, harmony, transforms, nesting.
Order: BIND-SWAR-0, CLEAN-P4-0, SUPER-Q-0 (CPU belt) -> TRI-2b, TRIT-CORE-0 -> DATALOG-FAIR, REC-2 -> BELT-TRI-0.
Perf rule for all harnesses (J 09-30, one core pegged while 7 idled): precompute clean-up tables once, batch queries into
one matmul, Pool(4) across independent buckets/heads, OPENBLAS_NUM_THREADS=2 per worker; report wall and CPU time.

## Wave-2 RESULTS (2026-09-30; ~/gpurun-out-0930/{bindswar0,cleanp4_0,superq0,tri_wave2}.result.json)
- BIND-SWAR-0 FAIL as registered: all arms bit-identical; SWAR in numpy 0.63 ns/phase is SLOWER than uint8 numpy 0.29
  (temporaries); C packed 0.099 ns/phase = 109x int64 (what tri_wave uses), 2.9x uint8, 6.4x SWAR-numpy. Use: C packed
  binding (or at least uint8, 38x over int64) in every omnitri harness.
- CLEAN-P4-0 FAIL: phase-quantized clean-up on the packed omniring kernel agrees with float clean-up 100% to fill 64,
  99.9% at 128, 97.1% at 256, but is 0.6x the speed (1-thread BLAS beats the LUT sift on a 1,000-atom codebook).
- SUPER-Q-0 KILL: superposed queries 1.6x (pairs) - 2.5x (triples) faster on the sift stage, but no config meets the
  0.5% mismatch bar on all 8 heads (best g2 M256: 0-0.8% on 6 heads, d3 heads 2.8-5.2%). Harness validity 0 mismatches.
- DATALOG-FAIR (calibration): ring exact on all chains but 2-15 ms/query vs SQLite 0.014-0.061 ms and hash 0.003-0.016
  ms (~250x / ~1,000x). Prediction held. RETRACTS the chat's "130-2,100x" speed framing (that baseline was our naive
  Tracker). Exact hops belong to SQL indexes.
- REC-2 PASS: 262,144 facts, 2 nesting levels, group 1.000, polygon 1.000, end-to-end 0.9995 in 3 lookups, 3.2 ms/query.
- TRIT-CORE-0 FAIL: trit-in-loop refinement 256: 0.975 -> 0.980; 192 and 384 no gain.
- TRI-2b PASS (fill 256 plain): +1 precision 0.993, coverage 0.967, absent committed 0.000, contradictions caught 0.967,
  false contra 0.007. Core at 256: precision 0.995, caught 0.990 but absent committed 0.0028, false contra 0.021.
  Fill 384 plain: precision 0.995 but coverage 0.298, false contra 0.416 (unusable); core at 384: coverage 0.0016
  (unusable). => Triage is safe to fill 256; SIFT-CORE's extra capacity is NOT safely usable. Truth must live in rows.
Thesis after two waves: triangles, chaining, triage (to 256), harmony, polygons, transforms, 2-level nesting HOLD.
Not supported: speed on exact lookups (SQL wins), trit solid as storage beyond ~64, superposition beyond 256/bucket,
"3 as 1" query superposition at our accuracy bar.

## Wave-3 STAGED (J 2026-09-30: SQL matrix + FTS5 + nomic substrate; ternary embedder; splice into the transformer)
Pre-registered here before any harness; none run.
- SQ-0 substrate prototype: SQLite table triangles(id, s, r, o, polygon, t, provenance, trit) + FTS5 over surface text
  + ring marks (one pen) + omnitri buckets (fill <= 256) on POLY-0's real sessions. Router: exact -> SQL index; words ->
  FTS5; gist -> ring sift -> omnitri -> SQL confirm. Mixed query set (exact / lexical / paraphrase-noisy). PASS: routed
  accuracy >= best single tool on every query type AND overall, p95 latency <= 20 ms.
- MATRIX-0: composed relations as sparse adjacency products (scipy CSR, CPU; iGPU later) vs SQL joins for BULK closure
  (all chains for all entities, TRI-1 worlds). Report wall time per closure; PASS if matrix closure beats SQL bulk joins
  >= 3x at 10 relations x 300 entities and stays exact.
- EMB-CACHE-0: on the live fleet text stream (events.jsonl), exact-hash and near-duplicate rate (near = same SimHash /
  same ring mark top-1 at cos >= 0.99). PASS: >= 20% of embeds avoidable with zero label changes on the eval heads.
- PEN-C (pod, J's go): contrastive fine-tune of BitNet-embedding-270M's own ternary weights on vault/fleet pairs, nomic
  (with search_query/search_document prefixes fixed) as teacher. PASS: R@5 >= 0.95 on the 09-26 vault test (nomic 0.95,
  BitNet 0.90), ring-head accuracy not worse. B (port nomic to ternary) only if PEN-C misses.
- TRI-HEAD-0 (pod, with PEN-C): ternary triangle head on hidden states -> (s, r, o) corner vectors; teacher triples from
  the chew. PASS: corner recovery >= 95% held-out, pooled embedding bit-identical.
- MEM-ATTN-0 (later): versioned ring memory attention at one layer; R@5 up, no drift within a version.
Perf rule stands (precompute, batch, Pool(4), C packed binding).
Order: EMB-CACHE-0, MATRIX-0, SQ-0 (laptop) -> PEN-C + TRI-HEAD-0 (pod) -> MEM-ATTN-0.

## Wave-3 laptop operationalization (declared 2026-10-01 BEFORE the harnesses ran; J "run these last laptop tests")
- EMB-CACHE-0: stream = control-center/events.jsonl notes in file order (the d5.verb input). Exact = note seen before.
  Near = same 64-bit SimHash over word 3-shingles (lowercased, digits -> 0) as an earlier, non-identical note. Label check:
  300 random near-duplicate pairs (non-identical) sent to the LIVE ring-headd d5.verb (socket; read-only predictions):
  near reuse counts as safe only if labels agree on all pairs; exact reuse is safe by construction (same input -> same
  vector). PASS: avoidable share (exact + safe near) >= 20%.
- MATRIX-0: TRI-1 worlds (function, branch). Bulk closure c_h for h = 1..9 (2..10 hops) for all 300 entities: scipy CSR
  boolean products vs SQLite one bulk join per h (indexes on (p, a)). Exactness = identical (x, z) pair sets per h.
- SQ-0: POLY-0 sessions (current opencode.db, >= 16 steps; first <= 256 steps each). Rows (session, pos, tool, arg).
  Query = (session, tool, arg surface) -> set of positions. Types (1,000 each): EXACT surface; LEXICAL = arg's word tokens
  reversed (single-token args upper-cased); NOISY = 2 random character substitutions (args >= 6 chars). Tools alone:
  SQL (exact index), FTS5 (all query tokens within session+tool), RING (trigram pen via logic.Lexicon over all args ->
  omnitri session polygon -> positions with normalized score >= 0.5). ROUTED = cascade SQL -> FTS5 -> RING -> SQL confirm
  (keep confirmed positions; if none, SQL lookup with the resolved arg). Correct = returned set == truth set. Diagnostic
  arm (not scored): resolve -> SQL. Pen = trigram (one pen; the BitNet pen is PEN-C's job).

## Wave-3 laptop RESULTS (2026-10-01; ~/gpurun-out-0930/{embcache0,matrix0,sq0}.result.json)
- EMB-CACHE-0 PASS: events.jsonl 97,385 notes, 86 distinct -> exact repeats 0.999; SimHash near-dups 0.001 and NOT safe
  (57/60 labels agree) -> avoidable = exact only = 0.999. (shadow.py already skips repeated (verb, note) pairs; ring_headd
  itself has no cache.)
- MATRIX-0 PASS: sparse CSR closure exact; function world 0.4 vs 10.3 ms (23x, 2,700 chains), branch 1.1 vs 114.7 ms
  (102x, 43,207 chains).
- SQ-0 FAIL as registered (latency): accuracy bars MET — exact routed 1.000 (= SQL), lexical 0.905 (FTS5 0.903), noisy
  0.626 (ring 0.580), overall 0.844 vs best single 0.612 — but p95 69 ms (lexical) / 23 ms (noisy) > 20 ms: the FTS5
  join (~44 ms, untuned). Diagnostic resolve->SQL: exact 0.997, lexical 0.343, noisy 0.626.
Disclosure Part 2 written: ~/release/omni-ring/TECHNICAL-DISCLOSURE-PART-2.md (gitleaks clean; uncommitted; J publishes).

## SIMPLEX-0 (J 2026-10-01: "there is no 3d triangle in our system ... the matrices are the base, it makes all other solids")
Built first: omni_ring/solids.py — codebooks whose Gram matrix IS the solid (harmonic frames, exact on K = 16 when
n | 16; tiled with a common phase per copy so the Gram is preserved): simplex, orthogonal, cross_polytope, hypercube,
polygon. tests/test_solids.py 9/9 (Gram exact to 1e-9). Worst-pair separation, 16 atoms at ~5,115 dims: simplex 1.4606
vs random 1.4006.
Pre-registered BEFORE the harness ran (experiments/harness/simplex0.py):
- Triangle buckets as TRI-0 (entities: random 1,000-atom codebook), relation corner = 16 atoms, two arms only differing
  in the relation codebook: RANDOM (uniform phases) vs SIMPLEX (solids.simplex(16)); width 5,115 dims in BOTH arms;
  same seeds for triangles. Fills 256, 512, 1,024, 2,048, 4,096; 8 buckets each; every member queried for its relation
  (two known corners -> relation). Reported: recovery, mean and 5th-percentile margin (true - best other), and the
  absent check: 2,000 (s, o) pairs never stored -> top relation score / median true score.
- PASS: SIMPLEX recovery >= RANDOM + 1 pt at some fill where RANDOM < 99%, OR 5th-pct margin higher by >= 10% at
  every fill. KILL: all differences within +-0.5 pt and margins within +-5% (the simplex only matters at small d).
- Prediction written now: small gain — at d ~ 5k random 16 atoms are already near-orthogonal (|cos| ~ 0.014-0.05)
  while the simplex buys a constant -1/15; expect the effect to show only near capacity.
- SIMPLEX-0 RESULT: PASS (recovery leg). Relation recovery random vs simplex: 256 0.9995 = 0.9995; 512 0.9929 vs 0.9932;
  1,024 0.896 vs 0.912 (+1.6); 2,048 0.672 vs 0.697 (+2.5); 4,096 0.447 vs 0.463 (+1.6). 5th-pct margin +9% (256),
  +29% (512). Absent/present p99 0.522 -> 0.496 (256), 0.739 -> 0.716 (512), 1.042 -> 1.015 (1,024). Prediction held:
  a real but small gain, showing near capacity. Next: simplex where codebooks are small and dims few (relations,
  triage classes, ring-head labels, the 320-dim pyramid prefix), orthogonal sets for roles.

## DESIGNED-0 (J 2026-10-01 "approved": rebuild the triangle memory on designed matrices) — pre-registered BEFORE running
Arms differ ONLY in codebooks; permutations, seeds, triangles, queries, thresholds-fitting procedure identical to the
baseline runs (TRI-0 and TRI-2b, wave 1/2, random codebooks):
- DESIGNED: entity atoms = exact orthogonal set from a Kronecker product of 16-point harmonic frames (16 x 16 x 4 = 1,024
  mutually orthogonal 1,024-dim rows, exact on the 16-phase torus, tiled x5 to 5,120 with a common phase per copy);
  relation atoms = the same construction at 64 (16 x 4), tiled x80. Pairwise cosine exactly 0 within each codebook.
- Measured: TRI-0 capacity curve (16..512 per bucket, int8, all three corners) and 100k-fact strict recovery; TRI-2b
  triage at fill 256 and 384, plain and core arms.
- PASS: TRI-0 int8 cap (largest fill with all corners >= 99%) above the baseline's 256, OR TRI-2b at fill 384 plain
  usable (coverage >= 0.90 with +1 precision >= 0.99 and absent committed <= 0.5%). KILL: every metric within noise of
  baseline or worse.
- Declared risk: tiling gives each atom only 1,024 independent dimensions; cross-talk between triangles may then behave
  like d = 1,024, not 5,120 (permutations differ per corner, which should break the alignment) — the test decides.
- DESIGNED-0 RESULT: FAIL as registered (neither leg), not a clean KILL. Codebooks exactly orthogonal (max |off-diag|
  2.2e-16). TRI-0: int8 cap 256 = baseline; 384: S/R/O 95.7/99.5/95.1 vs 95.8/99.4/95.8; 512: 88.7/97.9/87.5 vs
  87.4/97.7/87.2; 100k strict S/R/O 99.50/100/99.57 vs 99.43/100/99.33. TRI-2b 256 plain: precision 99.35 vs 99.30,
  coverage 96.8 vs 96.7, absent committed 0.12% vs 0%, contradictions 97.5 vs 96.7; 256 core: precision 99.79 vs 99.50,
  absent 0 vs 0.28%; 384 plain coverage 38.8 vs 29.8 (still unusable, false contra 42%); 384 core coverage 42.6 vs 0.2
  (false contra 52%). Reading: at d = 5,120 with 1,000/50 atoms, random atoms are already orthogonal to ~1/sqrt(d);
  the limit is superposition cross-talk (~sqrt(N/d) for N facts per bucket), which atom geometry cannot remove. The
  tiling risk did not materialise (no loss). Designed matrices pay where codebooks are small (SIMPLEX-0: +1.6..2.5 pts
  near capacity) — the next lever for crowded buckets is the bundle itself (structured superposition), not the atoms.

## CORE experiments (J 2026-10-01: "ensure this vision is saved, then build and run any experiments we have on this")
Vision saved: the author's torus-vision note (unpublished) (+ photo) and memory torus-vision-source. Pre-registered BEFORE running:

QUAT-0 — first 4-D core (S^3, unit quaternions). experiments/harness/quat0.py.
- World = TRI-1's translational tree (500 nodes, 10 relations, node = parent bound with relation vector; a node's
  relation labels are distinct among its siblings). Query = start node bound with the path's relation vectors in order,
  one clean-up against all 500 nodes; hops 1..10; every node with a long-enough path queried.
- Arms: TORUS (16-phase, D = 5,120, bind = add mod 16; commutes) vs QUAT (unit quaternion per coordinate, bind =
  right quaternion multiplication; does not commute), QUAT at D = 1,280 (same float count as 5,120 complex? no — report
  bytes) and D = 5,120. Collision = more than one node scoring >= 0.999.
- PASS: QUAT exact recovery >= 99.5% at every hop count with 0 collisions, where TORUS shows collisions (TRI-1: 95-96%).
  KILL: QUAT collisions not below TORUS. Cost (us per bind, bytes per vector) reported.

SOURCE-0 — the source: an intent-driven, persistent resonator loop. experiments/harness/source0.py.
- Memory = Part 1 coprime rings (3, 5, 7, 13 + check 17), float codebooks, D = 5,120; slot = bundle of M marks
  (M = 8, 12, 16); intent = the target mark's 13-ring residue (unique in its slot), as RING-1's single clamp.
- BASELINE: omni_ring.resonator.decode_soft with that clamp (one shot, 15 iterations).
- SOURCE: same unbind/project step, but the state persists (momentum: est <- normalise(0.5 est + 0.5 new)), activations
  sharpen over ticks (softmax with temperature rising linearly 1 -> 20 over 40 ticks) and the intent re-enters every tick.
- Score: target recall (CRT of decoded data residues == target), 200 slots per M, same slots both arms.
- Intent switch: after 40 ticks on target A, the intent changes to target B (another mark in the slot); success = B
  decoded within 40 more ticks without restart.
- PASS: SOURCE recall >= BASELINE + 10 pts at M = 12 (two-proportion p < 0.05) AND switch success >= 80% at M = 12.
  KILL: SOURCE recall not above BASELINE at any M.
- QUAT-0 RESULT: PASS. Unit-quaternion core (S^3): exact 1.000 at every hop count 1-10, 0 collisions, at D = 1,280 and
  5,120. Torus (same world): 0.946 / 0.945 / 0.950 / 0.965 / 0.982 / 0.983 / 0.993 / 1.000 / 1.000 / 1.000 with
  54 / 54 / 46 / 28 / 12 / 8 / 2 / 0 / 0 / 0 commutative collisions. Cost: 20,480 B per vector (D = 1,280, float32
  quaternions) vs 2,560 B (torus, 4-bit); bind 410 vs 105 us in numpy at 5,120 coordinates. Next: quantised
  quaternions (finite subgroups of S^3, e.g. the 24-element binary tetrahedral group) to restore integer exactness.
- SOURCE-0 RESULT: FAIL as registered (not a KILL). Target recall baseline vs source: M = 8 0.940 vs 0.975 (p 0.041);
  M = 12 0.880 vs 0.915 (p 0.12, bar +10 pts not met); M = 16 0.790 vs 0.835 (p 0.12). Intent switch without restart:
  0.25-0.27 (bar 0.80) — momentum holds the old attractor. Reading: a persistent, intent-fed loop helps a little and
  consistently; following a changed intent needs the state to release (decay or reset of the non-clamped rings on
  intent change), not more persistence.

## QUAT-1 + SOURCE-2 (J 2026-10-01 "approved, build and test") — pre-registered BEFORE running
QUAT-1 — quantised 4-D core: every coordinate is one of the 24 unit quaternions of the binary tetrahedral group 2T
(closed under multiplication, non-commuting). Stored as a byte index; bind = 24x24 multiplication-table lookup (integer,
exact); similarity = mean of a 24x24 dot table. Same tree world as QUAT-0, D = 5,120.
- PASS: exact >= 99.5% at every hop 1-10 with 0 collisions, at 5,120 B/vector (vs float quaternions 20,480 B at
  D = 1,280). KILL: collisions as many as the torus. Reported: bind us (numpy table lookup) vs torus.
SOURCE-2 — the source as code on the torus. experiments/harness/source2.py. The source loop is an interpreter: each tick it
decodes its own program vector (4 instruction slots + 1 order-sensitive checksum slot, bound to random position atoms,
bundled, int8-stored; instruction set NOP, HOLD m, SHARPEN rate, RELEASE, FREEZE ring, FOCUS ring) and executes it on
the Part 1 rings (same slots and intents as SOURCE-0). A program whose checksum fails is refused (last good program kept).
- T1 recall with the normal program (HOLD 0.5, SHARPEN): >= SOURCE-0 - 2 pts at M = 8 / 12 / 16.
- T2 intent switch: on a new intent the controller writes RELEASE for one tick, then the normal program: >= 80% at M = 12.
- T3 live rewrite: at tick 20 write FREEZE ring 1: ring 1's estimate unchanged at tick 21 while the others move, in
  100% of 100 trials; unfreezing resumes updates.
- T4 safety: 200 tampered programs (one instruction swapped, checksum stale) and 200 random vectors refused 100%;
  false refusals of 200 clean programs = 0.
- Measured, no bar: ticks per second (Python), sift share of a tick.
- QUAT-1 RESULT: PASS. 2T (24 unit quaternions; 29.2% of element pairs commute) closed under multiplication; bind = byte
  table lookup. Exact 1.000 and 0 collisions at every hop 1-10, 5,120 B/vector (5 bits/coord packable -> 3,200 B), vs
  float quaternions 20,480 B. Bind 60 us (numpy fancy indexing) vs torus 6.2 us; a C table lookup is the next step.
- SOURCE-2 RESULT: T1 PASS, T2 PASS, T3 FAIL as registered, T4 FAIL as registered.
  T1 recall with the normal program = SOURCE-0 (0.975 / 0.915 / 0.830 vs 0.975 / 0.915 / 0.835).
  T2 intent switch with a self-written RELEASE: 0.985 / 0.935 / 0.830 at M = 8 / 12 / 16 (SOURCE-0: 0.265 / 0.250 /
  0.270).
  T3 live rewrite 0.43. DIAGNOSTIC (labelled, not a re-score): FREEZE took effect on the next tick in 100/100, the other
  rings moved in 99/100; the failing leg was "unfreeze resumes" (43/100) — a converged ring is at its fixed point, so a
  resumed update leaves it numerically unchanged. The measure confused "updates again" with "value changes"; mine.
  T4 tampered refused 0.985, random refused 0.980, clean accepted 1.000 — exactly the 1/48 chance a single 48-way
  checksum collides (2.1%). Fix: a second, independent check slot (collision 1/2,304) or a check-ring-style residue.
  Timing: 272 ticks/s in Python (3.7 ms/tick), program decode 0.61 ms/tick.

## Wave-4 (J 2026-10-01 "prepare and run any and all tests we've accumulated") — ALL pre-registered BEFORE any harness
SOURCE-3 (source2 + fixes): two independent check slots (c1 = sum (k+1) id mod 48, c2 = sum p_k id mod 47 mapped onto
  instruction atoms; p = 3, 5, 7, 11). T3 measured as UPDATE EVENTS (which rings the interpreter updated each tick), not
  value changes. PASS: T4 1,000 tampered + 1,000 random refused 100%, 200 clean accepted 100%; T3 FREEZE excludes ring 1
  from the update set on the next tick and NORMAL re-includes it, 100/100; T1/T2 unchanged within 2 pts of SOURCE-2.
QUAT-2 (the prism's core): triangle buckets on the 2T core — triangle = P_S(s) * P_R(r) * P_O(o) per coordinate
  (quaternion product, order kept); bucket = sum of quaternions, int8 per component; unbind by multiplying with the
  inverse (conjugate) on the correct side; clean-up = mean quaternion dot. Same N grid / queries as TRI-0 (1,000
  entities, 50 relations, 12 buckets, 64 queries x 3 corners). PASS: int8 capacity (all corners >= 99%) >= 256 (the
  torus's). KILL: capacity < 128. Bytes per bucket reported (2T: 4 x int8 x 5,120 = 20 KB vs torus 10 KB).
TILE-1 (the seam-matched hexahedron): omniring_sift_batch_tiled tile (TQ queries x TM marks), d5.verb_log (8,738 x 5,120,
  45 MB), 256 queries. Current = (32, 64). Derived from this CPU (L2 256 KB/core): largest tile with TM*d + TQ*2d <= L2
  -> (TQ, TM) = (8, 32). Grid TQ {4,8,16,32,64} x TM {8,16,32,64,128,256}. All outputs must be bit-identical.
  PASS: derived >= 10% faster than current at 1 AND 4 threads. Grid best reported either way.
IPC-0 (the fast seam between languages): round trip of a 1 KB request/response, 10,000 trials (spawn: 200), median and
  p99: (a) spawn a Python process per request, (b) persistent Python server over a Unix socket, (c) shared-memory ring
  with a persistent C worker (single-producer/single-consumer, sequence counters). Plus the real path today: one
  bin/fleet-exec read op as a subprocess. PASS: (c) median <= 50 us and >= 1,000x faster than (a).
FEED-0 (the cube as feeder, real workload): replay 2,000 consecutive live events.jsonl notes through ring_headd (d5.verb).
  A = today (one socket request per note). B = feeder: exact-text digest cache in front + unique misses sent first in
  one burst, answers served from the digest. Same labels required (every B answer == A answer). Measured: notes/s,
  p99 per note, CPU package temperature at end. PASS: >= 1.5x notes/s, identical labels, temperature not higher by > 3 C.
Deferred (need design first, no bars yet): cube cores on D4; three sources (prism / cubes / sphere) end to end.
- WAVE-4 RESULTS (2026-10-01; ~/gpurun-out-0930/{source3,quat2,tile1,ipc0,feed0}.result.json):
  SOURCE-3 PASS (T1-T4): recall 0.975/0.915/0.830, switch 0.985/0.935/0.830; live rewrite by update events 100/100;
    tampered 1,000/1,000 and random 1,000/1,000 refused, clean 200/200 accepted; 544 ticks/s, decode 0.42 ms.
  QUAT-2 PASS: 2T triangle buckets int8 capacity 384 (torus 256); 512: S/R/O 99.2/99.9/99.0% (torus ~87%). Bytes per
    bucket 20 KB vs 10 KB — per byte about even; the 4-D core adds exact order (QUAT-1) at no capacity cost.
  TILE-1 PASS: derived L2-fitted tile (8 x 32) 1.18x (1 thread), 1.33x (4 threads) over (32 x 64); grid best 1.41x
    (4x128 at 1t, 4x8 at 4t); all 60 shapes bit-identical. Ship: default tile per CPU from cache sizes + autotune.
  IPC-0 PASS: 1 KB round trip median — shared-memory ring + resident C worker 0.9 us (p99 1.2), resident Python socket
    9.5 us, spawned Python 32,322 us, today's fleet-exec subprocess 169,696 us (~190,000x the shm ring).
  FEED-0 PASS as registered, with a strong caveat: the last 2,000 live notes had only 2 distinct texts, so the gain
    (299x, labels identical, 78 vs 79 C at end) is the digest cache alone; tiling/staging/batching were not exercised.
    FEED-1 must use a varied stream (the eval sets) to test the feeder's other layers.
- TILE-1 SHIPPED 2026-10-01: omni_ring.seed.tile_shape(d) (L2-fitted, OMNIRING_TILE override) used by sift_batch; omni-ring
  165 passed; ring_headd C++ uses omniring_sift_batch_tiled with the same rule (RING_TILE override) for batches >= 4;
  check.py PASS (0 label mismatches, d5.verb + d5.verb_log); live server restarted under supervise.sh and answering.
> Correction (2026-10-01, TRUEFORM TF-4b, `TRUEFORM plan (unpublished)`): the 1,000-trial random-vector result was
> small-sample. Over 20,000 uniform random program vectors per arm, both random-atom and orbit slot positions accept
> ~0.04% (8 and 9 of 20,000; analytic (1/48)^2 = 0.043%). Single-instruction tampering is always caught (c2 mod 47 is
> prime). Two small check slots are an error-DETECTING code, not a security boundary; a much larger check space is needed
> before program integrity can be claimed against random or adversarial vectors.


## IPC-1 (J approved: the fast seam for omnitool/talon) — pre-registered BEFORE running
bin/fleet-execd: a fork server. Loads bin/fleet-exec (and its preflight/guard imports) once; per request (Unix socket,
JSON {argv, cwd, env, stdin}) it forks a child that sets cwd/env, runs the unchanged main(), and returns stdout/stderr/
exit code. talon's OmnitoolBridge uses the socket when FLEET_EXEC_SOCK exists, else the subprocess (unchanged).
PASS: (1) median talon tool-call latency for a read op and a bash op <= 20 ms (today ~170 ms); (2) identical results
(JSON equal after removing timing/duration/pid/log_path fields) on 20 operations incl. a guard-blocked op and a
malformed payload; (3) talon test suite unchanged with the daemon on.
- IPC-1 RESULT: (1) latency FAIL as registered — talon tool call bash 197.7 -> 38.4 ms, write 179.0 -> 22.9 ms
  (5-8x faster; bar 20 ms); (2) PASS 20/20 identical after a real bug the test caught (workers inherited SIGCHLD=ignore
  and lost exit codes; fixed by resetting it in each worker); (3) PASS talon suite 21/21 with and without the daemon (a
  second bug caught: the daemon path skipped talon's result unwrapping; both paths now share it; the caller's full
  environment is forwarded so test isolation holds). Remaining per-call cost is inside fleet-exec itself (marks/area
  writes, guard, events) + fork: next lever = profile and trim per-op overhead. bin/fleet-execd runs by hand
  (/tmp/fleet-execd.sock); opt-in via FLEET_EXEC_SOCK; a supervised unit is J's call (systemd writes are blocked here).
## NOMIC-POLY-0 (J 2026-10-01: "the matrices will be made of nomic embed fts5 vectors, each vector will have thousands of
## points of data, and they will be recoverable with 100% accuracy") — pre-registered BEFORE running; experiments/harness/nomic_poly0.py
Data: vault-indexer/index/default__vector_store.json = 6,194 real vault embeddings x 1,024 floats (the codebook / the
square). Encode each embedding onto the 16-phase torus (D = 5,120) by a fixed random projection: phase = angle of
(R1 e + i R2 e) quantised to 16 (meaning-preserving: near embeddings -> near phases). Polygon = int8-quantised complex sum of
M items, item k bound to position atom k (phase add). Read slot k: unbind (phase subtract), sift against all 6,194 codebook
phase vectors, top-1 id -> the stored 1,024 floats returned from the row (bit-exact if the id is right).
Groups of M drawn at random from the vault, M = 16, 32, 64, 128, 256, 384; 40 polygons per M (seed 0).
Exact = returned 1,024 floats identical to the true item's (duplicate-embedding chunks count as exact; distinct-id rate
reported too). Abstain rule: commit only if margin (top1 - top2)/top1 >= tau, tau fixed at 0.05 BEFORE running.
- PASS: exact top-1 >= 99% at M = 64, AND committed precision = 100% at M = 128 with coverage >= 90%.
- KILL: exact < 90% at M = 16.  Reported: bytes per polygon vs raw floats, ms per read.
- NOMIC-POLY-0 RESULT: FAIL as registered (not a KILL). Exact (1,024 floats returned bit-identical) 0.997 / 0.983 / 0.953
  / 0.765 / 0.361 / 0.186 at M = 16 / 32 / 64 / 128 / 256 / 384; committed precision 1.000 to M = 64, 0.991 at 128 but
  coverage 0.38. Polygon 10,240 B vs raw 262,144 B at M = 64.
  DIAGNOSTIC (labelled, not a re-score; nomic_poly0_diag.py, 20 polygons): the vault embeddings sit in a narrow cone (mean
  pairwise cosine 0.82), so meaning-preserving phases crowd together. WHITENED (mean-centred, SVD-whitened) encoding:
  exact 1.000 / 0.997 / 0.976 at 64 / 128 / 256, committed precision 1.000 / 1.000 / 0.992, coverage 0.99 / 0.98 / 0.95.
  Random id atoms (meaning kept in the row): 1.000 / 1.000 / 0.994. Shape fix = whiten the meaning space before the torus.
## NOMIC-POLY-1 — pre-registered BEFORE running: NOMIC-POLY-0 with the WHITENED encoding, fresh seed 1, 40 polygons per M,
## same bars (exact >= 99% at M = 64; committed precision 100% at M = 128 with coverage >= 90%), tau 0.05.
- NOMIC-POLY-1 RESULT: leg 1 PASS (exact 0.9996 at M = 64, coverage 0.989, committed precision 1.000); leg 2 FAIL as
  registered by ONE item: M = 128 committed precision 0.9996 (1 wrong of ~2,505 committed), coverage 0.979, exact 0.996.
  M = 256 exact 0.971, committed 0.991, coverage 0.945. 10,240 B polygon holds 64-128 vault embeddings (262-524 KB raw),
  each returned as its exact 1,024 floats. Next (to register): the 100% leg via a second check — TRI-2's closing check
  or a check-ring residue stored per slot (the SOURCE-3 fix pattern); then the 2T core arm (QUAT-2 took 256 -> 384).
- CORRECTION (2026-10-01, later): the NOMIC-POLY-0/1 vectors are mistral-embed (1,024-d, the cloud-reindexed vault store,
  INDEX_EMBED_BACKEND=mistral), not nomic. Geometry result stands for real vault embeddings; nomic re-run after the vault is
  embedded locally on our bitnet.cpp build (EMBED-BN-0: cosine 1.00000 vs ollama nomic; see OMNID plan (unpublished) §2).
