# omni-ring — Dossier

*History, specification, metrics, tests and limits. Companion to `TECHNICAL-DISCLOSURE.md` (Part 1),
`TECHNICAL-DISCLOSURE-PART-2.md` and `README.md`. The full research record and essays: github.com/SovRing-Labs/omniring-research.*

## What it is

A memory that stores small structured records ("marks") in fixed-size high-dimensional vectors and reads them back by
resonance or exact sift — with a check ring that detects corruption and the ability to abstain. In the stack it is
the **memory**, sitting on the vsa-core bus. Part 2 extends it into a portable specification (SuperSeed), fast
geometric kernels and a triangle memory for facts. Stage (essay 07): primitive → substrate.

## Creation history

| Date (2026) | Milestone |
|---|---|
| Sep, study threads | The author's image: rings of different sizes turning around a resonator (corpus essay 01) |
| 09-27 | Resonator-VSA review panel and literature dossiers (resonator networks, VSA encodings, dynamics and codes) |
| 09-28 | Coprime rings measured (RING-1 series): cued recall, the 17 check ring (100% single-slip detection over 54,600 cases), direct keyed reads; a tick-desync bug fixed (separate data and check ticks); AVX2 `vpshufb` kernel and the two-faced bus built |
| 09-29 | Portal integration reviewed and cut back to a "dial, not an engine" after a review found a green gate over code that shipped zero bytes; peeling phantom-mark bug fixed (null-residual guard + eigengap floor); exhaustive lattice sift: 24–192 marks per slot with 0 fakes; release export f2e189a |
| 09-29 | Decision: publish as prior art under PolyForm (no patents) |
| 09-30 | SuperSeed v1/v1.1 portable spec; ring heads on fleet data; tie-inclusive top-k; cube tiling and pyramid kernels; integrated-GPU engine; published v0.1.0 (Software Heritage save succeeded) |
| 10-01 | Triangle memory, polygons, nesting, substrate experiments; Technical Disclosure Part 2 published as v0.1.1 |

## Specification (summary)

* **Part 1:** coprime rings 3, 5, 7, 13 (1,365 addresses) + check ring 17; phase face (5,120 dims, K = 16) and ternary
  face (10,240 dims) over one 2,560-byte slot; cue-driven soft resonator; peeling; diamond facets; mark store with
  decay; shards. Disclosure §2–11.
* **Part 2 (code in weekly updates):** SuperSeed integer spec + golden conformance; packed block-64 layout; cascade;
  tie-inclusive top-k; ring heads with margin abstention and provenance; cube / pyramid sift; triangle memory (three
  edge indexes, second-path confirmation, triage trit, decay, harmony, polygons, transforms, nesting). Disclosure Part 2.

## Metrics

Part 1 (this release's code; Intel i5-8365U):

| Property | Result |
|---|---|
| Single-ring slip detection (17 ring) | 100%, 54,600 cases |
| Cued recall, 12 marks/slot, one ring clamped | 67% (chance 8%); two clamps 93% |
| Exhaustive sift, 1,365 codewords | 24–192 marks/slot all read, 0 fakes |
| Peeling, 24 marks/slot | 16.65 true, 0 / 333 fakes |
| AVX2 sift | 0.525 µs per row; 1,365-row firing 0.72 ms |
| Shared-memory round trip | 4.41 µs |

Part 2 (research harness; code to follow): ~12x faster exact search (cubes 2.1–2.6x, pyramid 4.9x); triangles ≥ 99% to
256 per bucket; chaining exact to 10 hops; two-level nesting 0.9995 over 262,144 facts. Full table: Part 2 §8.

## Tests

`./build_kernel.sh && python3 -m pytest -q` — 111 passed on the 0.1.0 export (the development tree with Part 2 modules:
156 passed). Gates in the README: separation, cycle, resonator, clamped recall, check ring, latency, transcoder,
seqlock.

## Limits

* A cue is mandatory for resonance; unguided decoding is seed-dependent and abstains when it fails.
* Resonant capacity ≈ 12 marks per slot at one cue (the exhaustive sift reads more).
* Superposition is finite: triangle buckets are safe to 256 facts; beyond that, truth must live in exact rows.
* Exact multi-hop lookups are ~250x faster in an indexed database (Part 2 §9).

## Lineage and credits

Kymn et al. (residue HDC), Frady et al. (resonators), Plate, Kanerva, Fiete et al.; Part 2 adds TorusE, RotatE,
Conway–Sloane, Weyl, Jégou et al., Burt & Adelson, Imani et al. and others — thanked in Part 2 §13. `PROVENANCE.md`.
Updates: weekly, recorded here with dates.
