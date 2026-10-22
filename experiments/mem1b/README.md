# MEM-1B: clean-cue residue-address lookup

This is a reproducibility package for the **hypervector arm only** of GCP-B1, recorded on 2026-10-15 UTC. It publishes the [original harness](original/mem1b.py), [original result](results/xeon-8481c.json), a [portable harness](mem1b.py), and [small automated tests](../../tests/test_mem1b.py). See [provenance and hashes](PROVENANCE.json). No private outreach correspondence is included.

## What the recorded result establishes

On an Intel Xeon Platinum 8481C (8 vCPUs; the flight record identifies a 32 GB c3-standard-8 VM), the run allocated **1,000,000,000 random 8-byte payload rows**. It recovered the correct address and matching payload on **1,000 sampled clean-cue lookups**: recall 1.0, median 40.099 ms, p95 40.509 ms. The same process first passed its MEM-0 control, 200/200 lookups over 1,365 rows. These are historical measurements, not a new billion-row rerun or an exhaustive test of every address.

```text
known integer address x
  -> select one codeword per ring using x % modulus
  -> sum codewords, quantize to 16 phases (D = 5,120)
  -> direct_probe: dot products + argmax for each ring
  -> CRT over data AND check rings
  -> range-check decoded address -> fetch random payload row
  -> compare decoded address and payload with ground truth
```

This tests address representation, clean-cue decoding and indexed payload access. **It is not arbitrary-content retrieval, semantic search, noisy-cue robustness, learned association, or an iterative attractor benchmark.** The input address is known and is used to construct the cue inside each timed lookup. The check ring participates in CRT; this harness does not independently exercise corruption detection. Payload equality verification is outside the lookup timer. Lookup timing includes cue construction, decoding and payload-row access, but excludes codebook and payload-table construction. NumPy row indexing normally returns a view: the timer does not force a copy or checksum of all payload bytes; the equality check that reads the payload is outside the timer. The harness uses `time.time()` as the original did.

The table really occupies 8,000,000,000 bytes; the measurement does not store a billion payloads in a single hypervector. The fixed spec uses data moduli 997, 1009, 1013 (capacity 1,019,050,649), check modulus 1019, and 4,038 candidate probes. At fixed moduli and dimension, probe work does not scale with table size; this is not a universal constant-cost scaling claim as moduli grow.

### Accounting caveats

- `bytes_scanned: 0` means no payload-table scan. The codebook candidates **are** probed.
- `bytes_touched: 20,674,568` is the original **int8-equivalent accounting formula** `sum(moduli) * D + payload_bytes`, **not measured DRAM traffic**. Actual codebooks use NumPy complex128; codebooks plus their conjugates occupy approximately 662 MB (decimal), excluding temporaries.
- `rss_peak_mb: 8681.2890625` is Linux process peak RSS expressed in **MiB**, despite the legacy field name; it includes both stages in the process. It is not table size or per-lookup allocation.
- Seeds are fixed at 0. Payload generation and sampled addresses share one RNG. Chunk size and generation order are retained; changing them can change the sampled lookup sequence.
- The historical artifact does not identify the exact Python/NumPy/BLAS versions or dependency commit used on that VM. This public adaptation was checked against the source base recorded in PROVENANCE.json, not certified as a byte-identical reconstruction of that environment. Timings on other machines are not directly comparable.

## Reproduce safely (Linux, Python 3.10+)

From the repository root, use an isolated environment and install NumPy (the public package uses NumPy fallbacks; no GPU or compiled AVX2 kernel is required for this harness):

```sh
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install numpy
python3 -m unittest discover -s tests -p 'test_mem1b.py' -v
# A tiny explicit smoke; use a fresh output path:
python3 experiments/mem1b/mem1b.py /tmp/mem1b-small.json 1000 20 8 3 5 7 13 17
```

The smoke runs the original 200-lookup control first, then 20 lookups over 1,000 rows with small moduli. It tests the path, **not the billion-row capacity or historical timings**. Omitting moduli selects the historical large codebook even for a small N, requiring roughly 1 GB of codebook/temporary headroom. The CLI requires an explicit output path and N, rejects invalid sizes/non-coprime moduli/insufficient capacity, refuses an existing output file, and requires `--allow-large` above a 256 MB payload. The codeword cap is 10,000; these guards are not an OS memory limit.

### Full historical configuration — opt-in, substantial RAM

**Not run as part of publication or the automated tests.** Reserve ample RAM: the original VM had 32 GB, the payload alone is 8 GB, peak RSS was about 8.48 GiB, and allocations/BLAS versions can change the peak. This command fills the entire table and runs 1,000 lookups after the control:

```sh
python3 experiments/mem1b/mem1b.py /tmp/mem1b-full-new.json 1000000000 1000 8 997 1009 1013 1019 --allow-large
```

Keep results separate from `results/xeon-8481c.json`. The portable version changes repository import location and adds argument/allocation/output guards, but retains the original numerical loop, fixed seed, chunk size, timing boundaries and legacy result fields. The archived original is provided for inspection, not as the recommended entry point.
