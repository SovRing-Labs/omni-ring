#!/usr/bin/env python3
"""MEM-1B (GCP-B1 hypervector arm, GCP-CPU-0): MEM-1M's residue-address decode (`projects/condense/t50/mem1m.py`) with N, moduli and
payload width from argv, so the VM can run it at N = 1,000,000,000. Same path as `_Pads/GPU-RUN/mem0.py:49-62`: pack the address into a
hypervector (one codeword per ring, bundled, 16-phase quantised), unpack it by direct_probe (one dot product per candidate residue per
ring, argmax per ring, CRT over data + check rings), fetch the payload row. Only the RingSpec and the payload table change.
Portable copy: imports and CLI validation adapted for this public repository; numerical loop retained.
Default spec: data moduli 997·1009·1013 = 1,019,050,649 >= 1e9, check 1019 (Σ = 4,038 probes); D 5,120; payload 8 B per item
(1e9 x 8 B = 8 GB table, filled in chunks). Keys are drawn over all N stored items.
usage: mem1b.py OUT.json N [LOOKUPS=1000] [PAYLOAD_BYTES=8] [m1 m2 m3 ... check]   (last modulus = check ring)
Runs MEM-0's own spec (3·5·7·13, check 17) first in the same process as the control (must reproduce recall 1.0); the big run is the
last key of OUT.json (the GCP-B1 gate reads list(r.values())[-1]).
"""
import argparse, itertools, json, math, os, resource, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from omni_ring.types import RingSpec
from omni_ring.codebook import generate_direct_codebooks
from omni_ring.crt import direct_probe

parser = argparse.ArgumentParser(description="MEM-1B clean-cue address lookup; see README.md for memory limits.")
parser.add_argument("output", type=Path)
parser.add_argument("n", type=int)
parser.add_argument("lookups", type=int, nargs="?", default=1000)
parser.add_argument("payload_bytes", type=int, nargs="?", default=8)
parser.add_argument("moduli", type=int, nargs="*")
parser.add_argument("--allow-large", action="store_true",
                    help="Allow a payload larger than 256 MB (full billion run needs substantial RAM).")
args = parser.parse_args()
OUT, N, NLOOK, PB = str(args.output), args.n, args.lookups, args.payload_bytes
MODS = tuple(args.moduli) or (997, 1009, 1013, 1019)
if min(N, NLOOK, PB) <= 0:
    parser.error("N, LOOKUPS and PAYLOAD_BYTES must be positive")
if len(MODS) < 2 or any(m < 2 for m in MODS):
    parser.error("Supply at least two moduli, all >= 2 (last is the check ring)")
if any(math.gcd(a, b) != 1 for a, b in itertools.combinations(MODS, 2)):
    parser.error("All moduli must be pairwise coprime")
if math.prod(MODS[:-1]) > 2**63 - 1:
    parser.error("Data capacity must fit signed int64")
if N * PB > 256_000_000 and not args.allow_large:
    parser.error("Payload exceeds 256 MB; read the memory requirements and explicitly pass --allow-large")
if args.output.exists():
    parser.error("Output already exists; choose a new path to preserve previous evidence")
# This guard bounds this harness's codebook allocation even with arbitrary input moduli.
if sum(MODS) > 10000:
    parser.error("Sum of moduli exceeds this harness's 10000-codeword allocation limit")
big = RingSpec(moduli=MODS[:-1], check_modulus=MODS[-1])
if big.data_capacity < N:
    sys.exit(f"address space {big.data_capacity} < N {N}: give larger moduli")
CHUNK = 50_000_000
res = {"argv": sys.argv[1:], "cpu": next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), ""),
       "nproc": os.cpu_count()}
for name, spec, nstore, nlook, pb in (("MEM0_control_1365", RingSpec(), None, 200, 64),
                                      (f"MEM1B_{N}", big, N, NLOOK, PB)):
    g = np.random.default_rng(0)
    n = nstore or spec.data_capacity
    t = time.time(); dcb = generate_direct_codebooks(spec, seed=0); dcbc = [np.conj(c) for c in dcb]; cb_s = time.time() - t
    t = time.time(); payload = np.empty((n, pb), dtype=np.int8)
    for i in range(0, n, CHUNK):
        payload[i:i + CHUNK] = g.integers(-127, 127, (min(CHUNK, n - i), pb), dtype=np.int8)
    build_s = time.time() - t

    def lookup(x):
        v = sum(cb[x % m] for cb, m in zip(dcb, spec.all_moduli))
        ph = np.round(np.angle(v) * 16 / (2 * np.pi)).astype(int) % 16
        _, xd = direct_probe(np.exp(1j * ph * 2 * np.pi / 16), dcb, spec.all_moduli, dcbc)
        return xd

    ts, ok, wrong = [], 0, []
    for x in g.integers(0, n, nlook):
        x = int(x); t1 = time.time(); xd = lookup(x); row = payload[xd] if 0 <= xd < n else None; ts.append(time.time() - t1)
        if xd == x and row is not None and np.array_equal(row, payload[x]): ok += 1
        elif len(wrong) < 10: wrong.append([x, int(xd)])
    probes = sum(spec.all_moduli)
    res[name] = {"N_stored": n, "address_space": spec.data_capacity, "moduli": list(spec.all_moduli), "probes": probes,
                 "dimension": spec.dimension, "payload_bytes": pb, "lookups": nlook, "recall": ok / nlook, "wrong_examples": wrong,
                 "ms_median": 1000 * float(np.median(ts)), "ms_p95": 1000 * float(np.percentile(ts, 95)), "bytes_scanned": 0,
                 "bytes_touched": probes * spec.dimension + pb,   # MEM-0's accounting: probes x D (int8-equivalent) + one payload row
                 "table_bytes": n * pb, "codebook_s": cb_s, "payload_build_s": build_s,
                 "rss_peak_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
    print(name, json.dumps(res[name]), flush=True)
    del payload
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)
print("done")
