#!/usr/bin/env python3
"""MEM-1B (GCP-B1 hypervector arm, GCP-CPU-0): MEM-1M's residue-address decode (`projects/condense/t50/mem1m.py`) with N, moduli and
payload width from argv, so the VM can run it at N = 1,000,000,000. Same path as `_Pads/GPU-RUN/mem0.py:49-62`: pack the address into a
hypervector (one codeword per ring, bundled, 16-phase quantised), unpack it by direct_probe (one dot product per candidate residue per
ring, argmax per ring, CRT over data + check rings), fetch the payload row. Only the RingSpec and the payload table change.
Default spec: data moduli 997·1009·1013 = 1,019,050,649 >= 1e9, check 1019 (Σ = 4,038 probes); D 5,120; payload 8 B per item
(1e9 x 8 B = 8 GB table, filled in chunks). Keys are drawn over all N stored items.
usage: mem1b.py OUT.json N [LOOKUPS=1000] [PAYLOAD_BYTES=8] [m1 m2 m3 ... check]   (last modulus = check ring)
Runs MEM-0's own spec (3·5·7·13, check 17) first in the same process as the control (must reproduce recall 1.0); the big run is the
last key of OUT.json (the GCP-B1 gate reads list(r.values())[-1]).
"""
import json, os, resource, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "packages/omni-ring"))
from omni_ring.types import RingSpec
from omni_ring.codebook import generate_direct_codebooks
from omni_ring.crt import direct_probe

OUT, N = sys.argv[1], int(sys.argv[2])
NLOOK = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
PB = int(sys.argv[4]) if len(sys.argv) > 4 else 8
MODS = tuple(int(a) for a in sys.argv[5:]) or (997, 1009, 1013, 1019)
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