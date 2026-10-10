#!/usr/bin/env python3
"""MATRIX-0 (PREREGISTRATIONS-OMNITRI.md wave-3): bulk relation closure as sparse matrix products vs SQLite bulk joins."""
import json, os, sqlite3, sys, time
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import scipy.sparse as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tri_wave import rule_world
res = {}
for branch in (False, True):
    facts, _ = rule_world(branch); n = 300
    A = {}
    for p, a, b in facts:
        A.setdefault(p, []).append((int(a), int(b)))
    M = {p: sp.csr_matrix((np.ones(len(v), np.int8), ([x for x, _ in v], [y for _, y in v])), shape=(n, n)) for p, v in A.items()}
    db = sqlite3.connect(":memory:"); db.execute("create table f(p text, a int, b int)")
    db.executemany("insert into f values (?,?,?)", [(p, int(a), int(b)) for p, a, b in facts]); db.execute("create index ix on f(p, a)")
    def mat():
        out, C = {}, M["r0"]
        for h in range(1, 10):
            C = (C @ M[f"r{h}"]); C.data[:] = 1; out[h] = C
        return out
    def sql():
        out = {}
        for h in range(1, 10):
            j = " ".join(f"join f t{i} on t{i}.a = t{i-1}.b and t{i}.p = 'r{i}'" for i in range(1, h + 1))
            out[h] = set(db.execute(f"select distinct t0.a, t{h}.b from f t0 {j} where t0.p = 'r0'").fetchall())
        return out
    def best(f):
        r = f(); b = 1e9
        for _ in range(3):
            t = time.perf_counter(); f(); b = min(b, time.perf_counter() - t)
        return r, b
    mo, tm = best(mat); so, ts = best(sql)
    exact = all({(int(i), int(j)) for i, j in zip(*mo[h].nonzero())} == so[h] for h in range(1, 10))
    pairs = sum(len(so[h]) for h in range(1, 10))
    res["branch" if branch else "function"] = {"matrix_ms": tm * 1000, "sqlite_ms": ts * 1000, "x": ts / tm, "exact": exact, "pairs": pairs}
    print(f"MATRIX {'branch' if branch else 'function'}: closure 2..10 hops, {pairs} pairs | matrix {tm*1000:.1f} ms | sqlite {ts*1000:.1f} ms | {ts/tm:.1f}x | exact {exact}", flush=True)
res["verdict"] = {"PASS": all(r["x"] >= 3 and r["exact"] for r in res.values())}
print("MATRIX-0:", res["verdict"], flush=True)
json.dump(res, open(os.path.expanduser("~/gpurun-out-0930/matrix0.result.json"), "w"), indent=1, default=str)
