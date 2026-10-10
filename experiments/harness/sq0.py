#!/usr/bin/env python3
"""SQ-0 (PREREGISTRATIONS-OMNITRI.md wave-3): SQL square + FTS5 + ring (trigram pen) + omnitri polygons, routed vs each tool alone."""
import json, os, re, sqlite3, sys, time
import numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "<your-data>/shards")
from omni_ring.logic import Lexicon
from episodes import load, DBS

D = 5120; PH = np.exp(2j * np.pi * np.arange(16) / 16).astype(np.complex64); g = np.random.default_rng(50)
steps, _ = load(DBS["current"])
sess = {s: v[:256] for s, v in steps.items() if len(v) >= 16}
split = lambda tok: tok.split(":", 1) if ":" in tok else (tok, "(none)")  # noqa: E731
sids = sorted(sess); si = {s: i for i, s in enumerate(sids)}
rows = [(si[s], p, *split(t)) for s in sids for p, (t, _) in enumerate(sess[s])]
tools = sorted({r[2] for r in rows}); args = sorted({r[3] for r in rows}); ti = {t: i for i, t in enumerate(tools)}; ai = {a: i for i, a in enumerate(args)}
# SQL square + FTS5
db = sqlite3.connect(":memory:")
db.execute("create table tri(sess int, pos int, tool text, arg text)"); db.executemany("insert into tri values (?,?,?,?)", rows)
db.execute("create index ix on tri(sess, tool, arg)")
db.execute("create virtual table fts using fts5(arg, content='tri', content_rowid='rowid')"); db.execute("insert into fts(fts) values ('rebuild')")
# omnitri polygons (POLY-0 encoding) + trigram pen
TOOL = g.integers(0, 16, (len(tools), D)); ARG = g.integers(0, 16, (len(args), D)); theta = g.uniform(0, 2 * np.pi, D)
P = [g.permutation(D) for _ in range(3)]
pos = lambda p: np.rint(np.outer(np.atleast_1d(p), theta) / (np.pi / 8)).astype(np.int64) % 16  # noqa: E731
roleT, roleA = TOOL[:, P[1]], ARG[:, P[2]]; POSR = pos(np.arange(256))[:, P[0]]
cpos, spos = np.cos(np.pi / 8 * POSR).astype(np.float32), np.sin(np.pi / 8 * POSR).astype(np.float32)
poly = {}
for s in sids:
    T = np.array([(p, ti[split(t)[0]], ai[split(t)[1]]) for p, (t, _) in enumerate(sess[s])])
    a = PH[(POSR[T[:, 0]] + roleT[T[:, 1]] + roleA[T[:, 2]]) % 16].sum(0)
    sc = max(np.abs(a.real).max(), np.abs(a.imag).max()) / 127
    poly[si[s]] = ((np.rint(a.real / sc) + 1j * np.rint(a.imag / sc)) * sc).astype(np.complex64)
lex = Lexicon(d=2048)
for a in args:
    lex.add(a, a)
print(f"SQ-0 sessions {len(sids)}, rows {len(rows)}, tools {len(tools)}, args {len(args)}", flush=True)


def truth(s, t, a):
    return {r[0] for r in db.execute("select pos from tri where sess=? and tool=? and arg=?", (s, t, a))}


def q_sql(s, t, x):
    return truth(s, t, x)


def q_fts(s, t, x):
    toks = [w for w in re.findall(r"\w+", x)]
    if not toks:
        return set()
    m = " AND ".join('"' + w.replace('"', "") + '"' for w in toks)
    try:
        return {r[0] for r in db.execute("select tri.pos from fts join tri on tri.rowid = fts.rowid where fts match ? and tri.sess=? and tri.tool=?", (m, s, t))}
    except sqlite3.OperationalError:
        return set()


def q_ring(s, t, x, confirm=False):
    a, _, _ = lex.resolve(x)
    if a is None or t not in ti:
        return set()
    u = poly[s] * np.conj(PH[(roleT[ti[t]] + roleA[ai[a]]) % 16])
    n = len(sess[sids[s]]); sc = (cpos[:n] @ u.real + spos[:n] @ u.imag) / D
    got = set(np.nonzero(sc >= 0.5)[0].tolist())
    if confirm:
        ok = {p for p in got if db.execute("select 1 from tri where sess=? and pos=? and tool=? and arg=?", (s, p, t, a)).fetchone()}
        return ok or truth(s, t, a)
    return got


def q_resolve_sql(s, t, x):
    a, _, _ = lex.resolve(x)
    return truth(s, t, a) if a is not None else set()


def routed(s, t, x):
    r = q_sql(s, t, x)
    if r:
        return r
    r = q_fts(s, t, x)
    if r:
        return r
    return q_ring(s, t, x, confirm=True)


def variant(kind, a):
    if kind == "exact":
        return a
    if kind == "lexical":
        w = re.findall(r"\w+", a)
        return " ".join(w[::-1]) if len(w) > 1 else a.upper()
    c = list(a)
    for i in g.choice(len(c), 2, replace=False):
        c[i] = "x" if c[i] != "x" else "q"
    return "".join(c)


cands = [r for r in rows if r[3] != "(none)"]
res = {}
for kind in ("exact", "lexical", "noisy"):
    pool = [r for r in cands if kind != "noisy" or len(r[3]) >= 6]
    picks = [pool[i] for i in g.choice(len(pool), 1000, replace=False)]
    qs = [(s, t, a, variant(kind, a)) for s, p, t, a in picks]
    row = {}
    for name, fn in (("SQL", q_sql), ("FTS5", q_fts), ("RING", q_ring), ("ROUTED", routed), ("diag_resolve_sql", q_resolve_sql)):
        ok = 0; lat = []
        for s, t, a, x in qs:
            t0 = time.perf_counter(); got = fn(s, t, x); lat.append((time.perf_counter() - t0) * 1000)
            ok += int(got == truth(s, t, a))
        row[name] = {"acc": ok / len(qs), "p95_ms": float(np.percentile(lat, 95))}
    res[kind] = row
    print(f"SQ-0 {kind:7s} " + " | ".join(f"{k} {v['acc']:.3f} ({v['p95_ms']:.2f} ms p95)" for k, v in row.items()), flush=True)
tools_ = ("SQL", "FTS5", "RING")
overall = {k: float(np.mean([res[q][k]["acc"] for q in res])) for k in tools_ + ("ROUTED",)}
res["overall"] = overall
res["verdict"] = {"PASS": all(res[q]["ROUTED"]["acc"] >= max(res[q][k]["acc"] for k in tools_) for q in ("exact", "lexical", "noisy"))
                  and overall["ROUTED"] >= max(overall[k] for k in tools_) and max(res[q]["ROUTED"]["p95_ms"] for q in res if q != "overall") <= 20}
print("SQ-0 overall:", {k: round(v, 3) for k, v in overall.items()}, res["verdict"], flush=True)
json.dump(res, open(os.path.expanduser("~/gpurun-out-0930/sq0.result.json"), "w"), indent=1, default=str)
