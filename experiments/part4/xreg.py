#!/usr/bin/env python3
"""Experiment registry (floor F2): every result JSON in ~/gpurun-out-0930 flattened into rows, plus every RESULT line from the
seat plans, into control-center/experiments.db. Re-runnable; rebuilds from scratch. Query with any SQLite client.
Tables: results(exp, file, path, num, text) — one row per JSON leaf; result_lines(exp, plan, verdict, line);
experiments(exp, files, plan, verdict) — one row per experiment id."""
import glob, json, os, re, sqlite3

OUT = os.path.expanduser("~/gpurun-out-0930")
PLANS = "<your-plans>"
DB = "<your-data>/control-center/experiments.db"


def exp_id(fname):
    b = os.path.basename(fname).replace(".result.json", "").replace(".json", "")
    e = b.split(".")[0].upper().replace("_", "-")
    e = re.sub(r"^([A-Z]+)(\d)", r"\1-\2", e)          # QUAT0 -> QUAT-0, TF3B -> TF-3B (join with plan ids)
    e = re.sub(r"([A-Z])(\d+[A-Z]?)$", r"\1-\2", e)    # NOMIC-POLY0 -> NOMIC-POLY-0, EMBED-BN1D -> EMBED-BN-1D
    e = re.sub(r"-(\d+)([A-Z])$", lambda m: f"-{m.group(1)}{m.group(2).lower()}", e)   # TF-3B -> TF-3b, BN-1D -> BN-1d
    return e


def flatten(x, p=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from flatten(v, f"{p}/{k}")
    elif isinstance(x, list):
        if x and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in x) and len(x) > 20:
            yield p + "[len]", len(x), None          # long numeric lists: keep length + mean
            yield p + "[mean]", sum(x) / len(x), None
        else:
            for i, v in enumerate(x):
                yield from flatten(v, f"{p}/{i}")
    elif isinstance(x, bool):
        yield p, int(x), str(x)
    elif isinstance(x, (int, float)):
        yield p, float(x), None
    else:
        yield p, None, str(x)[:200]



# Hand-curated verdicts for experiments with no machine-readable verdict (source in the comment; plan/JSON verdicts win).
CURATED = {
    "BELT-CPP": "PASS",               # TORUS-TEL.plan: BELT-CPP PASS (B1 B2 B3)
    "BELT-0-SWEEP": "MEASURE", "BELT-BASELINE": "MEASURE", "BELT-EPP-PERFORMANCE": "MEASURE",
    "BELT-EPP-POWER": "MEASURE", "BELT-SCX-BPFLAND": "MEASURE",   # measurement arms, no gate (TORUS-TEL / RING-SEED)
    "EMBED-BN1C-EVAL": "FAIL as registered",   # OMNID.plan: nomic 0.680 vs mistral 0.745, bar 0.695
    "HOLO-2": "EXPLORATORY", "HOLO-2b": "EXPLORATORY",          # GEOM.plan: exploratory, no gate
    "HOLO-4-CPU": "KILL", "HOLO-4-GPU": "KILL",                 # GEOM.plan: HOLO-4 H4 KILL
    "IPC-1-EQUIV": "FAIL as registered",      # OMNITRI.plan: IPC-1 latency FAIL as registered (equivalence held)
    "LIMIT-0": "MEASURE",                     # GEOM.plan: LIMIT-0 RESULT (no gate)
    "LOGIC-0": "SPLIT",                       # CORTEX-LOGIC.plan: L1/L3/L4 PASS, L5 KILL as registered
    "LOOP-0": "SPLIT",                        # RING-SEED.plan: L1 pass, L2 FAIL as registered (J: re-scope)
    "NOMIC-POLY0-DIAG": "DIAGNOSTIC",         # OMNITRI.plan: diagnostic after NOMIC-POLY-0
    "PLM-2": "FAIL as registered",            # TRUEFORM.plan: Q1/Q2 fail; Q2 metric invalid (parser)
    "POLY-0": "PASS",                         # OMNITRI.plan wave-1: POLY-0 PASS (a, b, c)
    "RING-HEAD-0": "KILL", "RING-HEAD-1": "PASS", "RING-HEAD-2": "PASS",   # RING-SEED.plan
    "SEED-11-CASCADE": "PASS",                # TORUS-TEL.plan: SEED-1.1 P PASS
    "SOURCE-3": "PASS",                       # OMNITRI.plan wave 4: T1-T4 PASS (random-refusal claim corrected by TF-4b)
    "TORTEL-1": "PASS",                       # TORUS-TEL.plan: TORUS-TEL-1 PASS (P1 P2 P3)
    "XLAW": "ANALYSIS", "XLAW-OPERATING": "ANALYSIS",
}

VERD = re.compile(r"\b(PASS|KILL|FAIL as registered|FAIL|SPLIT)\b")
LINE = re.compile(r"^-?\s*\**([A-Z][A-Z0-9]*(?:-[A-Z0-9]+)*[a-z]?)\**[:]?\s+(?:RESULT\b|(?=PASS|KILL|FAIL|SPLIT|INCONCLUSIVE))(.*)$")

if __name__ == "__main__":
    if os.path.exists(DB):
        os.remove(DB)
    con = sqlite3.connect(DB)
    con.executescript("""create table results(exp text, file text, path text, num real, text text);
        create table result_lines(exp text, plan text, verdict text, line text);
        create table experiments(exp text primary key, files text, plan text, verdict text, field_prediction text);""")
    FP = {}
    fp_path = os.path.expanduser("field_predictions.tsv")
    if os.path.exists(fp_path):
        for line in open(fp_path):
            if "|" in line:
                k, _, v = line.rstrip("\n").partition("|"); FP[k] = v
    for f in sorted(glob.glob(f"{OUT}/*.json")):
        try:
            d = json.load(open(f))
        except ValueError:
            continue
        e = exp_id(f)
        con.executemany("insert into results values (?,?,?,?,?)", [(e, os.path.basename(f), p, n, t) for p, n, t in flatten(d)])
    for p in sorted(glob.glob(f"{PLANS}/*.plan.md")):
        for line in open(p):
            m = LINE.match(line.strip())
            if m:
                v = VERD.search(m.group(2))
                con.execute("insert into result_lines values (?,?,?,?)",
                            (m.group(1), os.path.basename(p), v.group(1) if v else None, line.strip()[:400]))
    exps = {r[0] for r in con.execute("select distinct exp from results union select distinct exp from result_lines")}
    for e in sorted(exps):
        files = ",".join(sorted({r[0] for r in con.execute("select file from results where exp=?", (e,))}))
        rl = con.execute("select plan, verdict from result_lines where exp=? order by rowid desc limit 1", (e,)).fetchone()
        jv = con.execute("select text from results where exp=? and path like '%verdict%PASS%' limit 1", (e,)).fetchone()
        verdict = (rl[1] if rl and rl[1] else None) or (("PASS" if jv[0] == "True" else "FAIL") if jv else None) or CURATED.get(e)
        con.execute("insert into experiments values (?,?,?,?,?)", (e, files, rl[0] if rl else None, verdict, FP.get(e)))
    for e, fp in FP.items():                                   # experiments known only by their field prediction (pre-registered, not yet run)
        con.execute("insert or ignore into experiments(exp, field_prediction) values (?,?)", (e, fp))
    con.commit()
    n = con.execute("select count(*) from experiments").fetchone()[0]
    r = con.execute("select count(*) from results").fetchone()[0]
    print(f"registry: {n} experiments, {r} result rows ->", DB)
    for v, c in con.execute("select coalesce(verdict,'(no verdict recorded)'), count(*) from experiments group by 1 order by 2 desc"):
        print(f"  {v}: {c}")
