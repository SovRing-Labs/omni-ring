#!/usr/bin/env python3
"""OMNITRI wave 2 support tests: TRI-2b, TRIT-CORE-0, DATALOG-FAIR, REC-2 (pre-registered: PREREGISTRATIONS-OMNITRI.md "Wave-2 STAGED").
Usage: tri_wave2.py [tri2b|tritcore|datalog|rec2]"""
import json, os, sqlite3, sys, time
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tri_wave import Space, PH, D, int8, bundle, rule_world
OUT = os.path.expanduser("~/gpurun-out-0930")


def trit(a):
    f = lambda x: (np.sign(x) * (np.abs(x) >= 0.5 * np.sqrt(np.mean(x ** 2)))).astype(np.float32)  # noqa: E731
    return (f(a.real) + 1j * f(a.imag)).astype(np.complex64)


def evaluate(sp, acc, T):
    ok = np.ones(len(T), bool); conf = []
    for miss in range(3):
        kn = [c for c in range(3) if c != miss]
        sc = sp.scores(np.broadcast_to(acc, (len(T), D)), miss, sp.key(miss, T[:, kn[0]], T[:, kn[1]]))
        tru = sc[np.arange(len(T)), T[:, miss]].copy(); sc[np.arange(len(T)), T[:, miss]] = -9
        best = sc.argmax(1); m = tru - sc[np.arange(len(T)), best]
        ok &= m > 0; wrong = T.copy(); wrong[:, miss] = best; conf.append((m, wrong))
    return ok, conf


def refine(sp, acc, T, passes=10, trit_loop=False, curve=None):
    ph = PH[sp.tri(T[:, 0], T[:, 1], T[:, 2])]
    for p in range(passes + 1):
        ok, conf = evaluate(sp, trit(acc) if trit_loop else acc, T)
        if curve is not None:
            curve.append(float(ok.mean()))
        if p == passes:
            break
        for m, wrong in conf:
            w = m < 0.1
            if w.any():
                acc = acc + 0.5 * ph[w].sum(0) - 0.5 * PH[sp.tri(wrong[w, 0], wrong[w, 1], wrong[w, 2])].sum(0)
    return acc


# ---------------------------------------------------------------------------------------------------------------- TRI-2b
SP2 = Space(1000, 50, seed=6)


def _core_bucket(T):
    return refine(SP2, bundle(SP2, T), T) if len(T) else np.zeros(D, np.complex64)


def tri2b():
    sp = SP2; g = np.random.default_rng(7)
    pairs = g.choice(1000 * 50, 26_000, replace=False); s_, r_ = pairs // 50, pairs % 50
    o_ = g.integers(0, 1000, 26_000); contra = slice(20_000, 21_000)
    alt = (o_[contra] + g.integers(1, 1000, 1000)) % 1000
    F = np.concatenate([np.stack([s_[:20_000], r_[:20_000], o_[:20_000]], 1), np.stack([s_[contra], r_[contra], o_[contra]], 1),
                        np.stack([s_[contra], r_[contra], alt], 1)])
    qs = np.concatenate([np.arange(0, 20_000, 4), np.arange(20_000, 21_000), np.arange(21_000, 26_000)])
    kind = np.array(["present"] * 5000 + ["contra"] * 1000 + ["absent"] * 5000)
    half = np.random.default_rng(70).permutation(len(qs)); cal, test = half[:len(qs) // 2], half[len(qs) // 2:]
    res = {}
    for fill in (256, 384):
        B = int(np.ceil(len(F) / fill))
        for arm in ("plain", "core"):
            idx = {}
            for miss, (a, b) in ((2, (0, 1)), (1, (0, 2))):
                h = (F[:, a] * 1_000_003 + F[:, b] * 7919) % B
                groups = [F[h == k] for k in range(B)]
                if arm == "plain":
                    accs = [bundle(sp, T) for T in groups]
                else:
                    with Pool(4) as pool:
                        accs = pool.map(_core_bucket, groups)
                idx[miss] = int8(np.stack(accs))
            S1 = np.zeros(len(qs)); S2 = np.zeros(len(qs)); A = np.zeros(len(qs), int); CL = np.zeros(len(qs))
            for c0 in range(0, len(qs), 1000):
                j = slice(c0, c0 + 1000); ss, rr = s_[qs[j]], r_[qs[j]]
                sc = sp.scores(idx[2][(ss * 1_000_003 + rr * 7919) % B], 2, sp.key(2, ss, rr)); top = np.argsort(-sc, 1)[:, :2]
                A[j] = top[:, 0]; S1[j] = sc[np.arange(len(ss)), top[:, 0]]; S2[j] = sc[np.arange(len(ss)), top[:, 1]]
                back = sp.scores(idx[1][(ss * 1_000_003 + top[:, 0] * 7919) % B], 1, sp.key(1, ss, top[:, 0]))
                CL[j] = np.where(back.argmax(1) == rr, back[np.arange(len(ss)), rr], 0)
            correct = (kind == "present") & (A == o_[qs])
            def trit_of(t, tc):
                return np.where(S2 >= tc, -1, np.where((S1 >= t) & (CL >= t), 1, 0))
            best = None
            for tc in np.arange(0.3, 3.0, 0.02):
                for t in np.arange(0.3, 3.0, 0.02):
                    z = trit_of(t, tc)[cal]; com = z == 1
                    prec = correct[cal][com].mean() if com.any() else 0
                    obj = (prec >= 0.995, com[kind[cal] == "present"].mean() + (z[kind[cal] == "contra"] == -1).mean()
                           - (z[kind[cal] != "contra"] == -1).mean())
                    if best is None or obj > best[0]:
                        best = (obj, t, tc)
            _, t, tc = best; z = trit_of(t, tc)[test]; kt = kind[test]; com = z == 1
            r = {"tau": float(t), "tau_contra": float(tc), "plus1_precision": float(correct[test][com].mean()) if com.any() else None,
                 "plus1_coverage_present": float(com[kt == "present"].mean()), "absent_committed": float(com[kt == "absent"].mean()),
                 "contra_caught": float((z[kt == "contra"] == -1).mean()), "false_contra": float((z[kt != "contra"] == -1).mean())}
            res[f"fill{fill}_{arm}"] = r
            print(f"TRI-2b fill {fill} {arm:5s}: " + " ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in r.items()), flush=True)
    p = res["fill256_plain"]
    res["verdict"] = {"PASS": (p["plus1_precision"] or 0) >= 0.99 and p["absent_committed"] <= 0.005 and p["contra_caught"] >= 0.95}
    print("TRI-2b:", res["verdict"], flush=True)
    return res


# ----------------------------------------------------------------------------------------------------------- TRIT-CORE-0
SP13 = Space(1000, 50, seed=13)


def _tritcore(args):
    N, b = args; g = np.random.default_rng(14 * 1000 + N * 10 + b)                    # the SIFT-CORE-0 buckets
    T = np.stack([g.integers(0, 1000, N), g.integers(0, 50, N), g.integers(0, 1000, N)], 1)
    curve = []; refine(SP13, bundle(SP13, T), T, trit_loop=True, curve=curve)
    return N, curve


def tritcore():
    with Pool(4) as pool:
        out = pool.map(_tritcore, [(N, b) for N in (192, 256, 384) for b in range(8)])
    res = {}
    for N in (192, 256, 384):
        c = np.mean([o[1] for o in out if o[0] == N], 0); res[N] = c.round(4).tolist()
        print(f"TRIT-CORE N={N}: trit pass0 {c[0]:.4f} -> pass10 {c[-1]:.4f} (best {c.max():.4f} at pass {int(c.argmax())})", flush=True)
    res["verdict"] = {"PASS": max(res[256]) >= 0.99 and res[256][-1] >= 0.99}
    print("TRIT-CORE-0:", res["verdict"], flush=True)
    return {str(k): v for k, v in res.items()}


# ---------------------------------------------------------------------------------------------------------- DATALOG-FAIR
def datalog():
    res = {}
    for branch in (False, True):
        facts, _ = rule_world(branch)
        db = sqlite3.connect(":memory:"); db.execute("create table f(p text, a int, b int)")
        db.executemany("insert into f values (?,?,?)", [(p, int(a), int(b)) for p, a, b in facts]); db.execute("create index ix on f(p, a)")
        H = {}
        for p, a, b in facts:
            H.setdefault((p, int(a)), []).append(int(b))
        sp = Space(300, 10, seed=3); F = np.array([(int(a), int(p[1:]), int(b)) for p, a, b in facts])
        B = int(np.ceil(len(F) / 64)); h = (F[:, 0] * 1_000_003 + F[:, 1] * 7919) % B
        accs = int8(np.stack([bundle(sp, F[h == b]) for b in range(B)]))
        def ring(x, hops):
            fr = {x}
            for r in range(hops):
                xs = np.array(sorted(fr)); rr = np.full(len(xs), r)
                sc = sp.scores(accs[(xs * 1_000_003 + rr * 7919) % B], 2, sp.key(2, xs, rr))
                fr = set(np.nonzero(sc >= 0.5)[1].tolist()) if branch else set(sc.argmax(1).tolist())
            return fr
        def sql(x, hops):
            j = " ".join(f"join f t{i} on t{i}.a = t{i-1}.b and t{i}.p = 'r{i}'" for i in range(1, hops))
            return {r[0] for r in db.execute(f"select distinct t{hops-1}.b from f t0 {j} where t0.p = 'r0' and t0.a = ?", (x,))}
        def hashc(x, hops):
            fr = {x}
            for r in range(hops):
                fr = {b for a in fr for b in H.get((f"r{r}", a), ())}
            return fr
        rows = {}
        for hops in (2, 4, 6, 8, 10) if not branch else (2, 3, 4, 5, 6):
            xs = list(range(0, 300, 3)); t = {}; agree = 0
            for nm, fn in (("ring", ring), ("sqlite", sql), ("hash", hashc)):
                t0 = time.perf_counter(); outs = [fn(x, hops) for x in xs]; t[nm] = (time.perf_counter() - t0) / len(xs) * 1000
                if nm == "ring":
                    ring_out = outs
                elif nm == "hash":
                    agree = sum(a == b for a, b in zip(ring_out, outs)) / len(xs)
            rows[hops] = {"ms": t, "ring_equals_exact": agree}
            print(f"DATALOG {'branch' if branch else 'function'} hops {hops:2d}: ring {t['ring']:.3f} ms | sqlite {t['sqlite']:.3f} ms | "
                  f"hash {t['hash']:.4f} ms | ring==exact {agree:.2f}", flush=True)
        res["branch" if branch else "function"] = rows
    return res


# ---------------------------------------------------------------------------------------------------------------- REC-2
def rec2():
    g = np.random.default_rng(30); NP, NG, PER, FILL = 4096, 64, 64, 64
    sp = Space(1000, 50, seed=31)                                   # facts inside polygons
    polys = np.empty((NP, D), np.complex64); facts = []
    for k in range(NP):
        T = np.stack([g.integers(0, 1000, FILL), g.integers(0, 50, FILL), g.integers(0, 1000, FILL)], 1)
        polys[k] = int8(bundle(sp, T)); facts.append(T)
    patom = (np.rint(np.angle(polys) / (np.pi / 8)).astype(np.int64) % 16)
    POS = g.integers(0, 16, (PER, D)); REL = g.integers(0, 16, (2, D)); Pm = [g.permutation(D) for _ in range(3)]
    rp, rr = POS[:, Pm[0]], REL[:, Pm[1]]
    pa = patom[:, Pm[2]]; cpa = (np.cos(np.pi / 8 * pa).astype(np.float32), np.sin(np.pi / 8 * pa).astype(np.float32))
    groups = np.arange(NP).reshape(NG, PER)                         # group g holds polygons groups[g]
    lvl1 = np.stack([int8(PH[(rp + rr[0] + pa[groups[gg]]) % 16].sum(0)) for gg in range(NG)])
    gatom = (np.rint(np.angle(lvl1) / (np.pi / 8)).astype(np.int64) % 16)[:, Pm[2]]
    cga = (np.cos(np.pi / 8 * gatom).astype(np.float32), np.sin(np.pi / 8 * gatom).astype(np.float32))
    top = int8(PH[(rp[:NG] + rr[1] + gatom) % 16].sum(0))
    Q = 2000; qg, qj, qf = g.integers(0, NG, Q), g.integers(0, PER, Q), g.integers(0, FILL, Q)
    t0 = time.perf_counter()
    u = top[None] * np.conj(PH[(rp[qg] + rr[1]) % 16]); g1 = (u.real @ cga[0].T + u.imag @ cga[1].T).argmax(1)
    u = lvl1[g1] * np.conj(PH[(rp[qj] + rr[0]) % 16]); k2 = (u.real @ cpa[0].T + u.imag @ cpa[1].T).argmax(1)
    T3 = np.stack([facts[k][f] for k, f in zip(groups[qg, qj], qf)])
    sc = sp.scores(polys[k2], 2, sp.key(2, T3[:, 0], T3[:, 1])); o3 = sc.argmax(1)
    ms = (time.perf_counter() - t0) / Q * 1000
    l1 = float((g1 == qg).mean()); l2 = float((k2 == groups[qg, qj]).mean()); e2e = float(((g1 == qg) & (k2 == groups[qg, qj]) & (o3 == T3[:, 2])).mean())
    res = {"facts": NP * FILL, "level1_group": l1, "level2_polygon": l2, "end_to_end": e2e, "lookups": 3, "ms_per_query_batched": ms,
           "verdict": {"PASS": e2e >= 0.99}}
    print("REC-2:", res, flush=True)
    return res


if __name__ == "__main__":
    part = sys.argv[1]; path = f"{OUT}/tri_wave2.result.json"
    allres = json.load(open(path)) if os.path.exists(path) else {}
    allres[{"tri2b": "TRI-2b", "tritcore": "TRIT-CORE-0", "datalog": "DATALOG-FAIR", "rec2": "REC-2"}[part]] = \
        {"tri2b": tri2b, "tritcore": tritcore, "datalog": datalog, "rec2": rec2}[part]()
    json.dump(allres, open(path, "w"), indent=1, default=str)
