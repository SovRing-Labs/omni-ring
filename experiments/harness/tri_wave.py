#!/usr/bin/env python3
"""OMNITRI wave 1 — TRI-0/1/2/3 + HARMONY-0 (pre-registered: experiments/PREREGISTRATIONS-OMNITRI.md, "Wave-1 operationalization").
Usage: tri_wave.py [tri0|tri1|tri2|tri3|harmony0|all]"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, ".")
from omni_ring.logic import Tracker, Prover

D = 5120
PH = np.exp(2j * np.pi * np.arange(16) / 16).astype(np.complex64)
OUT = os.path.expanduser("~/gpurun-out-0930")


class Space:
    """Codebooks + role permutations. Entities fill S and O corners, relations fill R."""
    def __init__(self, n_ent, n_rel, seed=0):
        g = np.random.default_rng(seed)
        self.E = g.integers(0, 16, (n_ent, D), dtype=np.int64)
        self.R = g.integers(0, 16, (n_rel, D), dtype=np.int64)
        self.P = [g.permutation(D) for _ in range(3)]                      # S, R, O role permutations
        self.role = [self.E[:, self.P[0]], self.R[:, self.P[1]], self.E[:, self.P[2]]]
        self.clean = [(np.cos(2 * np.pi * c / 16).astype(np.float32), np.sin(2 * np.pi * c / 16).astype(np.float32))
                      for c in self.role]                                  # clean-up tables per corner

    def tri(self, s, r, o):
        return (self.role[0][s] + self.role[1][r] + self.role[2][o]) % 16

    def key(self, missing, a, b):
        """Phase sum of the two known corners (in S,R,O order with `missing` left out)."""
        known = [c for c in range(3) if c != missing]
        return (self.role[known[0]][a] + self.role[known[1]][b]) % 16

    def scores(self, acc, missing, keys):
        """acc: (q, D) complex bucket per query; keys: (q, D) phases. Returns (q, n_atoms) normalized scores (~1 = one item)."""
        u = acc * np.conj(PH[keys])
        cr, ci = self.clean[missing]
        return (u.real.astype(np.float32) @ cr.T + u.imag.astype(np.float32) @ ci.T) / D


def int8(acc):
    """SHARDS slot format: int8 re/im + per-bucket scale, dequantized back."""
    a = np.atleast_2d(acc)
    s = np.maximum(np.abs(a.real).max(1), np.abs(a.imag).max(1))[:, None] / 127
    s[s == 0] = 1
    q = np.rint(a.real / s).clip(-127, 127) + 1j * np.rint(a.imag / s).clip(-127, 127)
    return (q * s).astype(np.complex64).reshape(acc.shape)


def bundle(sp, trips):
    return PH[sp.tri(trips[:, 0], trips[:, 1], trips[:, 2])].sum(0)


# ---------------------------------------------------------------------------------------------------------------- TRI-0
def tri0():
    sp = Space(1000, 50); g = np.random.default_rng(1); res = {"curve": {}}
    for N in (16, 32, 64, 128, 192, 256, 384, 512):
        row = {}
        for form in ("float", "int8"):
            ok = [0, 0, 0]; tot = 0
            for b in range(12):
                T = np.stack([g.integers(0, 1000, N), g.integers(0, 50, N), g.integers(0, 1000, N)], 1)
                acc = bundle(sp, T); acc = int8(acc) if form == "int8" else acc
                Q = T[:min(N, 64)]
                for miss in range(3):
                    kn = [c for c in range(3) if c != miss]
                    sc = sp.scores(np.broadcast_to(acc, (len(Q), D)), miss, sp.key(miss, Q[:, kn[0]], Q[:, kn[1]]))
                    ok[miss] += int((sc.argmax(1) == Q[:, miss]).sum())
                tot += len(Q)
            row[form] = {n: ok[i] / tot for i, n in enumerate("SRO")}
        res["curve"][N] = row
        print(f"TRI-0 N={N:3d}  float S/R/O {row['float']['S']:.4f} {row['float']['R']:.4f} {row['float']['O']:.4f} | "
              f"int8 {row['int8']['S']:.4f} {row['int8']['R']:.4f} {row['int8']['O']:.4f}", flush=True)
    cap = max([N for N, r in res["curve"].items() if min(r["int8"].values()) >= 0.99] or [0])
    res["cap_int8"] = cap
    # full scale: 100k facts, three edge indexes, buckets sized to the cap
    F = np.unique(np.stack([g.integers(0, 1000, 100_000), g.integers(0, 50, 100_000), g.integers(0, 1000, 100_000)], 1), axis=0)
    B = int(np.ceil(len(F) / cap)) if cap else 1
    full = {"facts": len(F), "buckets_per_index": B}
    for miss in range(3):
        kn = [c for c in range(3) if c != miss]
        h = (F[:, kn[0]] * 1_000_003 + F[:, kn[1]] * 7919) % B
        accs = np.zeros((B, D), np.complex64)
        for b in range(B):                                                  # per bucket (memory-bounded)
            accs[b] = bundle(sp, F[h == b])
        accs = int8(accs)
        answers = {}
        for i, f in enumerate(F):
            answers.setdefault((f[kn[0]], f[kn[1]]), set()).add(int(f[miss]))
        pick = g.choice(len(F), 3000, replace=False); good = 0
        for c0 in range(0, 3000, 500):
            idx = pick[c0:c0 + 500]
            sc = sp.scores(accs[h[idx]], miss, sp.key(miss, F[idx, kn[0]], F[idx, kn[1]]))
            for j, i in enumerate(idx):
                A = answers[(F[i, kn[0]], F[i, kn[1]])]
                top = set(np.argsort(-sc[j])[:len(A)].tolist())
                good += int(top == A)
        fill = np.bincount(h, minlength=B)
        full["SRO"[miss]] = {"strict_recovery": good / 3000, "fill_mean": float(fill.mean()), "fill_max": int(fill.max())}
        print(f"TRI-0 full 100k, missing {'SRO'[miss]}: strict {good/3000:.4f} (fill mean {fill.mean():.0f}, max {fill.max()})", flush=True)
    res["full"] = full
    ok_cap = cap >= 16
    res["verdict"] = {"PASS": ok_cap and all(full[c]["strict_recovery"] >= 0.99 for c in "SRO"), "KILL": cap < 16,
                      "query_cost_sifts": 1}
    print("TRI-0:", res["verdict"], "cap_int8", cap, flush=True)
    return res


# ---------------------------------------------------------------------------------------------------------------- TRI-1
def rule_world(branch, seed=2, n=300, nr=10, H=9):
    g = np.random.default_rng(seed)
    facts = set()
    for r in range(nr):
        for x in range(n):
            k = g.integers(1, 3) if branch else 1
            for y in g.choice(n, k, replace=False):
                facts.add((f"r{r}", str(x), str(int(y))))
    rules = [(("c1", "X", "Z"), [("r0", "X", "Y"), ("r1", "Y", "Z")])]
    for h in range(2, H + 1):
        rules.append(((f"c{h}", "X", "Z"), [(f"c{h-1}", "X", "Y"), (f"r{h}", "Y", "Z")]))
    return facts, rules


def tri1():
    res = {}
    for branch in (False, True):
        facts, rules = rule_world(branch)
        T = Tracker(rules)
        for f in facts:                                         # bulk load: assert_fact re-chains from ALL facts each call (O(n^2))
            T.base.add(f); T._add(f)
        t0 = time.perf_counter(); T.chain(); t_chain = time.perf_counter() - t0
        sp = Space(300, 10, seed=3)
        F = np.array([(int(a), int(p[1:]), int(b)) for p, a, b in facts])
        B = int(np.ceil(len(F) / 64)); h = (F[:, 0] * 1_000_003 + F[:, 1] * 7919) % B
        accs = int8(np.stack([bundle(sp, F[h == b]) for b in range(B)]))

        def hop(xs, r):
            xs = np.array(sorted(xs)); rr = np.full(len(xs), r)
            sc = sp.scores(accs[(xs * 1_000_003 + rr * 7919) % B], 2, sp.key(2, xs, rr))
            return set(np.nonzero(sc >= 0.5)[1].tolist()) if branch else set(sc.argmax(1).tolist())

        per_h = {}; t_ring = []
        Hmax = 5 if branch else 9
        g = np.random.default_rng(4)
        for hh in range(1, Hmax + 1):
            ok = 0
            for x in g.choice(300, 100, replace=False):
                truth = {int(f[2]) for f in T.facts if f[0] == f"c{hh}" and f[1] == str(x)}
                t0 = time.perf_counter(); fr = {int(x)}
                for r in range(hh + 1):
                    fr = hop(fr, r)
                t_ring.append(time.perf_counter() - t0)
                ok += int(fr == truth)
            per_h[hh + 1] = ok / 100
            print(f"TRI-1 {'branch' if branch else 'function'} hops={hh+1:2d}: {ok/100:.2f}", flush=True)
        # Prover single goal
        P = Prover(T); x = "7"; goal_z = next(f[2] for f in T.facts if f[0] == "c3" and f[1] == x)
        t0 = time.perf_counter(); P.prove(("c3", x, goal_z)); t_prove = time.perf_counter() - t0
        res["branch" if branch else "function"] = {"by_hops": per_h, "tracker_chain_s": t_chain,
                                                   "ring_query_ms_mean": float(np.mean(t_ring) * 1000), "prover_c3_ms": t_prove * 1000}
        print(f"  Tracker.chain {t_chain*1000:.0f} ms | ring query {np.mean(t_ring)*1000:.2f} ms | Prover c3 {t_prove*1000:.1f} ms", flush=True)
    # phase-addition arm: translational tree
    g = np.random.default_rng(5); n, nr = 500, 10
    V = g.integers(0, 16, (nr, D)); X = np.zeros((n, D), np.int64); X[0] = g.integers(0, 16, D)
    kids = {0: {}}; parent = {}
    for i in range(1, n):
        while True:
            p = int(g.integers(0, i)); free = [r for r in range(nr) if r not in kids[p]]
            if free:
                break
        r = int(g.choice(free)); kids[p][r] = i; kids[i] = {}; parent[i] = (p, r); X[i] = (X[p] + V[r]) % 16
    cr, ci = np.cos(2 * np.pi * X / 16).astype(np.float32), np.sin(2 * np.pi * X / 16).astype(np.float32)
    add = {}
    for hh in range(1, 11):
        ok = tot = coll = 0
        for z in range(n):
            path = []; y = z
            while y in parent and len(path) < hh:
                p, r = parent[y]; path.append(r); y = p
            if len(path) < hh:
                continue
            q = (X[y] + V[path].sum(0)) % 16
            s = (np.cos(2 * np.pi * q / 16) @ cr.T + np.sin(2 * np.pi * q / 16) @ ci.T) / D
            ok += int(s.argmax() == z); tot += 1; coll += int((s >= 0.999).sum() > 1)
        add[hh] = {"n": tot, "exact": ok / tot if tot else None, "commutative_collisions": coll}
    res["phase_addition"] = add
    print("TRI-1 phase addition:", {k: (v["exact"], v["commutative_collisions"]) for k, v in add.items()}, flush=True)
    f5 = [res["function"]["by_hops"][k] for k in range(2, 6)]
    res["verdict"] = {"PASS_function_to5": min(f5) >= 0.99, "faster_than_chain": res["function"]["ring_query_ms_mean"] / 1000 < res["function"]["tracker_chain_s"]}
    print("TRI-1:", res["verdict"], flush=True)
    return res


# ---------------------------------------------------------------------------------------------------------------- TRI-2
def tri2():
    sp = Space(1000, 50, seed=6); g = np.random.default_rng(7)
    pairs = g.choice(1000 * 50, 26_000, replace=False); s_, r_ = pairs // 50, pairs % 50
    o_ = g.integers(0, 1000, 26_000)
    pres = slice(0, 20_000); contra = slice(20_000, 21_000); absent = slice(21_000, 26_000)
    alt = (o_[contra] + g.integers(1, 1000, 1000)) % 1000
    F = np.concatenate([np.stack([s_[pres], r_[pres], o_[pres]], 1), np.stack([s_[contra], r_[contra], o_[contra]], 1),
                        np.stack([s_[contra], r_[contra], alt], 1)])
    B = int(np.ceil(len(F) / 64))
    idx = {}
    for miss, (a, b) in ((2, (0, 1)), (1, (0, 2))):                                # (s,r)->o and (s,o)->r indexes
        h = (F[:, a] * 1_000_003 + F[:, b] * 7919) % B
        idx[miss] = int8(np.stack([bundle(sp, F[h == k]) for k in range(B)]))
    def q_o(s, r):
        return sp.scores(idx[2][(s * 1_000_003 + r * 7919) % B], 2, sp.key(2, s, r))
    def q_r(s, o):
        return sp.scores(idx[1][(s * 1_000_003 + o * 7919) % B], 1, sp.key(1, s, o))
    qs = np.concatenate([np.arange(0, 20_000, 4), np.arange(20_000, 21_000), np.arange(21_000, 26_000)])
    kind = np.array(["present"] * 5000 + ["contra"] * 1000 + ["absent"] * 5000)
    S1 = np.zeros(len(qs)); S2 = np.zeros(len(qs)); A = np.zeros(len(qs), int); CL = np.zeros(len(qs))
    for c0 in range(0, len(qs), 1000):
        j = slice(c0, c0 + 1000); ss, rr = s_[qs[j]], r_[qs[j]]
        sc = q_o(ss, rr); top = np.argsort(-sc, 1)[:, :2]
        A[j] = top[:, 0]; S1[j] = sc[np.arange(len(ss)), top[:, 0]]; S2[j] = sc[np.arange(len(ss)), top[:, 1]]
        back = q_r(ss, top[:, 0]); CL[j] = np.where(back.argmax(1) == rr, back[np.arange(len(ss)), rr], 0)
    correct = (kind == "present") & (A == o_[qs])
    half = g.permutation(len(qs)); cal, test = half[:len(qs) // 2], half[len(qs) // 2:]
    def trit(t, tc):
        return np.where(S2 >= tc, -1, np.where((S1 >= t) & (CL >= t), 1, 0))
    best = None
    for tc in np.arange(0.3, 0.9, 0.02):
        for t in np.arange(0.3, 0.95, 0.02):
            z = trit(t, tc)[cal]; com = z == 1
            prec = correct[cal][com].mean() if com.any() else 0
            cov = com[kind[cal] == "present"].mean()
            rec_c = (z[kind[cal] == "contra"] == -1).mean()
            fa_c = (z[kind[cal] != "contra"] == -1).mean()
            obj = (prec >= 0.995, cov + rec_c - fa_c)
            if best is None or obj > best[0]:
                best = (obj, t, tc)
    _, t, tc = best; z = trit(t, tc)[test]; kt = kind[test]; com = z == 1
    res = {"tau": float(t), "tau_contra": float(tc),
           "plus1_precision": float(correct[test][com].mean()), "plus1_coverage_present": float(com[kt == "present"].mean()),
           "absent_committed": float(com[kt == "absent"].mean()), "absent_escalated": float((z[kt == "absent"] == 0).mean()),
           "contra_caught": float((z[kt == "contra"] == -1).mean()), "false_contra": float((z[kt != "contra"] == -1).mean())}
    res["verdict"] = {"PASS": res["plus1_precision"] >= 0.99 and res["contra_caught"] >= 0.95, "KILL": res["plus1_precision"] < 0.97}
    print("TRI-2:", {k: round(v, 4) if isinstance(v, float) else v for k, v in res.items()}, flush=True)
    return res


# ---------------------------------------------------------------------------------------------------------------- TRI-3
def tri3(cap):
    sp = Space(1000, 50, seed=8); g = np.random.default_rng(9); same = tot = 0
    for b in range(20):
        T = np.stack([g.integers(0, 1000, cap), g.integers(0, 50, cap), g.integers(0, 1000, cap)], 1)
        acc = int8(bundle(sp, T)); Q = T[:64]
        u = acc * np.conj(PH[sp.key(2, Q[:, 0], Q[:, 1])])
        cr, ci = sp.clean[2]
        full = (u.real @ cr.T + u.imag @ ci.T).argmax(1)
        pre = u[:, :320].real @ cr[:, :320].T + u[:, :320].imag @ ci[:, :320].T
        cand = np.argsort(-pre, 1)[:, :8]
        fin = np.array([cand[i][np.argmax(u[i].real @ cr[cand[i]].T + u[i].imag @ ci[cand[i]].T)] for i in range(len(Q))])
        same += int((fin == full).sum()); tot += len(Q)
    res = {"prefix_same": same / tot}
    T = np.stack([g.integers(0, 1000, 2000), g.integers(0, 50, 2000), g.integers(0, 1000, 2000)], 1)
    ph = PH[sp.tri(T[:, 0], T[:, 1], T[:, 2])]
    lam = 1 - 1 / (cap / 2)
    for name, l in (("no_decay", 1.0), ("decay", lam)):
        acc = np.zeros(D, np.complex64)
        for p in ph:
            acc = acc * l + p
        acc = int8(acc); Q = T[-64:]
        sc = sp.scores(np.broadcast_to(acc, (64, D)), 2, sp.key(2, Q[:, 0], Q[:, 1]))
        res[name] = float((sc.argmax(1) == Q[:, 2]).mean())
    res["lambda"] = lam
    res["verdict"] = {"PASS": res["prefix_same"] >= 0.995 and res["decay"] >= 0.99}
    print("TRI-3:", res, flush=True)
    return res


# ------------------------------------------------------------------------------------------------------------ HARMONY-0
def harmony0():
    sp = Space(1000, 50, seed=10); g = np.random.default_rng(11); H = []; Y = []
    for i in range(2000):
        contra = i % 2 == 1; m = int(g.integers(1, 5)) if contra else 0
        bg = np.stack([g.integers(0, 1000, 64), g.integers(0, 50, 64), g.integers(0, 1000, 64)], 1)
        s, r, o = int(g.integers(0, 1000)), int(g.integers(0, 50)), int(g.integers(0, 1000)); o2 = (o + 1 + int(g.integers(0, 998))) % 1000
        obs = np.array([[s, r, o2 if j < m else o] for j in range(8)])
        acc = int8(bundle(sp, np.concatenate([bg, obs])))
        sc = sp.scores(acc[None], 2, sp.key(2, np.array([s]), np.array([r])))[0]
        a, b = np.sort(sc)[-2:][::-1]
        H.append(1 - b / a); Y.append(contra)
    H, Y = np.array(H), np.array(Y)
    pos, neg = H[~Y], H[Y]                                     # consistent should have HIGHER harmony
    auc = float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())
    by_m = {}
    res = {"auc": auc, "harmony_consistent_mean": float(pos.mean()), "harmony_contra_mean": float(neg.mean()),
           "verdict": {"PASS": auc >= 0.95}}
    print("HARMONY-0:", res, flush=True)
    return res


if __name__ == "__main__":
    part = sys.argv[1] if len(sys.argv) > 1 else "all"
    path = f"{OUT}/tri_wave.result.json"
    allres = json.load(open(path)) if os.path.exists(path) else {}
    if part in ("tri0", "all"):
        allres["TRI-0"] = tri0()
    if part in ("tri1", "all"):
        allres["TRI-1"] = tri1()
    if part in ("tri2", "all"):
        allres["TRI-2"] = tri2()
    if part in ("tri3", "all"):
        allres["TRI-3"] = tri3(allres.get("TRI-0", {}).get("cap_int8") or 128)
    if part in ("harmony0", "all"):
        allres["HARMONY-0"] = harmony0()
    json.dump(allres, open(path, "w"), indent=1, default=str)
