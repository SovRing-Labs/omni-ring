#!/usr/bin/env python3
"""SUPER-Q-0 (PREREGISTRATIONS-OMNITRI.md wave-2): 2-3 queries superposed into one coarse sift, exact rescoring on top-M candidates."""
import ctypes, json, os, sys, time
import numpy as np
sys.path.insert(0, "."); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from omni_ring.heads import RingHead
import omni_ring.seed as S
from pyramid0 import top_incl, vote
H = "<your-data>/data/ring-heads"
PH = np.exp(2j * np.pi * np.arange(16) / 16).astype(np.complex64)


def superpose(q):
    return (np.rint(np.angle(PH[q].sum(0)) / (np.pi / 8)).astype(np.int64) % 16).astype(np.uint8)


def superq(h, q, packed, g, M, coarse_fn):
    groups = [np.arange(i, min(i + g, len(q))) for i in range(0, len(q), g)]
    Z = np.stack([superpose(q[gr]) for gr in groups]); C = coarse_fn(Z)
    out = [None] * len(q); k = min(h.k, len(h.marks))
    for gi, gr in enumerate(groups):
        cand = top_incl(C[gi], M)
        for i in gr:
            s = S.sift_rows_p4(q[i], packed, cand); out[i] = (cand, s)
    return out, k


if __name__ == "__main__":
    res = {}
    for name in sorted(os.listdir(H)):
        h = RingHead.load(f"{H}/{name}"); truth = np.array(json.load(open(f"{H}/{name}/eval_ref.json"))["truth"])
        q = S.encode(np.load(f"{H}/{name}/eval_X.npy").astype(np.float32), h.proj, h.centre)
        packed = S.pack4(h.marks); full = np.array(h.read_marks(q)[0]); res[name] = {"n_marks": len(h.marks), "n_q": len(q)}
        if name == sorted(os.listdir(H))[0]:                                 # harness validity: M = all marks == exact
            o, k = superq(h, q, packed, 3, len(h.marks), lambda Z: S.sift_batch_p4(Z, packed))
            lab = np.array([vote(h, c, s, k) for c, s in o]); res["validity_mismatch_all_marks"] = int((lab != full).sum())
            print(f"validity (M = all marks): mismatches {res['validity_mismatch_all_marks']}", flush=True)
        line = f"{name:15s} n={len(h.marks):6d}"
        for g in (2, 3):
            for M in (64, 128, 256):
                o, k = superq(h, q, packed, g, M, lambda Z: S.sift_batch(Z, h.marks))
                lab = np.array([vote(h, c, s, k) for c, s in o])
                mm = int((lab != full).sum()); dacc = float(np.mean(lab == truth) - np.mean(full == truth))
                res[name][f"g{g}_M{M}"] = {"mismatch_pct": mm / len(q) * 100, "dacc": dacc}
                line += f" | g{g} M{M}: {mm/len(q):.1%} {dacc:+.3f}"
        print(line, flush=True)
    ctypes.CDLL("libgomp.so.1").omp_set_num_threads(1)
    h = RingHead.load(f"{H}/d5.verb_log"); q = S.encode(np.load(f"{H}/d5.verb_log/eval_X.npy").astype(np.float32)[:255], h.proj, h.centre)
    packed = S.pack4(h.marks); k = min(h.k, len(h.marks))
    def best(f):
        f(); b = 1e9
        for _ in range(3):
            t = time.perf_counter(); f(); b = min(b, time.perf_counter() - t)
        return b
    def exact():
        sc = S.sift_batch(q, h.marks); return np.argpartition(-sc, k, 1)[:, :k]
    def exact_p4():
        sc = S.sift_batch_p4(q, packed); return np.argpartition(-sc, k, 1)[:, :k]
    sp = {"exact_tiled_ms_q": best(exact) * 1000 / len(q), "exact_p4_ms_q": best(exact_p4) * 1000 / len(q)}
    base = min(sp.values())
    for g in (2, 3):
        for M in (64, 128, 256):
            def run():
                o, _ = superq(h, q, packed, g, M, lambda Z: S.sift_batch(Z, h.marks))
                return [c[np.argpartition(-s, min(k, len(s) - 1))[:k]] for c, s in o]
            t = best(run) * 1000 / len(q); sp[f"g{g}_M{M}_ms_q"] = t; sp[f"g{g}_M{M}_x"] = base / t
    res["speed_d5.verb_log_1t"] = sp; print("speed:", {a: round(b, 3) for a, b in sp.items()}, flush=True)
    heads = [n for n in res if isinstance(res[n], dict) and "n_marks" in res[n]]
    ok = {f"g{g}_M{M}": all(res[n][f"g{g}_M{M}"]["mismatch_pct"] <= 0.5 and res[n][f"g{g}_M{M}"]["dacc"] >= -0.005 for n in heads)
          for g in (2, 3) for M in (64, 128, 256)}
    win = [c for c, v in ok.items() if v and sp[f"{c}_x"] >= 1.5]
    res["verdict"] = {"configs_meeting_mismatch_bar": [c for c, v in ok.items() if v], "PASS_configs": win, "PASS": bool(win),
                      "KILL": not any(v and sp[f"{c}_x"] > 1.0 for c, v in ok.items())}
    print("SUPER-Q-0:", res["verdict"], flush=True)
    json.dump(res, open(os.path.expanduser("~/gpurun-out-0930/superq0.result.json"), "w"), indent=1, default=str)
