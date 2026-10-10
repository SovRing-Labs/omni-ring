#!/usr/bin/env python3
"""CORE-ROUTE-1 (PIFS-RES plan (unpublished)): the core learns its mode sequence by Thompson sampling over marks."""
import itertools, json, os, sys, time
import numpy as np
sys.path.insert(0, ".")
from omni_ring.route import sieve, part, peel_kernel as peel_cued, check, encode_prod, codeword  # noqa: E402
from omni_ring.crt import solve_crt  # noqa: E402

OUT = os.path.expanduser("~/gpurun-out-0930/core_route1.result.json")
MS = (113, 127, 131); M = int(np.prod(MS)); D = 1024; T = int(os.environ.get("T", 400))
MODES = ["SIEVE", "PART", "PEEL"]; SEQS = list(itertools.permutations(MODES))


def signals(v):
    mag = np.abs(v); prof = float(mag.std() / max(mag.mean(), 1e-9))      # a single unit mark: 0; a sum of K: spread grows
    return 0 if prof < 0.05 else (1 if prof < 0.5 else 2)


def run_mode(mode, v, cbs, truth_set):
    t = time.time()
    if mode == "SIEVE":
        res = sieve(v, cbs, MS); ok = check(v, res, cbs) >= 0.9; ans = {solve_crt(res, MS)} if ok else None
    elif mode == "PART":
        res, agree = part(v, cbs, MS); ok = check(v, res, cbs) >= 0.5 and agree >= 0.6; ans = {solve_crt(res, MS)} if ok else None
    else:
        f = peel_cued(v, cbs, MS); ans = {solve_crt(r, MS) for r in f} if f else None
    cost = time.time() - t
    correct = ans is not None and len(ans & truth_set) >= max(1, int(0.75 * len(truth_set))) and ans <= truth_set | set()
    return ans is not None, correct, cost


def main():
    rng = np.random.default_rng(1)
    rots = [rng.integers(0, m, D) * (2 * np.pi / m) for m in MS]
    cbs = [np.exp(1j * np.outer(np.arange(m), r)) for m, r in zip(MS, rots)]
    alpha = np.ones((3, len(SEQS))); beta = np.ones((3, len(SEQS)))       # Beta posteriors per signal bucket per sequence
    log = []
    for t in range(T):
        cls = rng.choice(["CLEAN", "NOISY", "QUANT", "SUPER"])
        if cls == "SUPER":
            xs = {int(x) for x in rng.choice(M, 4, replace=False)}; v = sum(encode_prod(x, rots) for x in xs)
        else:
            x = int(rng.integers(0, M)); ang = sum(x * r for r in rots)
            if cls == "NOISY": ang = ang + rng.vonmises(0, 16.0, D)
            if cls == "QUANT": ang = np.round(ang / (2 * np.pi / 16)) * (2 * np.pi / 16)
            v = np.exp(1j * ang); xs = {x}
        s = signals(v)
        k = int(np.argmax(rng.beta(alpha[s], beta[s])))                     # Thompson: sample a sequence
        seq = SEQS[k]; total = 0.0; answered = False; correct = False; used = []
        for mode in seq:
            ok, corr, cost = run_mode(mode, v, cbs, xs); total += cost; used.append(mode)
            if ok:
                answered, correct = True, corr; break
        reward = (1.0 if correct else 0.0) * max(0.0, 1.0 - total / 1.5)     # correct AND cheap: fractional Beta update
        alpha[s, k] += reward; beta[s, k] += 1.0 - reward
        # oracle cost: cheapest single mode that is correct on this slot
        oc = min([c for m_ in MODES for (ok, corr, c) in [run_mode(m_, v, cbs, xs)] if ok and corr] or [total])
        log.append({"t": t, "cls": cls, "sig": s, "seq": used, "correct": bool(correct), "cost": total, "oracle": oc})
        if t % 50 == 49:
            last = log[-50:]
            print(f"t {t + 1}: acc {np.mean([l['correct'] for l in last]):.2f} cost {np.mean([l['cost'] for l in last]):.3f}s oracle {np.mean([l['oracle'] for l in last]):.3f}s", flush=True)
    last = log[-100:]
    by = {}
    for c in ("CLEAN", "NOISY", "QUANT", "SUPER"):
        L = [l for l in last if l["cls"] == c]
        by[c] = {"acc": float(np.mean([l["correct"] for l in L])) if L else None, "n": len(L)}
    ratio = float(np.mean([l["cost"] for l in last]) / max(np.mean([l["oracle"] for l in last]), 1e-9))
    learned = {str(s): list(SEQS[int(np.argmax(alpha[s] / (alpha[s] + beta[s])))]) for s in range(3)}
    res = {"T": T, "cost_over_oracle_last100": ratio, "acc_by_class_last100": by, "learned_sequence_by_signal": learned,
           "verdict": {"PASS": ratio <= 1.5 and all((v_["acc"] or 0) >= 0.95 for v_ in by.values() if v_["n"])}}
    json.dump({**res, "log": log}, open(OUT, "w"), indent=1); print("CORE-ROUTE-1", {k: v_ for k, v_ in res.items()})


if __name__ == "__main__":
    main()
