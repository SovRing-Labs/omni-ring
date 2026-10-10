#!/usr/bin/env python3
"""CORE-ROUTE-2: LearningCore (programs as VM vectors, Thompson over a program library) on the mixed stream, M = 1.88M, D 1024.
Bars (as CORE-ROUTE-1): last-100 cost/oracle <= 1.5 and accuracy >= 0.95 per class; reported: learned program per signal bucket."""
import json, os, sys, numpy as np
sys.path.insert(0, ".")
from omni_ring.core_learn import LearningCore  # noqa: E402
from omni_ring.route import encode_prod, sieve, part, peel_kernel, check  # noqa: E402
from omni_ring.crt import solve_crt  # noqa: E402
OUT = os.path.expanduser("~/gpurun-out-0930/core_route2.result.json")
MS = (113, 127, 131); M = int(np.prod(MS)); D = 1024; T = int(os.environ.get("T", 200))


def oracle_cost(v, cbs, truth):
    import time
    best = None
    for mode in ("SIEVE", "PART", "PEEL"):
        t = time.time()
        if mode == "SIEVE":
            r = sieve(v, cbs, MS); ans = {solve_crt(r, MS)} if check(v, r, cbs) >= 0.9 else None
        elif mode == "PART":
            r, ag = part(v, cbs, MS); ans = {solve_crt(r, MS)} if check(v, r, cbs) >= 0.5 and ag >= 0.6 else None
        else:
            f = peel_kernel(v, cbs, MS); ans = {solve_crt(x, MS) for x in f} if f else None
        c = time.time() - t
        if ans is not None and len(ans & truth) >= max(1, int(0.75 * len(truth))) and ans <= truth and (best is None or c < best):
            best = c
    return best


def main():
    rng = np.random.default_rng(3)
    rots = [rng.integers(0, m, D) * (2 * np.pi / m) for m in MS]
    cbs = [np.exp(1j * np.outer(np.arange(m), r)) for m, r in zip(MS, rots)]
    core = LearningCore(cbs, MS, seed=3); log = []
    for t in range(T):
        cls = rng.choice(["CLEAN", "NOISY", "QUANT", "SUPER"])
        if cls == "SUPER":
            xs = {int(x) for x in rng.choice(M, 4, replace=False)}; v = sum(encode_prod(x, rots) for x in xs)
        else:
            x = int(rng.integers(0, M)); ang = sum(x * r for r in rots)
            if cls == "NOISY": ang = ang + rng.vonmises(0, 16.0, D)
            if cls == "QUANT": ang = np.round(ang / (2 * np.pi / 16)) * (2 * np.pi / 16)
            v = np.exp(1j * ang); xs = {x}
        ans, m = core.tick(v, truth=xs)
        oc = oracle_cost(v, cbs, xs)
        log.append({"t": t, "cls": cls, "sig": list(m.signals), "program": m.program, "seq": m.seq, "fired": m.fired, "correct": bool(m.correct), "cost": m.cost_s, "oracle": oc if oc is not None else m.cost_s})
        if t % 50 == 49:
            last = log[-50:]; print(f"t {t + 1}: acc {np.mean([l['correct'] for l in last]):.2f} cost {np.mean([l['cost'] for l in last]):.3f}s oracle {np.mean([l['oracle'] for l in last]):.3f}s fired/slot {np.mean([len(l['fired']) for l in last]):.2f}", flush=True)
    last = log[-100:]
    by = {c: float(np.mean([l["correct"] for l in last if l["cls"] == c])) for c in ("CLEAN", "NOISY", "QUANT", "SUPER")}
    ratio = float(np.mean([l["cost"] for l in last]) / max(np.mean([l["oracle"] for l in last]), 1e-9))
    res = {"T": T, "cost_over_oracle_last100": ratio, "acc_by_class_last100": by, "learned": core.learned(),
           "programs_in_library": len(core.library), "verdict": {"PASS": bool(ratio <= 1.5 and all(a >= 0.95 for a in by.values()))}}
    json.dump({**res, "log": log}, open(OUT, "w"), indent=1); print("CORE-ROUTE-2", res)


if __name__ == "__main__":
    main()
