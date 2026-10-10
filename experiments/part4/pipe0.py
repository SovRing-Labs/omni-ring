#!/usr/bin/env python3
"""PIPE-0 (INTEGRATE-3D plan (unpublished)): the spinning prism — stations on a ring, each a process, every tick each works on the slot
under it; shapes = station layouts TRI / SQUARE / HEX vs SEQ (the sequential ladder)."""
import json, os, sys, time, multiprocessing as mp
import numpy as np
sys.path.insert(0, ".")
from omni_ring.route import sieve, part, peel_kernel, check, encode_prod  # noqa: E402
from omni_ring.crt import solve_crt  # noqa: E402
OUT = os.path.expanduser(os.environ.get("PIPE_OUT", "~/gpurun-out-0930/pipe0.result.json"))
MS = (113, 127, 131); M = int(np.prod(MS)); D = 1024; NSLOT = int(os.environ.get("NSLOT", 120))
LAYOUTS = {"TRI": ["SIEVE+CHECK", "PART+CHECK", "PEEL+CHECK"], "SQUARE": ["SIEVE", "PART", "PEEL", "CHECK"],
           "HEX": ["SIEVE", "CHECK", "PART", "CHECK", "PEEL", "CHECK"],
           "TRI-PEELx2": ["SIEVE+CHECK", "PART+CHECK", "PEEL+CHECK*2"], "TRI-PEELx3": ["SIEVE+CHECK", "PART+CHECK", "PEEL+CHECK*3"],
           "TRI-PARTx2-PEELx3": ["SIEVE+CHECK", "PART+CHECK*2", "PEEL+CHECK*3"]}
if os.environ.get("LAYOUTS"):
    LAYOUTS = {k: LAYOUTS[k] for k in os.environ["LAYOUTS"].split(",")}
_CBS = None


def init(rots):
    global _CBS
    _CBS = [np.exp(1j * np.outer(np.arange(m), r)) for m, r in zip(MS, rots)]


def do(mode, v, state):
    """One station's work on one slot. state = {'ans', 'cand', 'done'}."""
    if state.get("done"): return state
    if mode == "SIEVE":
        state["cand"] = ("S", sieve(v, _CBS, MS))
    elif mode == "PART":
        res, ag = part(v, _CBS, MS); state["cand"] = ("P", res, ag)
    elif mode == "PEEL":
        f = peel_kernel(v, _CBS, MS); state["cand"] = ("L", f)
    elif mode == "CHECK":
        c = state.get("cand")
        if c and c[0] == "S" and check(v, c[1], _CBS) >= 0.9: state["ans"] = {solve_crt(c[1], MS)}; state["done"] = True
        elif c and c[0] == "P" and check(v, c[1], _CBS) >= 0.5 and c[2] >= 0.6: state["ans"] = {solve_crt(c[1], MS)}; state["done"] = True
        elif c and c[0] == "L" and c[1]: state["ans"] = {solve_crt(r, MS) for r in c[1]}; state["done"] = True
    return state


def station(name, q_in, q_out, rots):
    init(rots)
    while True:
        item = q_in.get()
        if item is None:
            break
        sid, v, state = item
        for mode in name.split("+"):
            state = do(mode, v, state)
        q_out.put((sid, v, state))


def make_stream(rng):
    rots = [rng.integers(0, m, D) * (2 * np.pi / m) for m in MS]
    slots = []
    for i in range(NSLOT):
        cls = ["CLEAN", "NOISY", "QUANT", "SUPER"][i % 4]
        if cls == "SUPER":
            xs = {int(x) for x in rng.choice(M, 4, replace=False)}; v = sum(encode_prod(x, rots) for x in xs)
        else:
            x = int(rng.integers(0, M)); ang = sum(x * r for r in rots)
            if cls == "NOISY": ang = ang + rng.vonmises(0, 16.0, D)
            if cls == "QUANT": ang = np.round(ang / (2 * np.pi / 16)) * (2 * np.pi / 16)
            v = np.exp(1j * ang); xs = {x}
        slots.append((i, cls, v.astype(np.complex128), xs))
    return rots, slots


def correct(ans, xs):
    return ans is not None and len(ans & xs) >= max(1, int(0.75 * len(xs))) and ans <= xs


def run_layout(name, stations, rots, slots):
    qs = [mp.Queue() for _ in range(len(stations) + 1)]
    procs = []
    for i, st in enumerate(stations):                      # "MODE*k" = k copies of the same face side by side (the prism)
        name, _, k = st.partition("*"); k = int(k) if k else 1
        for _ in range(k):
            procs.append(mp.Process(target=station, args=(name, qs[i], qs[i + 1], rots)))
    for p in procs: p.start()
    # warm-up: one slot of each class through the ring (lattice builds), not timed
    for s in slots[:4]: qs[0].put((s[0], s[2], {}))
    for _ in range(4): qs[-1].get()
    t0 = time.time(); sent = {}
    for s in slots[4:]:
        qs[0].put((s[0], s[2], {})); sent[s[0]] = time.time()
    results = {}; lat = []
    for _ in range(len(slots) - 4):
        sid, v, st = qs[-1].get(); results[sid] = st; lat.append(time.time() - sent[sid])
    dt = time.time() - t0
    for p in procs: p.terminate()
    for p in procs: p.join(timeout=10)
    acc = {}
    for sid, cls, v, xs in slots[4:]:
        acc.setdefault(cls, []).append(correct(results[sid].get("ans"), xs))
    return {"throughput_slots_per_s": (len(slots) - 4) / dt, "mean_latency_s": float(np.mean(lat)), "acc": {c: float(np.mean(a)) for c, a in acc.items()}, "seconds": dt}


def run_seq(rots, slots):
    init(rots); t0 = time.time(); acc = {}
    for s in slots[:4]:
        st = {}
        for mode in ["SIEVE", "CHECK", "PART", "CHECK", "PEEL", "CHECK"]: st = do(mode, s[2], st)
    t0 = time.time()
    for sid, cls, v, xs in slots[4:]:
        st = {}
        for mode in ["SIEVE", "CHECK", "PART", "CHECK", "PEEL", "CHECK"]: st = do(mode, v, st)
        acc.setdefault(cls, []).append(correct(st.get("ans"), xs))
    dt = time.time() - t0
    return {"throughput_slots_per_s": (len(slots) - 4) / dt, "mean_latency_s": dt / (len(slots) - 4), "acc": {c: float(np.mean(a)) for c, a in acc.items()}, "seconds": dt}


def main():
    rng = np.random.default_rng(11); rots, slots = make_stream(rng); out = {}
    out["SEQ"] = run_seq(rots, slots); print("SEQ", out["SEQ"], flush=True)
    for name, st in LAYOUTS.items():
        out[name] = run_layout(name, st, rots, slots); print(name, out[name], flush=True)
    base = out["SEQ"]["throughput_slots_per_s"]; best = max(LAYOUTS, key=lambda k: out[k]["throughput_slots_per_s"])
    out["verdict"] = {"best_shape": best, "speedup": out[best]["throughput_slots_per_s"] / base,
                      "PASS": bool(out[best]["throughput_slots_per_s"] >= 2.5 * base and all(out[best]["acc"][c] >= out["SEQ"]["acc"][c] - 0.02 for c in out["SEQ"]["acc"]))}
    json.dump(out, open(OUT, "w"), indent=1); print("PIPE-0", out["verdict"])


if __name__ == "__main__":
    mp.set_start_method("fork"); main()
