#!/usr/bin/env python3
"""CORTEX-2 step 3 — FIRE: the VM as an always-on tick loop on the live fleet event stream.

Not request-response: the loop ticks at a fixed rate whether or not a new event arrived; every tick runs the program
register (the sifter's own code) and leaves a mark with the program hash. Queries come from control-center/events.jsonl
(the live source since 2026-09-12): the event's text is hash-encoded (ENC-0, seed.codeword — no model in the loop) and
sifted against a lattice of the distinct step labels seen so far (the lattice grows as new labels appear: marks written,
not trained). Programs: NARROW (commit on margin) <-> WIDE (abstain-triggered rewrite), as in test_vm G4.
Usage: python3 vm_loop.py --hours 1 [--hz 50] [--out data/vm-marks.jsonl]
G3: 1 h unattended; every answer traceable to its program sha; 0 refused programs on the clean stream; idle tick <= 1 ms.
"""
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from omni_ring import seed  # noqa: E402
from omni_ring.vm import VM, ProgramCodebook, iid  # noqa: E402

EVENTS = "<your-data>/control-center/events.jsonl"


def label_of(ev: dict) -> str:
    return str(ev.get("step") or ev.get("verb") or "?")


def text_of(ev: dict) -> str:
    return f"{ev.get('verb', '')} {ev.get('step', '')} {ev.get('note', '')}".strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=1.0)
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--out", default="<your-data>/data/vm-marks.jsonl")
    ap.add_argument("--summary", default=os.path.expanduser("~/gpurun-out-0930/cortex2_loop.result.json"))
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    labels, rows = [], []
    with open(EVENTS, "rb") as f:
        f.seek(0, 2); size = f.tell(); f.seek(max(0, size - 4_000_000))
        for line in f.read().decode(errors="ignore").splitlines()[1:]:
            try:
                lab = label_of(json.loads(line))
            except Exception:
                continue
            if lab not in labels:
                labels.append(lab); rows.append(seed.codeword("L:" + lab, 0))
        start_pos = size
    lattice = np.stack(rows)
    cb = ProgramCodebook(seed.D)
    vm = VM(lattice, cb)
    narrow = cb.write([iid("LOAD", 0), iid("SIFT", 1), iid("THRESHOLD", 6), iid("PROG", 1)])
    wide = cb.write([iid("LOAD", 0), iid("SIFT", 7), iid("THRESHOLD", 2), iid("JUMP", 0)])
    vm.programs[0], vm.programs[1] = narrow, wide
    vm.load_program(narrow)
    vm.set(0, lattice[0])
    period = 1.0 / a.hz
    t_end = time.time() + 3600 * a.hours
    n_events = n_idle = refused = abstains = commits = 0
    idle_ms, busy_ms = [], []
    pos = start_pos
    out = open(a.out, "a")
    while time.time() < t_end:
        t0 = time.time()
        new = []
        try:
            with open(EVENTS, "rb") as f:
                f.seek(pos); chunk = f.read(); pos += len(chunk)
            for line in chunk.decode(errors="ignore").splitlines():
                if line.strip():
                    try:
                        new.append(json.loads(line))
                    except Exception:
                        pass
        except OSError:
            pass
        if new:
            for ev in new:
                lab = label_of(ev)
                if lab not in labels:                       # a new label = a new mark written, no training
                    labels.append(lab); vm.append(seed.codeword("L:" + lab, 0)); lattice = vm.lattice   # memo extends by the new row only
                vm.set(0, seed.codeword("L:" + lab, 0))     # the live query (hash-encoded)
                m = vm.tick(); n_events += 1
                refused += int(m.refused); abstains += int(m.abstain); commits += int(not m.abstain and bool(m.topk))
                out.write(json.dumps({"ts": time.time(), "tick": m.tick, "program": m.program, "label": lab,
                                      "top": [labels[i] for i in m.topk[:3]], "abstain": m.abstain, "refused": m.refused}) + "\n")
            busy_ms.append((time.time() - t0) * 1000 / len(new))
        else:
            m = vm.tick(); n_idle += 1; refused += int(m.refused)
            idle_ms.append((time.time() - t0) * 1000)
        if n_idle % 1000 == 0 and n_idle:
            out.flush()
        time.sleep(max(0.0, period - (time.time() - t0)))
    out.close()
    idle_p50 = float(np.median(idle_ms)) if idle_ms else None
    idle_p99 = float(np.percentile(idle_ms, 99)) if idle_ms else None
    busy_p50 = float(np.median(busy_ms)) if busy_ms else None
    # bool() everywhere: numpy scalars are not JSON-serialisable, and `a and np.False_` returns np.False_,
    # which used to crash json.dump on exactly the ticks that MISSED the budget — hiding a failing verdict.
    idle_ok = bool(idle_p50 is not None and idle_p50 <= 1.0)
    res = {"hours": a.hours, "hz": a.hz, "events": int(n_events), "idle_ticks": int(n_idle), "labels": int(len(labels)),
           "refused": int(refused), "abstains": int(abstains), "commits": int(commits),
           "idle_ms_p50": idle_p50, "idle_ms_p99": idle_p99, "busy_ms_p50": busy_p50,
           "verdict": {"PASS": bool(refused == 0 and idle_ok and n_events > 0),
                       "no_refused": bool(refused == 0), "idle_tick_le_1ms": idle_ok,
                       "saw_events": bool(n_events > 0)}}
    json.dump(res, open(a.summary, "w"), indent=1)
    print("CORTEX-2 loop", res)


if __name__ == "__main__":
    main()
