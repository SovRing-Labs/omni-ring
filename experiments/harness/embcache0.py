#!/usr/bin/env python3
"""EMB-CACHE-0 (PREREGISTRATIONS-OMNITRI.md wave-3): how many embeds a cache in front of the pen could avoid on the live stream."""
import hashlib, json, os, re, socket, sys
import numpy as np
EV = "<your-data>/control-center/events.jsonl"; SOCK = "/tmp/ring_headd.sock"


def simhash(t):
    w = re.sub(r"\d", "0", t.lower()).split(); sh = [" ".join(w[i:i + 3]) for i in range(max(1, len(w) - 2))]
    v = np.zeros(64)
    for s in sh:
        h = int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little")
        v += np.array([1 if (h >> i) & 1 else -1 for i in range(64)])
    return int("".join("1" if x > 0 else "0" for x in v[::-1]), 2)


def ask(text):
    s = socket.socket(socket.AF_UNIX); s.settimeout(120); s.connect(SOCK)
    s.sendall(json.dumps({"head": "d5.verb", "text": text}).encode()); s.shutdown(socket.SHUT_WR); buf = b""
    while (k := s.recv(65536)):
        buf += k
    s.close(); return json.loads(buf)


notes = []
for line in open(EV):
    try:
        n = json.loads(line).get("note")
    except Exception:
        continue
    if n:
        notes.append(n)
seen, sims, exact, near, pairs = set(), {}, 0, 0, []
for n in notes:
    if n in seen:
        exact += 1; continue
    h = simhash(n)
    if h in sims:
        near += 1; pairs.append((sims[h], n))
    else:
        sims[h] = n
    seen.add(n)
N = len(notes); res = {"notes": N, "distinct": len(seen), "exact_dup": exact / N, "near_dup": near / N}
print(f"EMB-CACHE stream {N} notes, distinct {len(seen)} | exact dup {exact/N:.3f} | near (SimHash) {near/N:.3f}", flush=True)
g = np.random.default_rng(40); pick = [pairs[i] for i in g.choice(len(pairs), min(300, len(pairs)), replace=False)]
agree = 0; key = None
for a, b in pick:
    ra, rb = ask(a), ask(b)
    key = key or next(k for k in ("label", "pred", "answer") if k in ra)
    agree += int(ra[key] == rb[key])
res["near_pairs_checked"] = len(pick); res["near_label_agree"] = agree / max(len(pick), 1)
safe_near = res["near_dup"] if agree == len(pick) else 0.0
res["avoidable"] = res["exact_dup"] + safe_near
res["avoidable_if_near_accepted_at_measured_agreement"] = res["exact_dup"] + res["near_dup"]
res["verdict"] = {"PASS": res["avoidable"] >= 0.20}
print(f"near pairs label agreement {agree}/{len(pick)} | avoidable {res['avoidable']:.3f} | EMB-CACHE-0: {res['verdict']}", flush=True)
json.dump(res, open(os.path.expanduser("~/gpurun-out-0930/embcache0.result.json"), "w"), indent=1)
