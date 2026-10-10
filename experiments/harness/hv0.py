"""HV-0: the omnid hypervector wire on real fleet tool requests (pre-registered in OMNID plan (unpublished))."""
import json, random, sys, time, hashlib
import numpy as np
sys.path.insert(0, "<your-data>/shards"); sys.path.insert(0, ".")
from episodes import DBS, load
from omni_ring.shards import symbol, SEED_GEN

K, D = 16, 5120
W = np.exp(2j * np.pi * np.arange(K) / K).astype(np.complex64)
steps, _ = load(DBS["current"])
trip = []
for st in steps.values():
    for rich, coarse in st:
        tool = rich.split(":", 1)[0]; head = coarse.split(":", 1)[1] if ":" in coarse else "-"; arg = rich.split(":", 1)[1] if ":" in rich else "-"
        trip.append((tool, head, arg))
distinct = sorted(set(trip)); random.Random(0).shuffle(distinct); S = distinct[:5000]
atom_cache = {}
def atom(lbl):
    if lbl not in atom_cache: atom_cache[lbl] = symbol(lbl, gen=SEED_GEN).astype(np.int16)
    return atom_cache[lbl]
R = {f: atom("ROLE:" + f) for f in ("tool", "head", "arg")}
def encode(t):
    z = sum(W[(R[f] + atom(f"{f}:{v}")) % K] for f, v in zip(("tool", "head", "arg"), t))
    p = (np.round(np.angle(z) / (2 * np.pi) * K).astype(np.int64) % K).astype(np.uint8)
    return (p[0::2] | (p[1::2] << 4)).astype(np.uint8)            # p4 packed, 2,560 B
def unpack(b):
    p = np.empty(D, np.int16); p[0::2] = b & 15; p[1::2] = b >> 4; return p
t0 = time.perf_counter(); F = [encode(t) for t in S]; enc_ms = (time.perf_counter() - t0) * 1000 / len(S)
books = {f: sorted({t[i] for t in S}) for i, f in enumerate(("tool", "head", "arg"))}
C = {f: W[np.array([atom(f"{f}:{v}") for v in books[f]])] for f in books}
acc = {}; t0 = time.perf_counter()
P = np.array([unpack(b) for b in F])
for i, f in enumerate(("tool", "head", "arg")):
    U = W[(P - R[f]) % K]
    S_ = (U @ np.conj(C[f]).T).real
    pred = S_.argmax(1)
    acc[f] = float(np.mean([books[f][pred[k]] == S[k][i] for k in range(len(S))]))
dec_us = (time.perf_counter() - t0) * 1e6 / (3 * len(S))
dig = [hashlib.sha256(b.tobytes()).hexdigest() for b in F]
collisions = len(dig) - len(set(dig))
same = all(np.array_equal(encode(S[k]), F[k]) for k in range(200))
def sim(a, b): return float(np.mean(np.cos(2 * np.pi * (unpack(a).astype(int) - unpack(b).astype(int)) / K)))
by = {}
for k, t in enumerate(S): by.setdefault((t[0], t[1]), []).append(k)
near = [sim(F[v[0]], F[v[1]]) for v in by.values() if len(v) > 1][:500]
far = [sim(F[a], F[b]) for a, b in zip(range(0, 1000, 2), range(1, 1000, 2)) if S[a][0] != S[b][0]]
out = dict(n_steps=len(trip), n_distinct=len(distinct), n_tested=len(S), tool_acc=acc["tool"], head_acc=acc["head"], arg_acc=acc["arg"],
           identical_same_frame=same, collisions=collisions, encode_ms=enc_ms, decode_us_per_field=dec_us,
           near_dup_sim=float(np.mean(near)), diff_tool_sim=float(np.mean(far)), frame_bytes=int(F[0].nbytes),
           codebooks={f: len(v) for f, v in books.items()})
out["PASS"] = bool(acc["tool"] >= 0.999 and acc["head"] >= 0.99 and acc["arg"] >= 0.98 and same and collisions == 0 and enc_ms <= 1)
print(json.dumps(out, indent=1)); json.dump(out, open("experiments/results/hv0.result.json", "w"), indent=1)
