#!/usr/bin/env python3
"""CAL-3 (MAIN-0-HYPOTHESES plan (unpublished) §4, pre-registered 2026-10-02 BEFORE this harness was written; runs only after J's
D1-D4): the first learned-model test of the skeleton -- positions.

Model: tlm.py recipe at toy size (4L d128 h4 ctx256, BitLinear blocks, TLM-0's 8k byte-BPE tokenizer reused, our corpus
~/chew-out/corpus.jsonl.zst). Three arms, same seed, same data order, same FIXED WALL CLOCK (MINUTES per arm;
deviation from the plan's "≥ 20M tokens", declared before running: this CPU trains ~3-5k tok/s at this size):
  LEARNED  learned position table (4*CTX rows; only the first CTX rows are ever trained -- the rest stay random)
  ROPE     rotary on q,k; pair i rotates by pos * base^(-2i/d_head), base 10000 (Su 2021)
  RESIDUE  rotary on q,k; pair i rotates by pos * 2*pi / m_i with m_i the i-th prime from {3,5,7,11,...} (E3a: coprime
           periods -> position is exact and CRT-unique up to prod(m_i); a shift is a rotation)
Training mix: 90% corpus windows + 10% synthetic REPEAT sequences (random tokens of length n <= CTX/2, written twice) so
the copy skill exists in every arm.
Tasks: (a) eval nats on held-out corpus at CTX (in-dist); (b) REPEAT copy accuracy (second copy, teacher-forced) at
distance 64 and 128 inside CTX; (c) REPEAT copy at distance CTX and 2*CTX (windows 2x and 4x CTX, never seen in training; DEVIATION declared before the
     full run: the plan said 384/768 -- a repeat at distance n needs a window of 2n, so the far distances are 256 and 512);
(d) reported: eval nats on 2x and 4x windows.
FIELD PREDICTS: (a) ROPE = LEARNED +- 0.02 nats; (c) ROPE and LEARNED near chance (Press 2022; Peng 2023 YaRN).
OURS (H2): (a) RESIDUE within +-0.02 nats of ROPE; (b) all arms >= 0.9; (c) RESIDUE >= ROPE + 20 pts at 384 and 768.
KILL: (a) RESIDUE worse than ROPE by > 0.05 nats. Honest expectation recorded: RESIDUE may fail (c) too -- an exact
address is not understanding (A2); that FAIL is the finding (long range needs training-at-length or shards).
Verify: SMOKE=1 python3 cal3.py (seconds); full: MINUTES=25 python3 cal3.py; result ~/gpurun-out-0930/cal3.result.json
"""
import json, math, os, subprocess, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.expanduser("~/gpurun-out-0930/gpurun"))
from ternary import BitLinear, ternary_report  # noqa: E402

SMOKE = os.environ.get("SMOKE") == "1"
MINUTES = float(os.environ.get("MINUTES", "25"))
ARMS = os.environ.get("ARMS", "LEARNED,ROPE,RESIDUE").split(",")
OUT = os.path.expanduser(os.environ.get("CAL3_OUT", "~/gpurun-out-0930/cal3.result.json"))
TOK = os.path.expanduser("~/gpurun-out-0930/gpurun/tlm/tokenizer.json")
DATA = "~/chew-out/corpus.jsonl.zst"
V, D, L, H, CTX, B = (8192, 64, 2, 2, 64, 8) if SMOKE else (8192, 128, 4, 4, 256, 16)
DH = D // H
PRIMES = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137]
torch.manual_seed(0); np.random.seed(0)
torch.set_num_threads(os.cpu_count())


def load():
    rows = [json.loads(x) for x in subprocess.run(["zstd", "-dc", DATA], capture_output=True, text=True, check=True).stdout.splitlines()]
    if SMOKE:
        rows = rows[:3000]
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(TOK)
    eos = tok.token_to_id("<|eos|>")
    out = {}
    for split in ("train", "eval"):
        ids = []
        for enc in tok.encode_batch([r["input"]["text"] for r in rows if r["split"] == split]):
            ids.extend(enc.ids); ids.append(eos)
        out[split] = torch.tensor(ids, dtype=torch.long)
    return out["train"], out["eval"]


def rope_angles(kind, T):
    i = torch.arange(DH // 2)
    if kind == "ROPE":
        theta = 10000.0 ** (-2 * i / DH)
    else:
        theta = 2 * math.pi / torch.tensor(PRIMES[: DH // 2], dtype=torch.float32)
    ang = torch.arange(T)[:, None].float() * theta[None, :]
    return ang.cos(), ang.sin()


def rotate(x, cos, sin):  # x: b h t dh
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c, s = cos[None, None], sin[None, None]
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], -1).flatten(-2)


class Block(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.n1, self.n2 = nn.RMSNorm(D), nn.RMSNorm(D)
        self.qkv, self.o = BitLinear(D, 3 * D), BitLinear(D, D)
        self.up, self.down = BitLinear(D, 4 * D), BitLinear(4 * D, D)

    def forward(self, x, cs):
        b, t, _ = x.shape
        q, k, v = self.qkv(self.n1(x)).view(b, t, 3, H, DH).permute(2, 0, 3, 1, 4)
        if cs is not None:
            q, k = rotate(q, *cs), rotate(k, *cs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(b, t, D)
        x = x + self.o(y)
        return x + self.down(F.relu(self.up(self.n2(x))) ** 2)


class LM(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.emb = nn.Embedding(V, D)
        self.pos = nn.Embedding(4 * CTX, D) if kind == "LEARNED" else None
        self.blocks = nn.ModuleList(Block(kind) for _ in range(L))
        self.nf = nn.RMSNorm(D)

    def forward(self, idx):
        t = idx.shape[1]
        x = self.emb(idx)
        cs = None
        if self.kind == "LEARNED":
            x = x + self.pos(torch.arange(t))
        else:
            cs = rope_angles(self.kind, t)
        for blk in self.blocks:
            x = blk(x, cs)
        return self.nf(x) @ self.emb.weight.T


def repeat_batch(g, n, T, b=B):
    """random tokens of length n, written twice, padded to T with more random tokens; returns x, y, mask of 2nd copy."""
    seq = torch.randint(1, V, (b, T + 1), generator=g)
    seq[:, n:2 * n] = seq[:, :n]
    mask = torch.zeros(b, T, dtype=torch.bool)
    mask[:, n:2 * n - 1] = True          # predicting positions n+1 .. 2n-1 (targets seq[n+1:2n])
    return seq[:, :-1], seq[:, 1:], mask


def corpus_batch(g, data, T, b=B):
    ix = torch.randint(0, len(data) - T - 1, (b,), generator=g)
    return torch.stack([data[i:i + T] for i in ix]), torch.stack([data[i + 1:i + T + 1] for i in ix])


@torch.no_grad()
def eval_nats(model, data, T, n=20):
    model.eval(); g = torch.Generator().manual_seed(1); tot = 0.0
    for _ in range(n):
        x, y = corpus_batch(g, data, T)
        tot += F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1)).item()
    model.train(); return tot / n


@torch.no_grad()
def eval_copy(model, dist, T, n=20):
    model.eval(); g = torch.Generator().manual_seed(2); hit = tot = 0
    for _ in range(n):
        x, y, m = repeat_batch(g, dist, T)
        p = model(x).argmax(-1)
        hit += int((p[m] == y[m]).sum()); tot += int(m.sum())
    model.train(); return hit / tot


def train_arm(kind, dtr, dev_):
    torch.manual_seed(0)
    model = LM(kind); nparam = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, betas=(0.9, 0.95), weight_decay=0.1)
    g = torch.Generator().manual_seed(7)
    budget = 15 if SMOKE else 60 * MINUTES
    t0 = time.time(); step = seen = 0; curve = {}
    print(f"[{kind}] params {nparam / 1e6:.2f}M budget {budget:.0f}s", flush=True)
    while time.time() - t0 < budget:
        frac = min(1.0, (time.time() - t0) / budget)
        for gr in opt.param_groups:
            gr["lr"] = 1.5e-3 * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac))) * min(1.0, (step + 1) / 100)
        if torch.rand(1, generator=g).item() < 0.10:
            n = int(torch.randint(8, CTX // 2 + 1, (1,), generator=g))
            x, y, _ = repeat_batch(g, n, CTX)
        else:
            x, y = corpus_batch(g, dtr, CTX)
        loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        step += 1; seen += B * CTX
        if step % (5 if SMOKE else 200) == 0:
            e = eval_nats(model, dev_, CTX, 5); curve[step] = e
            print(f"  [{kind}] step {step} train {loss.item():.3f} eval {e:.3f} | {seen / 1e6:.2f}M tok | {seen / (time.time() - t0):,.0f} tok/s", flush=True)
    r = {"params": nparam, "steps": step, "tokens": seen, "curve": curve,
         "nats_1x": eval_nats(model, dev_, CTX, 40), "nats_2x": eval_nats(model, dev_, 2 * CTX, 20), "nats_4x": eval_nats(model, dev_, 4 * CTX, 10),
         "copy": {str(d): eval_copy(model, d, T) for d, T in ((CTX // 4, CTX // 2), (CTX // 2, CTX), (CTX, 2 * CTX), (2 * CTX, 4 * CTX))},
         "ternary_ok": ternary_report(model)[3], "seconds": time.time() - t0}
    print(f"[{kind}] nats 1x {r['nats_1x']:.3f} 2x {r['nats_2x']:.3f} 4x {r['nats_4x']:.3f} | copy {r['copy']}", flush=True)
    return r


def main():
    dtr, dev_ = load()
    print(f"CAL-3 train {len(dtr):,} eval {len(dev_):,} tokens; CTX {CTX} D {D} L {L} H {H}; arms {ARMS}", flush=True)
    res = {k: train_arm(k, dtr, dev_) for k in ARMS}
    v = {}
    if {"ROPE", "RESIDUE", "LEARNED"} <= set(res):
        ro, re, le = res["ROPE"], res["RESIDUE"], res["LEARNED"]
        far = [str(CTX), str(2 * CTX)]
        v = {"a_field_rope_eq_learned": abs(ro["nats_1x"] - le["nats_1x"]) <= 0.02,
             "a_ours_residue_eq_rope": abs(re["nats_1x"] - ro["nats_1x"]) <= 0.02,
             "b_all_ge_0.9": all(res[k]["copy"][d] >= 0.9 for k in res for d in (str(CTX // 4), str(CTX // 2))),
             "c_residue_beats_rope_20pts": all(re["copy"][d] >= ro["copy"][d] + 0.20 for d in far),
             "KILL_residue_worse_0.05": re["nats_1x"] - ro["nats_1x"] > 0.05}
        v["PASS"] = v["a_ours_residue_eq_rope"] and v["b_all_ge_0.9"] and v["c_residue_beats_rope_20pts"] and not v["KILL_residue_worse_0.05"]
    json.dump({"arms": res, "verdict": v, "CTX": CTX, "D": D, "L": L, "H": H, "smoke": SMOKE, "minutes": MINUTES}, open(OUT, "w"), indent=1)
    print("CAL-3 verdict", v)


if __name__ == "__main__":
    main()
