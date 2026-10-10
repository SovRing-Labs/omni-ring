#!/usr/bin/env python3
"""ROUTER-0 (H5): frozen simplex keys + learned projection vs learned linear head."""
import json, os
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
OUT = os.path.expanduser("~/gpurun-out-0930/router0.result.json")
torch.set_num_threads(2); D, C, N_TR, N_TE = 64, 4, 20000, 5000
TETRA = torch.tensor([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=torch.float32) / np.sqrt(3)


def data(seed, sep=1.2, noise=1.0):
    g = torch.Generator().manual_seed(seed); mu = torch.randn(C, D, generator=g) * sep / np.sqrt(D) * 4
    y = torch.randint(0, C, (N_TR + N_TE,), generator=g); x = mu[y] + noise * torch.randn(N_TR + N_TE, D, generator=g)
    return x[:N_TR], y[:N_TR], x[N_TR:], y[N_TR:]


class Learned(nn.Module):
    def __init__(self): super().__init__(); self.w = nn.Linear(D, C)
    def forward(self, x): return self.w(x)


class Simplex(nn.Module):
    def __init__(self): super().__init__(); self.p = nn.Linear(D, 3); self.t = nn.Parameter(torch.tensor(5.0))
    def forward(self, x, keys=TETRA):
        z = F.normalize(self.p(x), dim=1); return self.t * z @ keys.T


def ece(p, y, bins=15):
    conf, pred = p.max(1); acc = (pred == y).float(); e = 0.0
    for b in range(bins):
        m = (conf > b / bins) & (conf <= (b + 1) / bins)
        if m.any(): e += m.float().mean() * (acc[m].mean() - conf[m].mean()).abs()
    return float(e)


def train(model, x, y, steps=1500):
    opt = torch.optim.Adam(model.parameters(), 1e-2)
    for _ in range(steps):
        i = torch.randint(0, len(x), (256,)); loss = F.cross_entropy(model(x[i]), y[i]); opt.zero_grad(); loss.backward(); opt.step()
    return model


def main():
    res = {"LEARNED": [], "SIMPLEX": [], "freeze_spread": []}
    for s in range(3):
        xtr, ytr, xte, yte = data(s)
        for name, M in (("LEARNED", Learned), ("SIMPLEX", Simplex)):
            m = train(M(), xtr, ytr)
            with torch.no_grad():
                p = F.softmax(m(xte), 1); acc = float((p.argmax(1) == yte).float().mean())
            res[name].append({"acc": acc, "ece": ece(p, yte)})
            if name == "SIMPLEX":
                accs = []
                for drop in range(C):
                    keep = yte != drop; keys = TETRA[[c for c in range(C) if c != drop]]
                    with torch.no_grad():
                        z = F.normalize(m.p(xte[keep]), dim=1) @ keys.T
                    lab = yte[keep]; lab = lab - (lab > drop).int()
                    accs.append(float((z.argmax(1) == lab).float().mean()))
                res["freeze_spread"].append(float(np.max(accs) - np.min(accs)))
        print(f"seed {s}: LEARNED {res['LEARNED'][-1]}  SIMPLEX {res['SIMPLEX'][-1]}  freeze spread {res['freeze_spread'][-1]:.4f}", flush=True)
    la = np.mean([r["acc"] for r in res["LEARNED"]]); sa = np.mean([r["acc"] for r in res["SIMPLEX"]])
    le = np.mean([r["ece"] for r in res["LEARNED"]]); se = np.mean([r["ece"] for r in res["SIMPLEX"]]); fs = float(np.mean(res["freeze_spread"]))
    res["summary"] = {"acc_learned": la, "acc_simplex": sa, "ece_learned": le, "ece_simplex": se, "freeze_spread": fs}
    res["verdict"] = {"PASS": bool(sa >= la - 0.01 and se <= le and fs <= 0.01), "KILL": bool(sa < la - 0.02)}; res["summary"] = {k: float(v) for k, v in res["summary"].items()}
    json.dump(res, open(OUT, "w"), indent=1); print("ROUTER-0", res["summary"], res["verdict"])


if __name__ == "__main__":
    main()
