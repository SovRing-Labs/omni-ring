#!/usr/bin/env python3
"""DESIGNED-0 (PREREGISTRATIONS-OMNITRI.md): TRI-0 + TRI-2b with designed (exactly orthogonal) codebooks instead of random ones."""
import json, os, sys
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tri_wave as TW

K = 16


def kron_orthogonal(n, d, seed):
    """n exactly orthogonal phase rows: Kronecker product of harmonic frames of sizes dividing 16, tiled to d."""
    M = 1
    while M < n:
        M *= 2
    sizes, m = [], M
    while m > 1:
        f = 16 if m % 16 == 0 else m
        sizes.append(f); m //= f
    rows = np.arange(M)
    ph = np.zeros((M, M), np.int64)
    rdig, cdig, rr, cc = [], [], rows.copy(), rows.copy()
    for f in sizes:
        rdig.append(rr % f); cdig.append(cc % f); rr //= f; cc //= f
    for f, rd, cd in zip(sizes, rdig, cdig):
        ph += (rd[:, None] * cd[None, :] * (K // f))
    block = ph[:n] % K
    L = d // M
    u = np.random.default_rng(seed).integers(0, K, L)
    return ((block[:, :, None] + u[None, None, :]) % K).reshape(n, M * L)


class DesignedSpace(TW.Space):
    def __init__(self, n_ent, n_rel, seed=0):
        super().__init__(n_ent, n_rel, seed)                       # same permutations as the random baseline
        self.E = kron_orthogonal(n_ent, TW.D, seed + 101)
        self.R = kron_orthogonal(n_rel, TW.D, seed + 202)
        self.role = [self.E[:, self.P[0]], self.R[:, self.P[1]], self.E[:, self.P[2]]]
        self.clean = [(np.cos(2 * np.pi * c / 16).astype(np.float32), np.sin(2 * np.pi * c / 16).astype(np.float32))
                      for c in self.role]


if __name__ == "__main__":
    E = kron_orthogonal(1000, TW.D, 1)
    z = np.exp(2j * np.pi * E[:50].astype(float) / 16)
    g = (z @ z.conj().T).real / TW.D
    print("designed entity Gram check: max |off-diagonal| = %.2e" % np.abs(g - np.eye(50)).max(), flush=True)
    TW.Space = DesignedSpace                                        # patch BEFORE tri_wave2 imports it
    import tri_wave2 as TW2
    part = sys.argv[1] if len(sys.argv) > 1 else "all"
    res = {}
    if part in ("tri0", "all"):
        res["TRI-0"] = TW.tri0()
    if part in ("tri2b", "all"):
        res["TRI-2b"] = TW2.tri2b()
    json.dump(res, open(os.path.expanduser(f"~/gpurun-out-0930/designed0.{part}.json"), "w"), indent=1, default=str)
