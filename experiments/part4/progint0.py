#!/usr/bin/env python3
"""PROG-INTEGRITY-0: keyed check slots vs public checksums for VM programs (A11)."""
import hashlib, hmac, json, os, sys
import numpy as np
sys.path.insert(0, ".")
from omni_ring import seed
from omni_ring.vm import ProgramCodebook, iid, L, NI

OUT = os.path.expanduser("~/gpurun-out-0930/progint0.result.json")
KEY = b"fleet-secret-key-demo"


def keyed_checks(ids, key=KEY):
    h = hmac.new(key, bytes(int(i) for i in ids), hashlib.sha256).digest()
    return int.from_bytes(h[:4], "big") % NI, int.from_bytes(h[4:8], "big") % 103


class KeyedCodebook(ProgramCodebook):
    def write(self, ids, check=None):
        ids = list(ids) + [iid("NOP")] * (L - len(ids))
        return super().write(ids, check=keyed_checks(ids) if check is None else check)

    def read(self, vec):
        U = vec[None, :] * np.conj(np.exp(2j * np.pi * np.arange(16) / 16)[self.POS])
        sc = U.real @ self.IC.T + U.imag @ self.IS.T
        ids = [int(x) for x in sc.argmax(1)]
        return ids[:L] if tuple(keyed_checks(ids[:L])) == (ids[L], ids[L + 1]) else None


def main():
    d = 1280; rng = np.random.default_rng(0); out = {}
    for arm, CB in (("PUBLIC", ProgramCodebook(d)), ("KEYED", KeyedCodebook(d))):
        # random vectors
        acc_r = sum(CB.read(np.exp(1j * rng.uniform(0, 2 * np.pi, d)).astype(np.complex64)) is not None for _ in range(50_000))
        # chosen-program forgery: forger writes program ids with checks computed by the PUBLIC rule (no key)
        forged = 0; n_f = 50_000 if arm == "KEYED" else 2_000
        for _ in range(n_f):
            ids = [int(x) for x in rng.integers(0, NI, L)]
            c = (ProgramCodebook.checksum1(ids), ProgramCodebook.checksum2(ids)) if arm == "PUBLIC" else (int(rng.integers(0, NI)), int(rng.integers(0, 103)))
            forged += CB.read(ProgramCodebook.write(CB, ids, check=c)) is not None
        # tamper: one slot changed, checks kept
        caught = 0
        for _ in range(2_000):
            ids = [int(x) for x in rng.integers(0, NI, L)]; good = CB.write(ids); k = int(rng.integers(0, L))
            bad = list(ids); bad[k] = (bad[k] + int(rng.integers(1, NI))) % NI
            ch = keyed_checks(ids) if arm == "KEYED" else (ProgramCodebook.checksum1(ids), ProgramCodebook.checksum2(ids))
            caught += CB.read(ProgramCodebook.write(CB, bad, check=ch)) is None
        out[arm] = {"random_accept": acc_r / 50_000, "forgery": forged / n_f, "tamper_caught": caught / 2_000}
        print(arm, out[arm], flush=True)
    base = 1 / (NI * 103)
    out["analytic_random"] = base
    out["verdict"] = {"PASS": out["KEYED"]["forgery"] <= 2e-4 and all(out[a]["random_accept"] <= 2 * base + 1e-9 for a in ("PUBLIC", "KEYED")) and all(out[a]["tamper_caught"] == 1.0 for a in ("PUBLIC", "KEYED"))}
    json.dump(out, open(OUT, "w"), indent=1); print("PROG-INTEGRITY-0", out["verdict"], "analytic random", base)


if __name__ == "__main__":
    main()
