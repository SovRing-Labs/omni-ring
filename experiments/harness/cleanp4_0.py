#!/usr/bin/env python3
"""CLEAN-P4-0 (PREREGISTRATIONS-OMNITRI.md wave-2): triangle clean-up as an omniring packed sift on phase-quantized u vs float clean-up."""
import ctypes, json, os, sys, time
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
sys.path.insert(0, ".")
import omni_ring.seed as S
from tri_wave import Space, PH, D, int8
ctypes.CDLL("libgomp.so.1").omp_set_num_threads(1)
sp = Space(1000, 50, seed=21); g = np.random.default_rng(22)
packed = S.pack4(sp.role[2].astype(np.uint8)); cr, ci = sp.clean[2]; res = {}
for N in (16, 32, 64, 128, 192, 256):
    same = ok_f = ok_p = tot = 0; tf = tp = 0.0
    for b in range(12):
        T = np.stack([g.integers(0, 1000, N), g.integers(0, 50, N), g.integers(0, 1000, N)], 1)
        acc = int8(PH[sp.tri(T[:, 0], T[:, 1], T[:, 2])].sum(0)); Q = T[:min(N, 64)]
        u = acc * np.conj(PH[sp.key(2, Q[:, 0], Q[:, 1])])
        ur, ui = np.ascontiguousarray(u.real, np.float32), np.ascontiguousarray(u.imag, np.float32)
        t = time.perf_counter(); f = (ur @ cr.T + ui @ ci.T).argmax(1); tf += time.perf_counter() - t
        t = time.perf_counter()
        qa = (np.rint(np.angle(u) / (np.pi / 8)).astype(np.int64) % 16).astype(np.uint8)
        p = S.sift_batch_p4(qa, packed).argmax(1); tp += time.perf_counter() - t
        same += int((f == p).sum()); ok_f += int((f == Q[:, 2]).sum()); ok_p += int((p == Q[:, 2]).sum()); tot += len(Q)
    res[N] = {"same": same / tot, "float_acc": ok_f / tot, "p4_acc": ok_p / tot, "float_us_per_q": tf / tot * 1e6, "p4_us_per_q": tp / tot * 1e6}
    print(f"CLEAN-P4 N={N:3d} same {same/tot:.4f} | float acc {ok_f/tot:.4f} p4 acc {ok_p/tot:.4f} | float {tf/tot*1e6:.0f} us/q, p4 {tp/tot*1e6:.0f} us/q", flush=True)
sp_x = np.mean([res[N]["float_us_per_q"] / res[N]["p4_us_per_q"] for N in res])
res["verdict"] = {"PASS": all(res[N]["same"] >= 0.995 for N in (16, 32, 64, 128)) and sp_x >= 1.5, "speedup": float(sp_x),
                  "KILL": res[64]["same"] < 0.98}
print("CLEAN-P4-0:", res["verdict"], flush=True)
json.dump({str(k): v for k, v in res.items()}, open(os.path.expanduser("~/gpurun-out-0930/cleanp4_0.result.json"), "w"), indent=1, default=str)
