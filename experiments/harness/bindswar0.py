#!/usr/bin/env python3
"""BIND-SWAR-0 (PREREGISTRATIONS-OMNITRI.md wave-2): phase bind/unbind per representation; bit-identity + ns per phase (1 thread)."""
import ctypes, json, os, time
import numpy as np
n = 10_000_000; g = np.random.default_rng(20)
a8, b8 = g.integers(0, 16, n, dtype=np.uint8), g.integers(0, 16, n, dtype=np.uint8)
truth_b, truth_u = ((a8.astype(np.int64) + b8) % 16).astype(np.uint8), ((a8.astype(np.int64) - b8) % 16).astype(np.uint8)
pack = lambda x: (x[0::2] | (x[1::2] << 4)).astype(np.uint8)  # noqa: E731
unpack = lambda p: np.stack([p & 15, p >> 4], 1).reshape(-1)  # noqa: E731
pa, pb = pack(a8), pack(b8); wa, wb = pa.view(np.uint64), pb.view(np.uint64)
M = np.uint64(0x0F0F0F0F0F0F0F0F); K = np.uint64(0x1010101010101010); four = np.uint64(4)
lib = ctypes.CDLL(os.path.abspath("c/libbindswar.so")); P = ctypes.POINTER(ctypes.c_uint8)
for f in (lib.bind_p, lib.unbind_p):
    f.argtypes = [P, P, P, ctypes.c_size_t]
out = np.empty_like(pa)
arms = {
    "int64": (lambda: (a64 + b64) % 16, lambda: (a64 - b64) % 16),
    "uint8": (lambda: (a8 + b8) & 15, lambda: (a8 - b8) & 15),
    "swar_u64": (lambda: (((wa & M) + (wb & M)) & M) | ((((wa >> four) & M) + ((wb >> four) & M)) & M) << four,
                 lambda: (((wa & M) + K - (wb & M)) & M) | ((((wa >> four) & M) + K - ((wb >> four) & M)) & M) << four),
    "c_packed": (lambda: (lib.bind_p(pa.ctypes.data_as(P), pb.ctypes.data_as(P), out.ctypes.data_as(P), len(pa)), out)[1],
                 lambda: (lib.unbind_p(pa.ctypes.data_as(P), pb.ctypes.data_as(P), out.ctypes.data_as(P), len(pa)), out)[1]),
}
a64, b64 = a8.astype(np.int64), b8.astype(np.int64)
res = {}
for name, (fb, fu) in arms.items():
    rb = fb(); rb = np.asarray(rb); rb = unpack(rb.view(np.uint8)) if rb.dtype != np.int64 and len(rb.view(np.uint8)) == n // 2 else rb.astype(np.uint8)
    ok_b = bool(np.array_equal(rb, truth_b))
    ru = np.asarray(fu()); ru = unpack(ru.view(np.uint8)) if ru.dtype != np.int64 and len(ru.view(np.uint8)) == n // 2 else ru.astype(np.uint8)
    ok_u = bool(np.array_equal(ru, truth_u))
    best = 1e9
    for _ in range(5):
        t = time.perf_counter(); fb(); best = min(best, time.perf_counter() - t)
    res[name] = {"bit_identical_bind": ok_b, "bit_identical_unbind": ok_u, "ns_per_phase": best / n * 1e9}
    print(f"BIND-SWAR {name:9s} identical bind {ok_b} unbind {ok_u} | {best/n*1e9:.3f} ns/phase", flush=True)
x = lambda u, v: res[v]["ns_per_phase"] / res[u]["ns_per_phase"]  # noqa: E731
res["speedups"] = {"swar_vs_uint8": x("swar_u64", "uint8"), "swar_vs_int64": x("swar_u64", "int64"), "c_vs_swar": x("c_packed", "swar_u64")}
res["verdict"] = {"PASS": all(r["bit_identical_bind"] and r["bit_identical_unbind"] for k, r in res.items() if k in arms)
                  and res["speedups"]["swar_vs_uint8"] >= 4 and res["speedups"]["swar_vs_int64"] >= 16 and res["speedups"]["c_vs_swar"] >= 2}
print("BIND-SWAR-0:", {k: round(v, 2) for k, v in res["speedups"].items()}, res["verdict"], flush=True)
json.dump(res, open(os.path.expanduser("~/gpurun-out-0930/bindswar0.result.json"), "w"), indent=1)
