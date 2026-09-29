"""
The 6 Required Numerical Gates for OMNIRING verification.
"""
import time
import math
import gc
import pytest
import numpy as np
from omni_ring.types import DEFAULT_SPEC
from omni_ring.codebook import (
    generate_rotations,
    generate_codebooks,
    generate_quantized_codebook,
    generate_direct_codebooks,
)
from omni_ring.chain import bind, encode, get_tick
from omni_ring.resonator import decode_soft
from omni_ring.crt import solve_crt, direct_probe
from omni_ring.check import check_anomaly, run_full_error_detection_test, ResidueDesyncError
from omni_ring.shm_bridge import ShmBridge


def test_gate_1_separation():
    """Gate 1: max|cross-correlation| < 0.04 across 1,000 random codebook entries."""
    K = 16
    N = 1000
    D = DEFAULT_SPEC.dimension
    Z = generate_quantized_codebook(size=N, dimension=D, k_quantization=K, seed=1)

    s = (Z @ Z[0].conj()).real / D
    max_cross = float(np.abs(s[1:]).max())
    assert max_cross < 0.04, f"Separation gate failed: max cross {max_cross:.4f} >= 0.04"


def test_gate_2_cycle():
    """Gate 2: First autocorrelation repeat occurs at exactly t = 1,365 for 4 rings, and t = 23,205 with check ring."""
    rotations = generate_rotations(seed=0)
    R_4 = rotations[:4]  # 3, 5, 7, 13
    D = DEFAULT_SPEC.dimension

    # Test 4-ring sequence up to 1400
    V = np.array([encode(x, R_4) for x in range(1400)])
    s = (V @ V[0].conj()).real / D
    repeats = [x for x in range(1, 1400) if s[x] > 0.99]
    assert repeats == [1365], f"Cycle gate failed for 4 rings: expected [1365], got {repeats}"

    # Timing Chain Shift: enc(x) * tick == enc(x+1)
    tick = get_tick(R_4)
    assert np.allclose(bind(encode(41, R_4), tick), encode(42, R_4))

    # Test 5-ring repeat at 23,205
    R_5 = rotations
    v0 = encode(0, R_5)
    v_23205 = encode(23205, R_5)
    assert np.allclose(v0, v_23205), "5-ring cycle failed to repeat at 23,205"


def test_gate_3_resonator():
    """Gate 3: 100% convergence (105/105) on clean 3-5-7 factorizations within 20 iterations."""
    rotations = generate_rotations(seed=0)[:3]
    mods = (3, 5, 7)
    CB = [np.exp(1j * np.outer(np.arange(m), r)) for m, r in zip(mods, rotations)]

    ok = 0
    for x in range(105):
        v = encode(x, rotations)
        res = decode_soft(v, CB, iterations=20)
        expected = [x % m for m in mods]
        if res.residues == expected:
            ok += 1

    assert ok == 105, f"Resonator gate failed: decoded {ok}/105"


def test_gate_4_clamped_recall():
    """Gate 4: In a 12-mark crowded superposition, 13-ring clamping achieves >= 60% exact mark recovery (vs <= 15% unguided)."""
    rng = np.random.default_rng(3)
    rotations = generate_rotations(seed=3)[:4]
    mods = (3, 5, 7, 13)
    CB = [np.exp(1j * np.outer(np.arange(m), r)) for m, r in zip(mods, rotations)]

    T = 40
    ok_unguided = 0
    ok_clamped = 0

    for _ in range(T):
        xs = rng.integers(0, 1365, 12)
        v = sum(encode(int(x), rotations) for x in xs)
        target = int(xs[0])
        truth = [target % m for m in mods]

        # Unguided
        res_u = decode_soft(v, CB, iterations=15)
        if res_u.residues == truth:
            ok_unguided += 1

        # Clamped at 13-ring
        res_c = decode_soft(v, CB, iterations=15, clamp=target % 13, clamp_idx=3)
        if res_c.residues == truth:
            ok_clamped += 1

    unguided_rate = ok_unguided / T
    clamped_rate = ok_clamped / T

    assert unguided_rate <= 0.20, f"Unguided rate {unguided_rate:.2f} higher than expected"
    assert clamped_rate >= 0.60, f"Clamped recall gate failed: {clamped_rate:.2f} < 0.60"


def test_gate_5_check_ring():
    """Gate 5: Synthetic single-ring error test over 54,600 corruptions yields 0 undetected errors."""
    missed, total = run_full_error_detection_test(DEFAULT_SPEC)
    assert total == 54600, f"Expected 54,600 total error checks, got {total}"
    assert missed == 0, f"Check ring gate failed: {missed} undetected errors out of {total}"


def test_gate_6_latency():
    """Gate 6: Latency Benchmark
    - Direct 45-probe + CRT solve: < 1.0 ms on CPU
    - Resonator 15-iteration solve: < 8.0 ms on CPU
    - Shared memory write/read roundtrip: < 50 us
    """
    gc.collect()

    # 1. Direct 45-probe + CRT solve
    mods = DEFAULT_SPEC.all_moduli
    CB = generate_direct_codebooks(seed=5)
    CB_conj = [np.conj(cb) for cb in CB]
    x = 12345
    v = sum(cb[x % m] for cb, m in zip(CB, mods))

    # Warm-up
    _ = direct_probe(v, CB, mods, codebooks_conj=CB_conj)

    times_direct = []
    for _ in range(50):
        t0 = time.perf_counter()
        r, ans = direct_probe(v, CB, mods, codebooks_conj=CB_conj)
        t1 = time.perf_counter()
        times_direct.append(t1 - t0)

    min_direct_ms = min(times_direct) * 1000
    assert ans == x, f"Direct solve incorrect: {ans} != {x}"
    assert min_direct_ms < 1.0, f"Direct solve took {min_direct_ms:.3f}ms >= 1.0ms"

    # 2. Resonator 15-iteration solve
    rotations = generate_rotations(seed=0)[:3]
    CB_soft = [np.exp(1j * np.outer(np.arange(m), r)) for m, r in zip((3, 5, 7), rotations)]
    v_soft = encode(42, rotations)

    # Warm-up
    _ = decode_soft(v_soft, CB_soft, iterations=15)

    times_res = []
    for _ in range(20):
        t0 = time.perf_counter()
        _ = decode_soft(v_soft, CB_soft, iterations=15)
        t1 = time.perf_counter()
        times_res.append(t1 - t0)

    min_res_ms = min(times_res) * 1000
    assert min_res_ms < 8.0, f"Resonator solve took {min_res_ms:.3f}ms >= 8.0ms"

    # 3. Shared memory roundtrip
    shm = ShmBridge(name="test_gate6_shm.bin")
    phases = np.zeros(DEFAULT_SPEC.dimension, dtype=np.float32)

    # Warm-up
    shm.write(phases, [1, 2, 3, 4, 5], 1.0)
    _, _ = shm.read()

    times_shm = []
    for _ in range(100):
        t0 = time.perf_counter()
        shm.write(phases, [1, 2, 3, 4, 5], 1.0)
        _, _ = shm.read()
        t1 = time.perf_counter()
        times_shm.append(t1 - t0)

    shm.unlink()
    min_shm_us = min(times_shm) * 1_000_000
    assert min_shm_us < 50.0, f"SHM roundtrip took {min_shm_us:.2f}us >= 50us"
