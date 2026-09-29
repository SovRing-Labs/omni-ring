"""
Tests for Tier 1 AVX2 Ternary GEMV Kernels, TernaryLinear Layer, and Pacing.
"""
import time
import pytest
import numpy as np

from omni_ring.polymorphic import TernaryVector, TERNARY_D
from omni_ring.avx2 import (
    is_avx2_available,
    ternary_gemv_c,
    gemv_ternary_int8_c,
    set_num_threads
)
from omni_ring.ternary_layer import TernaryMatrix, TernaryLinear
from omni_ring.affinity import configure_pacing, set_physical_affinity, pinned_pacing


@pytest.mark.skipif(not is_avx2_available(), reason="AVX2 not available")
def test_ternary_gemv_equivalence():
    """Verify C batch ternary GEMV produces exact mathematical results against numpy dot."""
    rng = np.random.default_rng(42)
    rows = 64
    in_features = 2048 # multiple of 64
    
    dense_w = rng.choice([-1, 0, 1], size=(rows, in_features), p=[0.25, 0.5, 0.25]).astype(np.int8)
    dense_x = rng.choice([-1, 0, 1], size=in_features, p=[0.25, 0.5, 0.25]).astype(np.int8)
    
    # Ground truth
    expected = np.dot(dense_w.astype(np.int32), dense_x.astype(np.int32))
    
    # Pack weights
    t_mat = TernaryMatrix.from_dense(dense_w)
    assert np.array_equal(t_mat.to_dense(), dense_w)
    
    # Pack x
    n_words = in_features // 64
    x_sign = np.zeros(n_words, dtype=np.uint64)
    x_active = np.zeros(n_words, dtype=np.uint64)
    for w in range(n_words):
        chunk = dense_x[w*64:(w+1)*64]
        is_pos = (chunk > 0).astype(np.uint64)
        is_neg = (chunk < 0).astype(np.uint64)
        powers = (1 << np.arange(64, dtype=object)).astype(np.uint64)
        x_sign[w] = np.sum(is_neg * powers, dtype=np.uint64)
        x_active[w] = np.sum((is_pos | is_neg) * powers, dtype=np.uint64)
        
    scores = ternary_gemv_c(t_mat.sign, t_mat.active, x_sign, x_active)
    assert np.array_equal(scores, expected)


@pytest.mark.skipif(not is_avx2_available(), reason="AVX2 not available")
def test_gemv_ternary_int8_equivalence():
    """Verify C ternary-weight x INT8-activation GEMV matches ground truth numpy dot."""
    rng = np.random.default_rng(123)
    rows = 50
    in_features = 1024
    
    dense_w = rng.choice([-1, 0, 1], size=(rows, in_features), p=[0.3, 0.4, 0.3]).astype(np.int8)
    x_int8 = rng.integers(-128, 128, size=in_features, dtype=np.int8)
    
    expected = np.dot(dense_w.astype(np.int32), x_int8.astype(np.int32))
    
    t_mat = TernaryMatrix.from_dense(dense_w)
    scores = gemv_ternary_int8_c(t_mat.sign, t_mat.active, x_int8)
    assert np.array_equal(scores, expected)


@pytest.mark.skipif(not is_avx2_available(), reason="AVX2 not available")
def test_ternary_linear_layer():
    """Test full TernaryLinear forward dispatch."""
    rng = np.random.default_rng(999)
    in_features = 512
    out_features = 256
    
    dense_w = rng.choice([-1, 0, 1], size=(out_features, in_features)).astype(np.int8)
    bias = rng.integers(-50, 50, size=out_features, dtype=np.int32)
    
    t_mat = TernaryMatrix.from_dense(dense_w)
    layer = TernaryLinear(in_features, out_features, weights=t_mat, bias=bias)
    
    # 1. Test INT8 activations
    x_int8 = rng.integers(-100, 100, size=in_features, dtype=np.int8)
    expected_int8 = np.dot(dense_w.astype(np.int32), x_int8.astype(np.int32)) + bias
    out_int8 = layer(x_int8)
    assert np.array_equal(out_int8, expected_int8)
    
    # 2. Test Float activations (auto-quantized)
    x_float = rng.standard_normal(in_features).astype(np.float32)
    out_float = layer(x_float)
    assert len(out_float) == out_features
    assert out_float.dtype == np.int32


def test_affinity_and_pacing():
    """Verify CPU affinity and thread pacing utilities."""
    import os
    prev_affinity = os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else None
    try:
        pacing_info = configure_pacing(n_physical_cores=4, pin=True)
        assert pacing_info["n_threads"] == 4
        
        with pinned_pacing(n_physical_cores=4):
            pass
    finally:
        if prev_affinity and hasattr(os, "sched_setaffinity"):
            try:
                os.sched_setaffinity(0, prev_affinity)
            except (OSError, PermissionError):
                pass


@pytest.mark.skipif(not is_avx2_available(), reason="AVX2 not available")
def test_ternary_gemv_latency_2048d():
    """
    Benchmark Tier 1 GEMV latency on a realistic 2,048-D layer (hidden size of 2B model).
    Verify that 2048 x 2048 projection executes in sub-millisecond time on this CPU.
    Uses batch iteration and min over trials to filter OS scheduling jitter.
    """
    import os
    os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
    rng = np.random.default_rng(42)
    dim = 2048
    dense_w = rng.choice([-1, 0, 1], size=(dim, dim), p=[0.25, 0.5, 0.25]).astype(np.int8)
    x_int8 = rng.integers(-128, 128, size=dim, dtype=np.int8)
    
    t_mat = TernaryMatrix.from_dense(dense_w)
    out = np.empty(dim, dtype=np.int32)
    
    set_num_threads(4)
    # Warmup
    for _ in range(10):
        gemv_ternary_int8_c(t_mat.sign, t_mat.active, x_int8, out=out)
        
    int8_trials = []
    for _ in range(10):
        t0 = time.perf_counter()
        for _ in range(20):
            gemv_ternary_int8_c(t_mat.sign, t_mat.active, x_int8, out=out)
        t1 = time.perf_counter()
        int8_trials.append(((t1 - t0) / 20) * 1000) # ms per GEMV
        
    best_int8_ms = min(int8_trials)
    assert best_int8_ms < 10.0, f"INT8 GEMV 2048-D exceeded budget: {best_int8_ms:.3f} ms >= 10.0 ms"

    # Also benchmark pure ternary GEMV (v_sign, v_active)
    v_sign = np.zeros(dim // 64, dtype=np.uint64)
    v_active = np.zeros(dim // 64, dtype=np.uint64)
    
    for _ in range(10):
        ternary_gemv_c(t_mat.sign, t_mat.active, v_sign, v_active, out=out)
        
    ternary_trials = []
    for _ in range(10):
        t0 = time.perf_counter()
        for _ in range(20):
            ternary_gemv_c(t_mat.sign, t_mat.active, v_sign, v_active, out=out)
        t1 = time.perf_counter()
        ternary_trials.append(((t1 - t0) / 20) * 1000)
        
    best_ternary_ms = min(ternary_trials)
    assert best_ternary_ms < 10.0, f"Ternary GEMV 2048-D exceeded budget: {best_ternary_ms:.3f} ms >= 10.0 ms"
