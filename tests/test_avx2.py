"""
Tests for AVX2 accelerated nibble-packed kernel.
"""
import time
import pytest
import numpy as np
from omni_ring.types import DEFAULT_SPEC
from omni_ring.avx2 import (
    is_avx2_available,
    pack_nibbles,
    unpack_nibbles,
    bind_avx2,
    unbind_avx2,
    similarity_avx2,
    batch_similarity_avx2
)

@pytest.mark.skipif(not is_avx2_available(), reason="AVX2 not available")
def test_pack_unpack_roundtrip():
    rng = np.random.default_rng(42)
    phases = rng.integers(0, 16, 5120, dtype=np.uint8)
    
    packed = pack_nibbles(phases)
    assert len(packed) == 2560
    
    unpacked = unpack_nibbles(packed)
    assert np.array_equal(unpacked, phases)

@pytest.mark.skipif(not is_avx2_available(), reason="AVX2 not available")
def test_avx2_bind_unbind():
    rng = np.random.default_rng(42)
    a_phases = rng.integers(0, 16, 5120, dtype=np.uint8)
    b_phases = rng.integers(0, 16, 5120, dtype=np.uint8)
    
    a_pack = pack_nibbles(a_phases)
    b_pack = pack_nibbles(b_phases)
    
    bound = bind_avx2(a_pack, b_pack)
    unbound = unbind_avx2(bound, b_pack)
    
    assert np.array_equal(unbound, a_pack)

@pytest.mark.skipif(not is_avx2_available(), reason="AVX2 not available")
def test_avx2_similarity_identity():
    rng = np.random.default_rng(42)
    a_phases = rng.integers(0, 16, 5120, dtype=np.uint8)
    a_pack = pack_nibbles(a_phases)
    
    sim = similarity_avx2(a_pack, a_pack)
    # Cosine(0) = 127 in lookup table, across 5120 dimensions
    assert sim == 5120 * 127

@pytest.mark.skipif(not is_avx2_available(), reason="AVX2 not available")
def test_avx2_batch_similarity():
    rng = np.random.default_rng(42)
    query = pack_nibbles(rng.integers(0, 16, 5120, dtype=np.uint8))
    matrix = np.ascontiguousarray([pack_nibbles(rng.integers(0, 16, 5120, dtype=np.uint8)) for _ in range(50)])
    
    scores = batch_similarity_avx2(query, matrix)
    assert len(scores) == 50
    for i in range(50):
        assert scores[i] == similarity_avx2(query, matrix[i])

@pytest.mark.skipif(not is_avx2_available(), reason="AVX2 not available")
def test_avx2_batch_similarity_latency():
    """
    Verify that scanning across 500 candidate vectors in the bus achieves < 1.25 microseconds per 5,120-D vector.
    Uses min over repeated measurement batches to filter OS scheduling jitter and frequency governor ramp-up.
    """
    rng = np.random.default_rng(42)
    query = pack_nibbles(rng.integers(0, 16, 5120, dtype=np.uint8))
    n_slots = 500
    matrix = np.ascontiguousarray([pack_nibbles(rng.integers(0, 16, 5120, dtype=np.uint8)) for _ in range(n_slots)])
    out = np.empty(n_slots, dtype=np.int32)

    # Warm-up governor and cache
    for _ in range(10):
        batch_similarity_avx2(query, matrix, out=out)
    
    trials = []
    for _ in range(15):
        t0 = time.perf_counter()
        for _ in range(20):
            batch_similarity_avx2(query, matrix, out=out)
        t1 = time.perf_counter()
        trials.append(((t1 - t0) / (20 * n_slots)) * 1e6)
    
    best_per_vector_us = min(trials)
    assert best_per_vector_us < 1.25, f"AVX2 per-vector latency exceeded budget: {best_per_vector_us:.3f} us >= 1.25 us"
