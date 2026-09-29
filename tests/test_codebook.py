"""
Tests for omni_ring.codebook
"""
import numpy as np
from omni_ring.types import DEFAULT_SPEC
from omni_ring.codebook import (
    generate_rotations,
    generate_codebooks,
    generate_quantized_codebook,
    generate_direct_codebooks,
)

def test_generate_rotations():
    rotations = generate_rotations(DEFAULT_SPEC, seed=42)
    assert len(rotations) == len(DEFAULT_SPEC.all_moduli)
    for r, m in zip(rotations, DEFAULT_SPEC.all_moduli):
        assert r.shape == (DEFAULT_SPEC.dimension,)
        # Values should be multiples of 2*pi / m
        discrete_vals = np.round(r * m / (2 * np.pi)).astype(int)
        assert np.all(discrete_vals >= 0)
        assert np.all(discrete_vals < m)

def test_generate_codebooks():
    rotations = generate_rotations(DEFAULT_SPEC, seed=42)
    cbs = generate_codebooks(rotations, DEFAULT_SPEC)
    assert len(cbs) == len(DEFAULT_SPEC.all_moduli)
    for cb, m in zip(cbs, DEFAULT_SPEC.all_moduli):
        assert cb.shape == (m, DEFAULT_SPEC.dimension)
        # Unit modulus check
        magnitudes = np.abs(cb)
        assert np.allclose(magnitudes, 1.0)
        # First entry (x=0) should be 1 + 0j
        assert np.allclose(cb[0], 1.0 + 0j)

def test_generate_quantized_codebook():
    cb = generate_quantized_codebook(size=50, dimension=128, k_quantization=16, seed=123)
    assert cb.shape == (50, 128)
    assert np.allclose(np.abs(cb), 1.0)

def test_generate_direct_codebooks():
    cbs = generate_direct_codebooks(DEFAULT_SPEC, seed=42)
    assert len(cbs) == len(DEFAULT_SPEC.all_moduli)
    for cb, m in zip(cbs, DEFAULT_SPEC.all_moduli):
        assert cb.shape == (m, DEFAULT_SPEC.dimension)
        assert np.allclose(np.abs(cb), 1.0)
