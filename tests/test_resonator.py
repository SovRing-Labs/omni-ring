"""
Tests for omni_ring.resonator
"""
import numpy as np
from omni_ring.types import DEFAULT_SPEC
from omni_ring.codebook import generate_rotations, generate_codebooks
from omni_ring.chain import encode
from omni_ring.resonator import decode_soft

def test_resonator_clean_decoding():
    rotations = generate_rotations(DEFAULT_SPEC, seed=42)[:3]
    mods = (3, 5, 7)
    cbs = [np.exp(1j * np.outer(np.arange(m), r)) for m, r in zip(mods, rotations)]
    
    # Test random state
    x = 47
    v = encode(x, rotations)
    result = decode_soft(v, cbs, iterations=15)
    
    expected = [x % m for m in mods]
    assert result.residues == expected
    assert result.converged
    assert all(c > 0.8 for c in result.confidences)

def test_resonator_clamping_advantage():
    rng = np.random.default_rng(101)
    rotations = generate_rotations(DEFAULT_SPEC, seed=101)[:4]
    mods = (3, 5, 7, 13)
    cbs = [np.exp(1j * np.outer(np.arange(m), r)) for m, r in zip(mods, rotations)]
    
    # 8-mark superposition
    xs = rng.integers(0, 1365, 8)
    v = sum(encode(int(x), rotations) for x in xs)
    target = int(xs[0])
    truth = [target % m for m in mods]
    
    # Clamped at ring 13
    res_clamped = decode_soft(v, cbs, iterations=15, clamp=target % 13, clamp_idx=3)
    # The clamped ring should match exact target
    assert res_clamped.residues[3] == truth[3]
