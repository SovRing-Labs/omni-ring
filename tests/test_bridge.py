"""
Tests for ActivationProjector (Bridge from neural model embeddings to OMNIRING bus).
"""
import numpy as np
import pytest
from omni_ring.bridge import ActivationProjector
from omni_ring.polymorphic import TERNARY_D, PHASE_D

def test_project_to_ternary():
    proj = ActivationProjector(input_dim=768, seed=42)
    embedding = np.random.randn(768).astype(np.float32)
    
    tv = proj.project_to_ternary(embedding)
    dense = tv.to_dense()
    
    assert len(dense) == TERNARY_D
    # Values must be only -1, 0, +1
    unique_vals = set(np.unique(dense))
    assert unique_vals.issubset({-1, 0, 1})
    
    # Check sparsity approximately 50%
    zero_ratio = np.mean(dense == 0)
    assert 0.40 <= zero_ratio <= 0.60

def test_project_to_phase():
    proj = ActivationProjector(input_dim=1024, seed=10)
    embedding = np.random.randn(1024).astype(np.float32)
    
    pv = proj.project_to_phase(embedding)
    phases = pv.to_phases()
    
    assert len(phases) == PHASE_D
    assert np.all(phases >= 0) and np.all(phases < 16)

def test_direct_ternary_adaptation():
    proj = ActivationProjector(input_dim=512)
    native_ternary = np.random.choice([-1, 0, 1], size=2048).astype(np.int8)
    
    tv = proj.direct_ternary_to_bus(native_ternary)
    dense = tv.to_dense()
    
    assert len(dense) == TERNARY_D
    assert np.array_equal(dense[:2048], native_ternary)
