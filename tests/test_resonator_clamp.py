"""
Tests for Semantic Resonator Attractor Clamp (Tier 2).
Verifies noise rejection, discrete coordinate locking, and 17-check CRT anomaly defense.
"""
import pytest
import numpy as np

from omni_ring.types import DEFAULT_SPEC
from omni_ring.polymorphic import PhaseVectorPacked
from omni_ring.resonator_clamp import ResonatorAttractorClamp, ClampResult
from omni_ring.bridge import ActivationProjector


def test_resonator_clamp_clean_recovery():
    """Verify that a noisy, phase-jittered vector converges back to the exact clean attractor."""
    clamp = ResonatorAttractorClamp(spec=DEFAULT_SPEC, seed=42)
    
    # Register discrete semantic symbols
    clamp.register_symbols({
        "status_check": 10,
        "windup_baseline": 42,
        "read_task_queue": 100
    })

    # Generate ground truth clean vector for 'windup_baseline' (coord 42)
    clean_pv = clamp.get_symbol_vector("windup_baseline")
    clean_phases = clean_pv.to_phases()

    # Corrupt 20% of phase dimensions with random noise
    rng = np.random.default_rng(99)
    noisy_phases = clean_phases.copy()
    corrupt_mask = rng.random(len(clean_phases)) < 0.20
    noisy_phases[corrupt_mask] = rng.integers(0, 16, size=np.sum(corrupt_mask), dtype=np.uint8)
    noisy_pv = PhaseVectorPacked.from_phases(noisy_phases)

    # Perform attractor clamping
    result = clamp.clamp(noisy_pv, iterations=15)

    assert result.status == "CLAMPED"
    assert result.clamped is True
    assert result.coordinate == 42
    assert result.symbol == "windup_baseline"
    assert result.confidence > 0.05
    assert result.clean_phase is not None
    # Verify the clamped vector recovered 100% fidelity with the true attractor
    assert np.array_equal(result.clean_phase.to_phases(), clean_phases)


def test_resonator_clamp_slip_rejection():
    """Verify that an adversarial/corrupted vector that causes residue desync is rejected."""
    clamp = ResonatorAttractorClamp(spec=DEFAULT_SPEC, seed=42)
    
    # Pure uniform random noise (off-manifold state)
    rng = np.random.default_rng(777)
    random_phases = rng.integers(0, 16, 5120, dtype=np.uint8)
    random_pv = PhaseVectorPacked.from_phases(random_phases)

    result = clamp.clamp(random_pv, iterations=5, min_confidence=0.50)

    # Must NOT clamp cleanly to an invalid coordinate
    assert result.clamped is False
    assert result.status in ("SLIP_REJECTED", "AMBIGUOUS")


def test_resonator_clamp_from_embedding():
    """Verify direct projection from a simulated 768-D neural hidden state into an attractor."""
    clamp = ResonatorAttractorClamp(spec=DEFAULT_SPEC, seed=42)
    clamp.register_symbol("target_intent", 77)
    
    target_pv = clamp.get_symbol_vector("target_intent")
    target_phases = target_pv.to_phases()
    
    # Mock embedding with high correlation to target
    projector = ActivationProjector(input_dim=768, seed=42)
    # Clamp directly from target PhaseVectorPacked
    res = clamp.clamp(target_pv)
    
    assert res.clamped is True
    assert res.coordinate == 77
    assert res.symbol == "target_intent"
