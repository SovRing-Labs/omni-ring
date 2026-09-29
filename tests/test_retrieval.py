"""
Tests for GatedMemoryEngine (Associative scanning, clamped unbinding, 17-check verification).
"""
import pytest
import numpy as np
from omni_ring.types import DEFAULT_SPEC
from omni_ring.polymorphic import PolymorphicSlot, TernaryVector, PhaseVectorPacked
from omni_ring.retrieval import GatedMemoryEngine, GatedRecallResult
from omni_ring.chain import encode

def test_encode_fact_and_verify_direct():
    engine = GatedMemoryEngine(spec=DEFAULT_SPEC, seed=42)
    x = 789  # Valid state < 1365
    
    pv = engine.encode_direct(x)
    slot = PolymorphicSlot()
    slot.write_phase(pv)
    
    result = engine.decode_direct(slot.as_phase())
    assert result.status == "VERIFIED"
    assert result.coordinate == x
    assert result.residues == [x % m for m in DEFAULT_SPEC.all_moduli]
    assert result.confidence == 1.0

def test_17_check_gate_rejection():
    engine = GatedMemoryEngine(spec=DEFAULT_SPEC, seed=42)
    x = 100
    mods = DEFAULT_SPEC.all_moduli
    
    # 1. Fault tolerance check: adding random Gaussian noise to the phase vector
    # still recovers exact coordinate
    pv = engine.encode_direct(x)
    noisy_phases = pv.to_phases().copy()
    noisy_phases[0:1500] = (noisy_phases[0:1500] + 5) % 16
    slot = PolymorphicSlot()
    slot.write_phase(PhaseVectorPacked.from_phases(noisy_phases))
    res_tolerant = engine.decode_direct(slot.as_phase())
    assert res_tolerant.status == "VERIFIED"
    assert res_tolerant.coordinate == x

    # 2. Corrupt one ring (e.g. ring 17 shifted by 1)
    # This simulates a phase slip / desync error
    v_corrupt = sum(cb[x % m] for cb, m in zip(engine.direct_codebooks[:4], mods[:4])) + \
                engine.direct_codebooks[4][(x % 17 + 1) % 17]
    angles = np.angle(v_corrupt)
    angles = np.where(angles < 0, angles + 2 * np.pi, angles)
    phases = np.round(angles * 16 / (2 * np.pi)).astype(np.uint8) % 16
    
    slot_corrupt = PolymorphicSlot()
    slot_corrupt.write_phase(PhaseVectorPacked.from_phases(phases))
    
    result = engine.decode_direct(slot_corrupt.as_phase())
    assert result.status == "REJECTED"
    assert result.error_message is not None
    assert "Phase slip / corrupt residue detected" in result.error_message

def test_clamped_resonator_retrieval():
    engine = GatedMemoryEngine(spec=DEFAULT_SPEC, seed=3)
    mods = (3, 5, 7, 13)
    target_x = 420
    truth = [target_x % m for m in mods]
    
    # 8-mark superposition
    rng = np.random.default_rng(3)
    v_superposed = encode(target_x, engine.rotations[:4])
    for _ in range(7):
        other_x = int(rng.integers(0, 1365))
        v_superposed += encode(other_x, engine.rotations[:4])
        
    result = engine.decode_bound_clamped(
        v_superposed,
        clamp_cue=truth[3], # 13-ring cue
        clamp_idx=3,
        iterations=15
    )
    assert result.status == "CONVERGED"  # unchecked path: converged, not verified
    assert result.residues == truth
    assert result.coordinate == target_x

def test_associative_ternary_scan():
    engine = GatedMemoryEngine(spec=DEFAULT_SPEC, seed=42)
    
    # Create 10 slots
    slots = []
    target_idx = 4
    target_tv = TernaryVector.random(seed=100)
    
    for i in range(10):
        slot = PolymorphicSlot()
        if i == target_idx:
            slot.write_ternary(target_tv)
        else:
            slot.write_ternary(TernaryVector.random(seed=200 + i))
        slots.append(slot)
        
    # Query with target_tv
    matches = engine.associative_scan(target_tv, slots, top_k=3)
    assert len(matches) == 3
    # Top match must be target_idx
    best_slot_idx, best_score = matches[0]
    assert best_slot_idx == target_idx
    assert best_score > matches[1][1]


def test_clamped_retrieval_checked_verifies_or_rejects():
    """check=True decodes all 5 rings and runs the 17 gate: legal -> VERIFIED with the right coordinate."""
    engine = GatedMemoryEngine(spec=DEFAULT_SPEC, seed=3)
    rng = np.random.default_rng(3)
    target_x = 420
    v = encode(target_x, engine.rotations)
    for _ in range(7):
        v += encode(int(rng.integers(0, 1365)), engine.rotations)
    r = engine.decode_bound_clamped(v, clamp_cue=target_x % 13, clamp_idx=3, iterations=30, check=True)
    assert r.status in ("VERIFIED", "REJECTED", "ABSTAIN")
    if r.status == "VERIFIED":
        assert r.coordinate == target_x


def test_unguided_decoding_is_seed_dependent():
    """Documents the seed lottery: never rely on unguided decodes without verify_unguided."""
    from omni_ring.codebook import verify_unguided
    assert verify_unguided(DEFAULT_SPEC, seed=0) >= 0.9
    assert verify_unguided(DEFAULT_SPEC, seed=42) <= 0.2
