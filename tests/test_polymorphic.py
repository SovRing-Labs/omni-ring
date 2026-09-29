"""
Tests for Two-Faced Polymorphic Bus and Transcoders.
"""
import os
import pytest
import numpy as np
from omni_ring.polymorphic import (
    TernaryVector,
    PhaseVectorPacked,
    PolymorphicSlot,
    PolymorphicBus,
    SLOT_BYTES,
    TERNARY_D,
    PHASE_D,
    TERNARY_TO_PHASE_LUT,
    transcode_ternary_to_phase_python,
)
from omni_ring.avx2 import is_avx2_available, transcode_ternary_to_phase_c


def _unpack_phase(packed: np.ndarray, k: int) -> int:
    """Read phase index k out of a 2-nibbles-per-byte buffer."""
    byte = int(packed[k // 2])
    return (byte >> 4) & 0x0F if k % 2 else byte & 0x0F

def test_ternary_vector_dense_roundtrip():
    rng = np.random.default_rng(42)
    dense = rng.choice([-1, 0, 1], size=TERNARY_D).astype(np.int8)
    
    tv = TernaryVector.from_dense(dense)
    recovered = tv.to_dense()
    assert np.array_equal(dense, recovered)
    
    # Check serialization
    raw = tv.to_bytes()
    assert len(raw) == SLOT_BYTES
    tv2 = TernaryVector.from_bytes(raw)
    assert np.array_equal(tv.sign, tv2.sign)
    assert np.array_equal(tv.active, tv2.active)

def test_ternary_vector_dot_and_bind():
    rng = np.random.default_rng(123)
    dense_a = rng.choice([-1, 0, 1], size=TERNARY_D).astype(np.int8)
    dense_b = rng.choice([-1, 0, 1], size=TERNARY_D).astype(np.int8)
    
    t_a = TernaryVector.from_dense(dense_a)
    t_b = TernaryVector.from_dense(dense_b)
    
    # Self-dot equals active count
    expected_self_dot = int(np.sum(dense_a != 0))
    assert t_a.dot(t_a) == expected_self_dot
    
    # Cross dot equals sum of elementwise products
    expected_cross_dot = int(np.sum(dense_a.astype(int) * dense_b.astype(int)))
    assert t_a.dot(t_b) == expected_cross_dot
    
    # Bind
    bound = t_a.bind(t_b)
    expected_bound = dense_a * dense_b
    assert np.array_equal(bound.to_dense(), expected_bound)

def test_phase_vector_packed():
    rng = np.random.default_rng(42)
    phases = rng.integers(0, 16, PHASE_D, dtype=np.uint8)
    
    pv = PhaseVectorPacked.from_phases(phases)
    assert len(pv.to_bytes()) == SLOT_BYTES
    
    unpacked = pv.to_phases()
    assert np.array_equal(unpacked, phases)
    
    # Self similarity
    if is_avx2_available():
        assert pv.similarity(pv) == PHASE_D * 127
        
        # Bind / unbind
        pv2 = PhaseVectorPacked.from_phases(rng.integers(0, 16, PHASE_D, dtype=np.uint8))
        bound = pv.bind(pv2)
        unbound = bound.unbind(pv2)
        assert np.array_equal(unbound.to_phases(), pv.to_phases())

def test_transcode_reversibility():
    """
    Round-trip all 9 ternary pairs, including the neutral (0, 0) block.

    The 8 active pairs round-trip with 100% fidelity. (0, 0) shares phase 0 with the
    active pair (+1, 0), so it round-trips as (+1, 0): 9 pairs cannot map injectively
    into the 8-point even-phase constellation, and OSEAM-D1-TRANSCODE resolved (0, 0)
    in favour of the null vector (a null ternary block IS a null phase block) rather
    than in favour of an injective round trip. That asymmetry is asserted here, not
    papered over.
    """
    pairs = [(1, 0), (1, 1), (0, 1), (-1, 1),
             (-1, 0), (-1, -1), (0, -1), (1, -1), (0, 0)]

    dense = np.zeros(TERNARY_D, dtype=np.int8)
    for i in range(PHASE_D):
        pair = pairs[i % len(pairs)]
        dense[2 * i] = pair[0]
        dense[2 * i + 1] = pair[1]

    assert int(np.sum((dense[0::2] == 0) & (dense[1::2] == 0))) > 0, "no (0,0) blocks present"

    tv = TernaryVector.from_dense(dense)
    pv = tv.to_phase()
    dense_recovered = pv.to_ternary().to_dense()

    # The 8 active pairs survive bit-exactly; (0, 0) comes back as its defined alias.
    expected = dense.astype(np.int8).copy()
    for i in range(PHASE_D):
        if int(dense[2 * i]) == 0 and int(dense[2 * i + 1]) == 0:
            expected[2 * i] = 1
            expected[2 * i + 1] = 0
    assert np.array_equal(expected, dense_recovered)

    # With no (0, 0) blocks at all, the round trip is exact and lossless.
    active_only = [p for p in pairs if p != (0, 0)]
    dense_active = np.zeros(TERNARY_D, dtype=np.int8)
    for i in range(PHASE_D):
        pair = active_only[i % len(active_only)]
        dense_active[2 * i] = pair[0]
        dense_active[2 * i + 1] = pair[1]
    assert np.array_equal(
        TernaryVector.from_dense(dense_active).to_phase().to_ternary().to_dense(),
        dense_active,
    )


def test_ternary_to_phase_lut_is_total():
    """
    Every one of the 9 ternary pairs has a defined phase, so no path can ever
    fabricate one. The 8 active pairs occupy the 8 even phases bijectively; (0, 0)
    is the additive identity. (OSEAM-D1-TRANSCODE)
    """
    all_pairs = {(t0, t1) for t0 in (-1, 0, 1) for t1 in (-1, 0, 1)}
    assert set(TERNARY_TO_PHASE_LUT) == all_pairs
    assert all(0 <= v <= 15 for v in TERNARY_TO_PHASE_LUT.values())

    active = {p: v for p, v in TERNARY_TO_PHASE_LUT.items() if p != (0, 0)}
    assert len(active) == 8
    assert all(v % 2 == 0 for v in active.values()), "active pairs must use even phases"
    assert len(set(active.values())) == 8, "active pairs must be bijective onto even phases"
    assert TERNARY_TO_PHASE_LUT[(0, 0)] == 0


def test_all_zero_ternary_transcodes_to_all_zero_phases():
    """
    (a) A null ternary vector must transcode to a null phase vector on BOTH paths.

    Regression guard: the C kernel used to emit 4,800 of 5,120 non-zero phases for
    the null vector, and the Python fallback emitted a position-dependent
    `(k * 5 + 7) & 0x0F`. (OSEAM-D1-TRANSCODE)
    """
    tv = TernaryVector()
    assert not np.any(tv.sign) and not np.any(tv.active)

    py_packed = transcode_ternary_to_phase_python(tv.sign, tv.active)
    assert len(py_packed) == SLOT_BYTES
    assert np.count_nonzero(py_packed) == 0

    if is_avx2_available():
        c_packed = transcode_ternary_to_phase_c(tv.sign, tv.active)
        assert np.count_nonzero(c_packed) == 0
        assert np.array_equal(c_packed, py_packed)

    # The public API agrees, on whichever path this host takes.
    phases = tv.to_phase().to_phases()
    assert len(phases) == PHASE_D
    assert np.count_nonzero(phases) == 0

    # A single (0,0) block in an otherwise all-active vector is neutral, and the
    # phase it yields is 0 -- independent of where the block sits.
    for position in (0, PHASE_D // 2, PHASE_D - 1):
        dense = np.ones(TERNARY_D, dtype=np.int8)
        dense[2 * position] = 0
        dense[2 * position + 1] = 0
        phases = TernaryVector.from_dense(dense).to_phase().to_phases()
        assert phases[position] == 0, position


def test_c_and_python_transcode_paths_agree_on_sparse_vector():
    """
    (b) A random ternary vector with ~38% zeros must transcode identically on the C
    kernel and the Python fallback. Before OSEAM-D1-TRANSCODE the two paths
    disagreed on every (0, 0) block.
    """
    rng = np.random.default_rng(20260929)
    dense = rng.choice([-1, 0, 1], size=TERNARY_D, p=[0.31, 0.38, 0.31]).astype(np.int8)

    zero_frac = float(np.mean(dense == 0))
    assert 0.36 <= zero_frac <= 0.40, zero_frac
    assert int(np.sum((dense[0::2] == 0) & (dense[1::2] == 0))) > 0, "no (0,0) blocks"

    tv = TernaryVector.from_dense(dense)
    py_packed = transcode_ternary_to_phase_python(tv.sign, tv.active)
    api_packed = tv.to_phase().data

    if is_avx2_available():
        c_packed = transcode_ternary_to_phase_c(tv.sign, tv.active)
        assert np.array_equal(c_packed, py_packed)
        assert np.array_equal(api_packed, c_packed)
    else:
        assert np.array_equal(api_packed, py_packed)

    # The transcode is a pure function of the pair, not of history or position.
    assert np.array_equal(transcode_ternary_to_phase_python(tv.sign, tv.active), py_packed)

    # Phase 0 is emitted by exactly the two pairs that map to it: the neutral (0,0)
    # and the active (+1,0). No other block collapses onto 0.
    got = np.array([_unpack_phase(api_packed, k) for k in range(PHASE_D)], dtype=np.int8)
    neutral = (dense[0::2] == 0) & (dense[1::2] == 0)
    plus_one_zero = (dense[0::2] == 1) & (dense[1::2] == 0)
    assert np.array_equal(got == 0, neutral | plus_one_zero)
    assert int(np.count_nonzero(got == 0)) == int(np.sum(neutral | plus_one_zero))

def test_polymorphic_slot():
    slot = PolymorphicSlot()
    assert len(slot.raw_bytes) == SLOT_BYTES
    
    tv = TernaryVector.random(seed=42)
    slot.write_ternary(tv)
    
    tv_read = slot.as_ternary()
    assert np.array_equal(tv.sign, tv_read.sign)
    assert np.array_equal(tv.active, tv_read.active)
    
    # In-place transcode to phase
    pv = slot.transcode_to_phase_in_place()
    assert isinstance(pv, PhaseVectorPacked)
    
    # Read as phase
    pv_read = slot.as_phase()
    assert np.array_equal(pv.data, pv_read.data)

def test_polymorphic_bus_ipc():
    bus_path = "/dev/shm/test_vsa_poly_bus"
    seq_path = "/dev/shm/test_vsa_poly_seq"
    
    try:
        with PolymorphicBus(num_slots=10, bus_path=bus_path, seq_path=seq_path, create=True) as bus:
            tv = TernaryVector.random(seed=99)
            bus.write_ternary(3, tv)
            
            # Read as ternary
            read_tv = bus.read_ternary(3)
            assert np.array_equal(tv.sign, read_tv.sign)
            assert np.array_equal(tv.active, read_tv.active)
            
            # Read as polymorphic slot and view as phase
            slot = bus.read_slot(3)
            pv_from_slot = slot.as_phase()
            assert len(pv_from_slot.data) == SLOT_BYTES
            
            # Write phase directly to slot 7
            rng = np.random.default_rng(101)
            pv_write = PhaseVectorPacked.from_phases(rng.integers(0, 16, PHASE_D, dtype=np.uint8))
            bus.write_phase(7, pv_write)
            
            read_pv = bus.read_phase(7)
            assert np.array_equal(pv_write.data, read_pv.data)
            
    finally:
        for p in (bus_path, seq_path):
            if os.path.exists(p):
                os.remove(p)
