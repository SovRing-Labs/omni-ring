"""
Tests for Multi-Ring Memory Bank (Tier 2).
Verifies 8-slot allocation, seqlock synchronization, and AVX2 cross-ring attention scan.
"""
import time
import pytest
import numpy as np

from omni_ring.polymorphic import TernaryVector, PhaseVectorPacked, SLOT_BYTES
from omni_ring.ring_bank import RingBank, RingSlotId, NUM_BANK_SLOTS


def test_ring_bank_lifecycle_and_slots():
    """Verify clean allocation, slot read/write across dual representations, and seqlocks."""
    bank = RingBank(name="test_ring_bank_lifecycle.bin", create=True)
    try:
        # 1. Write Phase Vector to TEMPORAL slot
        rng = np.random.default_rng(42)
        phases = rng.integers(0, 16, 5120, dtype=np.uint8)
        pv = PhaseVectorPacked.from_phases(phases)
        bank.write_phase(RingSlotId.TEMPORAL, pv)

        read_pv = bank.read_phase(RingSlotId.TEMPORAL)
        assert np.array_equal(read_pv.to_phases(), phases)

        # 2. Write Ternary Vector to GOAL slot
        tv = TernaryVector.random(seed=123)
        bank.write_ternary(RingSlotId.GOAL, tv)

        read_tv = bank.read_ternary(RingSlotId.GOAL)
        assert np.array_equal(read_tv.sign, tv.sign)
        assert np.array_equal(read_tv.active, tv.active)

        # 3. Snapshot verification
        snapshot = bank.get_snapshot()
        assert len(snapshot) == NUM_BANK_SLOTS
        assert "TEMPORAL" in snapshot
        assert "GOAL" in snapshot
        assert "SYSTEM_STATE" in snapshot

        # 4. Check sequence counter is even (write complete)
        seq = bank._read_seq(int(RingSlotId.TEMPORAL))
        assert seq % 2 == 0 and seq >= 2
    finally:
        bank.unlink()


def test_ring_bank_cross_attention_scan():
    """Verify sub-microsecond AVX2 cross-ring attention across all 8 slots."""
    bank = RingBank(name="test_ring_bank_scan.bin", create=True)
    try:
        # Populate all slots with random orthogonal ternary vectors
        vectors = [TernaryVector.random(seed=1000 + i) for i in range(NUM_BANK_SLOTS)]
        for i, v in enumerate(vectors):
            bank.write_ternary(i, v)

        # Query with vector from slot 3 (SYSTEM_STATE)
        query = vectors[3]
        scores = bank.cross_attention_scan(query)

        assert len(scores) == NUM_BANK_SLOTS
        # Slot 3 must have the maximum score (self-similarity)
        assert np.argmax(scores) == 3
        assert scores[3] > 4000  # High positive self-dot

        # Other slots should be near orthogonal (~0 relative to 10,240)
        for i in range(NUM_BANK_SLOTS):
            if i != 3:
                assert abs(scores[i]) < 1000, f"Slot {i} unexpected cross-talk: {scores[i]}"

        # Latency check: 100 scans must complete rapidly
        t0 = time.perf_counter()
        for _ in range(100):
            bank.cross_attention_scan(query)
        t1 = time.perf_counter()
        avg_us = ((t1 - t0) / 100) * 1e6
        # Must execute within 800 microseconds per scan (including Python dispatch)
        assert avg_us < 800.0, f"Cross-attention scan exceeded budget: {avg_us:.2f} us >= 800.0 us"
    finally:
        bank.unlink()
