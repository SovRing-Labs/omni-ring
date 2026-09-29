import pytest
import numpy as np
import os
from omni_ring.polymorphic import PhaseVectorPacked
from omni_ring.query_bus import QueryBus, fractional_power_encoding

def test_query_bus_basic():
    bus = QueryBus(capacity=4, name="test_query_bus.bin", create=True)
    try:
        # Create a dummy phase vector
        phases = np.random.randint(0, 16, 5120, dtype=np.uint8)
        pv = PhaseVectorPacked.from_phases(phases)
        
        # Test Query Ring
        assert bus.send_query(pv, 42) is True
        recv = bus.recv_query()
        assert recv is not None
        recv_pv, recv_seq = recv
        assert recv_seq == 42
        assert np.array_equal(recv_pv.to_phases(), phases)
        assert bus.recv_query() is None
        
        # Test Response Ring
        assert bus.send_response(pv, 100) is True
        recv = bus.recv_response()
        assert recv is not None
        recv_pv, recv_seq = recv
        assert recv_seq == 100
        assert np.array_equal(recv_pv.to_phases(), phases)
        assert bus.recv_response() is None
    finally:
        bus.unlink()

def test_query_bus_capacity():
    bus = QueryBus(capacity=4, name="test_query_bus_cap.bin", create=True)
    try:
        pv = PhaseVectorPacked.from_phases(np.zeros(5120, dtype=np.uint8))
        
        # Ring holds capacity-1 elements
        assert bus.send_query(pv, 1) is True
        assert bus.send_query(pv, 2) is True
        assert bus.send_query(pv, 3) is True
        assert bus.send_query(pv, 4) is False # Full
        
        assert bus.recv_query()[1] == 1
        assert bus.send_query(pv, 4) is True # Now has space
        
        assert bus.recv_query()[1] == 2
        assert bus.recv_query()[1] == 3
        assert bus.recv_query()[1] == 4
        assert bus.recv_query() is None
    finally:
        bus.unlink()

def test_fractional_power_encoding():
    # Setup base vector with phases 0, 1, 2, ..., 15
    phases = np.zeros(5120, dtype=np.uint8)
    for i in range(16):
        phases[i] = i
        
    base_pv = PhaseVectorPacked.from_phases(phases)
    
    # Scale by 2.5
    fpe_pv = fractional_power_encoding(base_pv, 2.5)
    fpe_phases = fpe_pv.to_phases()
    
    for i in range(16):
        expected = int(round(i * 2.5)) % 16
        assert fpe_phases[i] == expected
        
    # Test continuity/float properties
    fpe_pv_small = fractional_power_encoding(base_pv, 0.1)
    fpe_phases_small = fpe_pv_small.to_phases()
    # 0.1 * 1 = 0.1 -> 0
    assert fpe_phases_small[1] == 0
    # 0.1 * 10 = 1.0 -> 1
    assert fpe_phases_small[10] == 1
