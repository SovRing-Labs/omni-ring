"""
Tests for omni_ring.shm_bridge
"""
import os
import pytest
import numpy as np
from omni_ring.types import DEFAULT_SPEC
from omni_ring.shm_bridge import ShmBridge, MAGIC_BYTES

def test_shm_bridge_write_read():
    test_shm_name = "test_omniring_shm_bridge.bin"
    shm = ShmBridge(spec=DEFAULT_SPEC, name=test_shm_name)
    
    try:
        assert os.path.exists(shm.filepath)
        phases = np.linspace(0, 2 * np.pi, DEFAULT_SPEC.dimension, dtype=np.float32)
        residues = [2, 4, 6, 12, 16]
        confidence = 0.985
        
        seq = shm.write(phases, residues, confidence)
        assert seq == 1
        
        header, read_phases = shm.read()
        assert header["magic"] == MAGIC_BYTES
        assert header["counter"] == 1
        assert header["residues"] == residues
        assert abs(header["confidence"] - confidence) < 1e-5
        assert np.allclose(read_phases, phases)
        
        # Second write
        seq2 = shm.write(phases * 0.5, [1, 2, 3, 4, 5], 0.75)
        assert seq2 == 2
        
        header2, read_phases2 = shm.read()
        assert header2["counter"] == 2
        assert header2["residues"] == [1, 2, 3, 4, 5]
        assert np.allclose(read_phases2, phases * 0.5)
        
    finally:
        shm.unlink()

def test_shm_bridge_context_manager():
    test_shm_name = "test_omniring_cm.bin"
    with ShmBridge(spec=DEFAULT_SPEC, name=test_shm_name) as shm:
        phases = np.ones(DEFAULT_SPEC.dimension, dtype=np.float32)
        shm.write(phases, [0, 1, 2, 3, 4], 0.5)
        header, read_phases = shm.read()
        assert header["counter"] == 1
        assert np.allclose(read_phases, 1.0)
    
    # Clean up file
    if os.path.exists(shm.filepath):
        os.remove(shm.filepath)
