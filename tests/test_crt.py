"""
Tests for omni_ring.crt and omni_ring.check
"""
import pytest
import numpy as np
from omni_ring.types import DEFAULT_SPEC
from omni_ring.codebook import generate_direct_codebooks
from omni_ring.crt import modular_inverse, solve_crt, direct_probe
from omni_ring.check import check_anomaly, ResidueDesyncError, run_full_error_detection_test

def test_modular_inverse():
    assert (3 * modular_inverse(3, 7)) % 7 == 1
    assert (13 * modular_inverse(13, 17)) % 17 == 1

def test_solve_crt():
    moduli = (3, 5, 7)
    x = 42
    residues = [x % m for m in moduli]
    assert solve_crt(residues, moduli) == x

    moduli_4 = (3, 5, 7, 13)
    x2 = 1234
    residues_4 = [x2 % m for m in moduli_4]
    assert solve_crt(residues_4, moduli_4) == x2

def test_direct_probe():
    moduli = DEFAULT_SPEC.all_moduli
    cbs = generate_direct_codebooks(DEFAULT_SPEC, seed=77)
    x = 9876
    v = sum(cb[x % m] for cb, m in zip(cbs, moduli))
    
    residues, solved_x = direct_probe(v, cbs, moduli)
    assert residues == [x % m for m in moduli]
    assert solved_x == x

def test_check_ring_integrity():
    # Valid value within [0, 1365)
    valid_res = [500 % m for m in DEFAULT_SPEC.all_moduli]
    assert check_anomaly(valid_res, DEFAULT_SPEC) == 500

    # Corrupt one residue
    corrupted_res = list(valid_res)
    corrupted_res[0] = (corrupted_res[0] + 1) % DEFAULT_SPEC.all_moduli[0]
    
    with pytest.raises(ResidueDesyncError):
        check_anomaly(corrupted_res, DEFAULT_SPEC)

def test_full_error_detection_comprehensive():
    missed, total = run_full_error_detection_test(DEFAULT_SPEC)
    assert total == 54600
    assert missed == 0
