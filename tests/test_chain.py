"""
Tests for omni_ring.chain
"""
import numpy as np
from omni_ring.types import DEFAULT_SPEC
from omni_ring.codebook import generate_rotations
from omni_ring.chain import bind, unbind, encode, get_tick, advance

def test_bind_unbind_inverse():
    rotations = generate_rotations(DEFAULT_SPEC, seed=42)
    v1 = encode(10, rotations)
    v2 = encode(25, rotations)

    bound = bind(v1, v2)
    unbound = unbind(bound, v2)

    assert np.allclose(unbound, v1)

def test_tick_operator():
    rotations = generate_rotations(DEFAULT_SPEC, seed=42)
    tick = get_tick(rotations)
    
    for x in [0, 1, 5, 42, 100]:
        vx = encode(x, rotations)
        vx_plus_1 = encode(x + 1, rotations)
        stepped = bind(vx, tick)
        assert np.allclose(stepped, vx_plus_1)

def test_advance_multi_step():
    rotations = generate_rotations(DEFAULT_SPEC, seed=42)
    tick = get_tick(rotations)
    v0 = encode(15, rotations)
    
    v_adv = advance(v0, tick, steps=7)
    v_target = encode(22, rotations)
    assert np.allclose(v_adv, v_target)
