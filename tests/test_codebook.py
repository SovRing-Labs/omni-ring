"""
Tests for omni_ring.codebook
"""
import numpy as np
from omni_ring.types import DEFAULT_SPEC
from omni_ring.codebook import (
    generate_rotations,
    generate_codebooks,
    generate_quantized_codebook,
    generate_direct_codebooks,
)

def test_default_seed_is_unguided_safe():
    """The shared default seed must be a winning draw for unguided decode.

    Unguided decode of the rotation rings is a seed lottery at D = 5,120
    (2026-09-29): seeds 0/1/2 -> 15/15, seeds 3/21/42 -> 1/15. The generators in
    codebook.py used to default to seed=42 -- the worst measured case -- so any
    caller that omitted `seed=` silently got the bad codebook. Every live decode
    path is cued, so this was latent, not an outage, but the default was a trap.

    This test is the guard: if SAFE_SEED is ever repointed at a losing draw, or a
    generator stops inheriting it, this fails here rather than in a decode 15x
    down the line.
    """
    from omni_ring.codebook import SAFE_SEED, verify_unguided

    rate = verify_unguided(seed=SAFE_SEED)
    assert rate >= 0.9, (
        f"SAFE_SEED={SAFE_SEED} only decodes {rate:.2%} unguided; "
        "pick a seed from the winning set (0/1/2 -> 15/15)"
    )


def test_generators_inherit_safe_seed():
    """No generator may hardcode a losing default.

    Guards the four signatures in codebook.py that take `seed`, plus the
    GatedMemoryEngine constructor in retrieval.py, which builds both a direct
    codebook set and a rotation set and so must not default to a bad draw either.
    """
    import inspect

    from omni_ring import codebook, retrieval

    for name in (
        "generate_rotations",
        "generate_quantized_codebook",
        "generate_direct_codebooks",
        "verify_unguided",
    ):
        default = inspect.signature(getattr(codebook, name)).parameters["seed"].default
        assert default == codebook.SAFE_SEED, (
            f"{name} defaults to seed={default}, not SAFE_SEED={codebook.SAFE_SEED}"
        )

    engine_default = inspect.signature(retrieval.GatedMemoryEngine).parameters["seed"].default
    assert engine_default == codebook.SAFE_SEED, (
        f"GatedMemoryEngine defaults to seed={engine_default}, not SAFE_SEED={codebook.SAFE_SEED}"
    )


def test_generate_rotations():
    rotations = generate_rotations(DEFAULT_SPEC, seed=42)
    assert len(rotations) == len(DEFAULT_SPEC.all_moduli)
    for r, m in zip(rotations, DEFAULT_SPEC.all_moduli):
        assert r.shape == (DEFAULT_SPEC.dimension,)
        # Values should be multiples of 2*pi / m
        discrete_vals = np.round(r * m / (2 * np.pi)).astype(int)
        assert np.all(discrete_vals >= 0)
        assert np.all(discrete_vals < m)

def test_generate_codebooks():
    rotations = generate_rotations(DEFAULT_SPEC, seed=42)
    cbs = generate_codebooks(rotations, DEFAULT_SPEC)
    assert len(cbs) == len(DEFAULT_SPEC.all_moduli)
    for cb, m in zip(cbs, DEFAULT_SPEC.all_moduli):
        assert cb.shape == (m, DEFAULT_SPEC.dimension)
        # Unit modulus check
        magnitudes = np.abs(cb)
        assert np.allclose(magnitudes, 1.0)
        # First entry (x=0) should be 1 + 0j
        assert np.allclose(cb[0], 1.0 + 0j)

def test_generate_quantized_codebook():
    cb = generate_quantized_codebook(size=50, dimension=128, k_quantization=16, seed=123)
    assert cb.shape == (50, 128)
    assert np.allclose(np.abs(cb), 1.0)

def test_generate_direct_codebooks():
    cbs = generate_direct_codebooks(DEFAULT_SPEC, seed=42)
    assert len(cbs) == len(DEFAULT_SPEC.all_moduli)
    for cb, m in zip(cbs, DEFAULT_SPEC.all_moduli):
        assert cb.shape == (m, DEFAULT_SPEC.dimension)
        assert np.allclose(np.abs(cb), 1.0)
