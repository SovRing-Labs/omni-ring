"""Bit-exact equality: the C gram-hash kernel (OS-RING-0 step 2) vs its numpy fallback vs store.py's own
_gram_phases (the function it ports)."""
import numpy as np
import pytest

from omni_ring import store
from omni_ring.osring import gramhash


def _hashes(grams):
    import hashlib
    return np.array([int.from_bytes(hashlib.blake2b(g.encode(), digest_size=4).digest(), "big") for g in grams],
                     dtype=np.uint32)


@pytest.mark.parametrize("n,d", [(1, 16), (7, 64), (257, 5120)])
def test_numpy_fallback_matches_store_reference(n, d):
    grams = [f"gram-{i}" for i in range(n)]
    want = store._gram_phases(grams, d)
    got = gramhash.gram_phases_py(_hashes(grams), d)
    assert np.array_equal(want, got)


@pytest.mark.skipif(not gramhash.is_c_kernel_available(), reason="libomniring_avx2.so not built with gramhash")
@pytest.mark.parametrize("n,d", [(1, 16), (7, 64), (257, 5120), (4000, 5120)])
def test_c_kernel_bit_exact_vs_numpy_fallback(n, d):
    rng = np.random.default_rng(0)
    h = rng.integers(0, 2**32, size=n, dtype=np.uint64).astype(np.uint32)
    c_out = gramhash.gram_phases(h, d, backend="c")
    py_out = gramhash.gram_phases(h, d, backend="numpy")
    assert np.array_equal(c_out, py_out)


@pytest.mark.skipif(not gramhash.is_c_kernel_available(), reason="libomniring_avx2.so not built with gramhash")
def test_c_kernel_matches_store_reference():
    grams = [f"word{i}" for i in range(50)] + ["ümlaut", ""]
    want = store._gram_phases(grams, store.seed.D)
    got = gramhash.gram_phases(_hashes(grams), store.seed.D, backend="c")
    assert np.array_equal(want, got)
