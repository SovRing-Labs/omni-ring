"""AVX-512 kernels (OMNIKIT 2026-10-08): VNNI sift == AVX2 sift == numpy, bit for bit; VPOPCNTDQ Hamming / ternary == numpy.
Runs under pytest or plain python (the pod has no pytest): `python3 tests/test_avx512.py`. AVX-512 checks skip on CPUs without the flags."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402

from omni_ring import sift as S  # noqa: E402

SHAPES = [(1, 32), (3, 64), (7, 96), (100, 256), (513, 1024), (2000, 5120), (64, 8192), (37, 4160)]


class Skip(Exception):
    pass


def _need512():
    if not S.has_avx512():
        if "pytest" in sys.modules and __name__ != "__main__":
            sys.modules["pytest"].skip("no AVX-512 (flags or lib)")
        raise Skip("no AVX-512 (flags or lib)")


def _slot(rng, d, kind):
    if kind == "unit":
        return np.exp(1j * rng.uniform(0, 2 * np.pi, d))
    if kind == "bundle":
        return sum(np.exp(1j * rng.uniform(0, 2 * np.pi, d)) for _ in range(9))
    v = np.zeros(d, complex)          # saturated: max-magnitude int8 everywhere (worst case for overflow / saturation)
    v.real, v.imag = rng.choice([-1.0, 1.0], d), rng.choice([-1.0, 1.0], d)
    return v


def test_dispatch_is_consistent():
    assert S.cpu_has_avx512() or not S.has_avx512()
    be = S.backend_for(64)
    assert be == ("avx512" if S.has_avx512() else ("c" if S.has_kernel() else "numpy"))


def test_sift_avx512_equals_avx2_and_numpy():
    _need512()
    rng = np.random.default_rng(11)
    for n, d in SHAPES:
        lat = rng.integers(0, 16, (n, d), dtype=np.uint8)
        for kind in ("unit", "bundle", "sat"):
            v = _slot(rng, d, kind)
            a5 = S.scores(v, lat, "avx512")
            npy = S.scores(v, lat, "numpy")
            assert np.array_equal(a5, npy), (n, d, kind)
            if S.has_kernel() and d % 32 == 0:
                assert np.array_equal(a5, S.scores(v, lat, "c")), (n, d, kind)


def test_sift_batch_equals_single():
    _need512()
    rng = np.random.default_rng(5)
    lat = rng.integers(0, 16, (300, 1024), dtype=np.uint8)
    vs = [_slot(rng, 1024, k) for k in ("unit", "bundle", "sat", "unit")]
    b = S.scores_batch(vs, lat, "avx512")
    assert np.array_equal(b, np.stack([S.scores(v, lat, "numpy") for v in vs]))


def test_hamming_and_ternary_equal_numpy():
    _need512()
    rng = np.random.default_rng(3)
    for n, dims in [(1, 64), (5, 100), (257, 1024), (1000, 4096), (33, 10000)]:
        bits = rng.integers(0, 2, (n, dims)).astype(bool)
        lat, q = S.pack_bits(bits), S.pack_bits(rng.integers(0, 2, dims).astype(bool))
        assert np.array_equal(S.hamming_scores(q, lat, "avx512"), S.hamming_scores(q, lat, "numpy"))
        t = rng.integers(-1, 2, (n, dims))
        qt = rng.integers(-1, 2, dims)
        ls, la = S.pack_ternary(t)
        qs, qa = S.pack_ternary(qt)
        got = S.ternary_scores(qs, qa, ls, la, "avx512")
        assert np.array_equal(got, S.ternary_scores(qs, qa, ls, la, "numpy"))
        assert np.array_equal(got, (t * qt).sum(1).astype(np.int32))


def test_numpy_reference_ternary_is_true_dot():
    rng = np.random.default_rng(9)
    t, qt = rng.integers(-1, 2, (20, 300)), rng.integers(-1, 2, 300)
    ls, la = S.pack_ternary(t)
    qs, qa = S.pack_ternary(qt)
    assert np.array_equal(S.ternary_scores(qs, qa, ls, la, "numpy"), (t * qt).sum(1))
    b = rng.integers(0, 2, (20, 300)).astype(bool)
    qb = rng.integers(0, 2, 300).astype(bool)
    assert np.array_equal(S.hamming_scores(S.pack_bits(qb), S.pack_bits(b), "numpy"), (b != qb).sum(1))


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted((k, v) for k, v in globals().items() if k.startswith("test_") and callable(v)):
        try:
            fn()
            print(f"PASS {name}")
        except Skip as e:
            print(f"SKIP {name}: {e}")
        except Exception as e:  # noqa: BLE001
            fails += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    sys.exit(1 if fails else 0)
