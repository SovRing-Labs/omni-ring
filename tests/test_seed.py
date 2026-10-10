"""SuperSeed Spec v1 conformance (RING-SEED plan step 1). The golden file pins the bytes every dialect must reproduce."""
import json
import os

import numpy as np
import pytest

from omni_ring import seed as S

GOLDEN = os.path.join(os.path.dirname(__file__), "golden", "seed_v1.json")


def test_reference_hash_values():
    # Published reference values: FNV-1a 64 and the splitmix64 generator's first output from state 0.
    assert S.fnv1a64(b"") == 0xCBF29CE484222325
    assert S.fnv1a64(b"a") == 0xAF63DC4C8601EC8C
    assert int(S.splitmix64(np.array([0], dtype=np.uint64))[0]) == 0xE220A8397B1DCDAF


def test_golden_conformance():
    assert list(S.check_golden(GOLDEN)) == []


def test_golden_detects_mutation(monkeypatch):
    monkeypatch.setattr(S, "_GOLD", np.uint64(0x9E3779B97F4A7C17))
    assert "codewords" in list(S.check_golden(GOLDEN))


def test_c_kernel_equals_numpy():
    lat = np.stack([S.codeword(f"k{i}") for i in range(64)])
    for i in range(4):
        q = S.codeword(f"k{i}")
        assert np.array_equal(S.sift_phase(q, lat, backend="numpy"), S.sift_phase(q, lat))


def test_self_score_and_orthogonality():
    lat = np.stack([S.codeword(f"o{i}") for i in range(32)])
    sc = S.sift_phase(lat[0], lat, backend="numpy")
    assert sc[0] == max(sc)
    other = sc[1:] / sc[0]
    assert np.abs(other).max() < 0.06


def test_encode_deterministic_and_similarity_preserving():
    p = S.Projection(64, 512, 0, "t")
    rng = np.random.default_rng(3)
    x = rng.standard_normal((50, 64)).astype(np.float32)
    a, b = S.encode(x, p), S.encode(x.copy(), p)
    assert np.array_equal(a, b)
    near = S.encode(x + 0.05 * rng.standard_normal(x.shape).astype(np.float32), p)
    far = S.encode(rng.standard_normal(x.shape).astype(np.float32), p)
    agree_near, agree_far = (a == near).mean(), (a == far).mean()
    assert agree_near > 0.7 and agree_far < 0.2


def test_sector16_matches_float_off_boundary():
    rng = np.random.default_rng(1)
    re, im = rng.integers(-2 ** 20, 2 ** 20, 200000), rng.integers(-2 ** 20, 2 ** 20, 200000)
    ref = np.round(np.angle(re + 1j * im) / (2 * np.pi / 16)).astype(int) % 16
    assert (S.sector16(re, im) == ref).mean() > 0.9999


def test_c_sector16_equals_numpy():
    rng = np.random.default_rng(7)
    re, im = rng.integers(-90000, 90000, 300000), rng.integers(-90000, 90000, 300000)
    re[:8], im[:8] = [0, 0, -5, 5, 0, -1, 1, 0], [0, 5, 0, 0, -5, -1, 1, 0]
    assert np.array_equal(S.sector16(re, im), S.sector16(re, im, backend="numpy"))


def test_sift_batch_equals_single():
    lat = np.stack([S.codeword(f"b{i}") for i in range(97)])
    qs = np.stack([S.codeword(f"q{i}") for i in range(13)])
    ref = np.stack([S.sift_phase(q, lat, backend="numpy") for q in qs])
    assert np.array_equal(S.sift_batch(qs, lat), ref)
