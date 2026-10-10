"""SuperSeed v1.1 (TORUS-TEL step 7): packed lattice is storage only (bit-identical scores); cascade is exact whenever the
true top-k survive the coarse pass. Golden: tests/golden/seed_v1_1.json (packed bytes of the v1 golden codewords)."""
import json, os
import numpy as np
from omni_ring import seed as S
from omni_ring.heads import RingHead

GOLD = os.path.join(os.path.dirname(__file__), "golden", "seed_v1_1.json")


def _cw():
    return np.stack([S.codeword(f"golden-{i}") for i in range(20)])


def test_pack_roundtrip_and_layout():
    lat = _cw()
    pk = S.pack4(lat)
    assert pk.shape == (20, S.D // 2) and np.array_equal(S.unpack4(pk), lat)
    # block-64 split: byte j of block b = phase[64b+j] | phase[64b+32+j] << 4
    assert pk[3, 32 * 5 + 7] == (lat[3, 64 * 5 + 7] | lat[3, 64 * 5 + 32 + 7] << 4)


def test_packed_scores_identical():
    rng = np.random.default_rng(3)
    lat = rng.integers(0, 16, (257, S.D), dtype=np.uint8); q = rng.integers(0, 16, (7, S.D), dtype=np.uint8)
    pk = S.pack4(lat)
    assert np.array_equal(S.sift_batch_p4(q, pk), S.sift_batch(q, lat))
    assert np.array_equal(S.sift_batch_p4(q, pk, 1280), S.sift_batch(q[:, :1280], np.ascontiguousarray(lat[:, :1280])))
    rows = np.array([256, 0, 99, 3])
    assert np.array_equal(S.sift_rows_p4(q[2], pk, rows), S.sift_batch(q[2:3], lat)[0, rows])
    assert np.array_equal(S.sift_batch_p4(q, pk, 1280), S.sift_batch(q[:, :1280], S.unpack4(pk)[:, :1280], backend="numpy"))


def test_cascade_exact_when_candidates_cover_all():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(300, 64)).astype(np.float32); y = rng.choice(["a", "b", "c"], 300)
    h = RingHead("t", 64); h.fit_centre(X); h.write(X, y)
    q = S.encode(rng.normal(size=(40, 64)).astype(np.float32), h.proj, h.centre)
    full = h.read_marks(q)
    cas = h.read_marks(q, cascade=(640, len(h.marks) - 1))   # nearly every mark rescored -> must equal full
    assert full[0] == cas[0] and np.array_equal(full[1], cas[1]) and np.allclose(full[2], cas[2], atol=1e-12)


def test_golden_v1_1():
    pk = S.pack4(_cw())
    got = {"spec": "seedv1.1 packed block-64 split", "packed_codewords": S._h(pk)}
    if not os.path.exists(GOLD):
        json.dump(got, open(GOLD, "w"), indent=1)
    assert json.load(open(GOLD)) == got


def test_sift_batch_p4_jit_route_exact(monkeypatch):
    """Batch reads >= 4 queries take the register-blocked JIT p4 kernel; scores must equal the C kernel's, on full and prefix
    dims, for query counts that are and are not multiples of 8."""
    rng = np.random.default_rng(5)
    lat = rng.integers(0, 16, (700, 1024), dtype=np.uint8); pk = S.pack4(lat)
    for nq in (4, 8, 13):
        q = rng.integers(0, 16, (nq, 1024), dtype=np.uint8)
        for du in (None, 256):
            monkeypatch.setenv("OMNIRING_P4_JIT", "1"); a = S.sift_batch_p4(q, pk, du)
            monkeypatch.setenv("OMNIRING_P4_JIT", "0"); b = S.sift_batch_p4(q, pk, du)
            assert np.array_equal(a, b)
