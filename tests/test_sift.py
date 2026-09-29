"""Exhaustive lattice sift + star prune-join (TORUS/EXH/STAR-0, 2026-09-29)."""
import numpy as np
import pytest

from omni_ring.chain import encode
from omni_ring.codebook import generate_rotations
from omni_ring.sift import has_kernel, prune_join, read, ring_lattice, scores, sift
from omni_ring.types import DEFAULT_SPEC

ROT = generate_rotations(DEFAULT_SPEC, seed=0)
LAT = ring_lattice(ROT)
N = int(np.prod(DEFAULT_SPEC.moduli))


def _slot(xs):
    return sum(encode(int(x), ROT) for x in xs)


def test_lattice_shape_and_rows():
    assert LAT.shape == (N, DEFAULT_SPEC.dimension) and LAT.dtype == np.uint8 and LAT.max() < 16


@pytest.mark.parametrize("m", [1, 24, 192])
def test_sift_reads_every_mark(m):
    xs = set(np.random.default_rng(m).choice(N, m, replace=False).tolist())
    got = set(sift(_slot(xs), LAT))
    assert got == xs


def test_empty_slot_returns_nothing():
    assert sift(np.zeros(DEFAULT_SPEC.dimension, complex), LAT) == []


@pytest.mark.skipif(not has_kernel(), reason="AVX2 kernel not built")
def test_c_kernel_matches_numpy_exactly():
    v = _slot(np.random.default_rng(7).choice(N, 96, replace=False))
    assert np.array_equal(scores(v, LAT, "c"), scores(v, LAT, "numpy"))


def test_read_auto_uses_sift_for_small_lattice():
    xs = {5, 99, 1300}
    got, mode = read(_slot(xs), ROT, lattice=LAT)
    assert mode == "sift" and set(got) == xs


def test_prune_join_star():
    rng = np.random.default_rng(3)
    V, D = 15, DEFAULT_SPEC.dimension
    books = [np.exp(1j * rng.uniform(0, 2 * np.pi, (V, D))) for _ in range(4)]
    cs = np.exp(1j * rng.uniform(0, 2 * np.pi, (17, D)))

    def joint(c):
        c = np.atleast_2d(c)
        return books[0][c[:, 0]] * books[1][c[:, 1]] * books[2][c[:, 2]] * books[3][c[:, 3]] * cs[c.sum(1) % 17]

    recs = {tuple(int(x) for x in rng.integers(0, V, 4)) for _ in range(6)}
    centre = sum(joint(np.array(r))[0] for r in recs)
    points = [sum(books[i][r[i]] for r in recs) for i in range(4)]
    found, tested = prune_join(centre, points, books, joint)
    assert found == recs and tested < V ** 4 // 10
