import numpy as np
import pytest

from omni_ring import solids as SO


@pytest.mark.parametrize("n", [2, 4, 8, 16])
def test_simplex_gram_is_exact(n):
    c = SO.simplex(n, 5120)
    g = SO.gram(c)
    off = g[~np.eye(n, dtype=bool)]
    assert np.allclose(np.diag(g), 1.0)
    assert np.allclose(off, -1.0 / (n - 1), atol=1e-9)


def test_tetrahedron_is_four_equidistant_points():
    c = SO.simplex(4, 3 * 1000)
    assert c.shape == (4, 3000)
    assert np.isclose(SO.min_separation(c), np.sqrt(2 + 2 / 3))      # edge of a unit-circumradius tetrahedron


def test_orthogonal_and_cross_polytope():
    o = SO.gram(SO.orthogonal(8, 4096))
    assert np.allclose(o, np.eye(8), atol=1e-9)
    x = SO.gram(SO.cross_polytope(4, 4096))
    want = np.block([[np.eye(4), -np.eye(4)], [-np.eye(4), np.eye(4)]])
    assert np.allclose(x, want, atol=1e-9)


def test_polygon_is_circulant():
    g = SO.gram(SO.polygon(8, 8192))
    for k in range(8):
        assert np.allclose(np.diag(g, k), g[0, k], atol=0.05)


def test_hypercube_vertices_are_plus_minus_one():
    c = SO.hypercube(5, 1000)
    assert set(np.unique(c)) <= {0, 8}


def test_simplex_beats_random_on_worst_pair():
    rnd = np.random.default_rng(0).integers(0, 16, (16, 5115)).astype(np.uint8)
    assert SO.min_separation(SO.simplex(16, 5120)) > SO.min_separation(rnd)
