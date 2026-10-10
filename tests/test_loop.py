"""The five-line tick (omni_ring.loop): equivalence, pipelining, Check rollback, Decay bound."""
import numpy as np

from omni_ring import seed as S
from omni_ring.heads import RingHead
from omni_ring.loop import Event, Loop


def _data(n=600, dim=32, classes=5, seed=0):
    rng = np.random.default_rng(seed)
    centres = rng.standard_normal((classes, dim)).astype(np.float32)
    y = rng.integers(0, classes, n)
    X = (centres[y] + 0.6 * rng.standard_normal((n, dim))).astype(np.float32)
    return X, np.array([f"c{v}" for v in y])


def _head(X):
    h = RingHead("t", X.shape[1], d=512)
    h.fit_centre(X)
    return h


def _stream(X, Y, n_train):
    ev = [Event("write", X[i], Y[i], f"w{i}") for i in range(n_train)]
    return ev + [Event("read", X[i], None, f"r{i}") for i in range(n_train, len(X))]


def test_loop_equals_direct_read_and_depths_agree():
    X, Y = _data()
    direct = _head(X); direct.write(X[:500], Y[:500]); want = direct.read(X[500:])[0]
    for depth in (1, 2, 3):
        h = _head(X)
        got = Loop(h, batch=64, depth=depth).run(_stream(X, Y, 500))
        assert [r.label for r in got] == want


def test_check_rolls_back_a_corrupt_write():
    X, Y = _data(200)
    h = _head(X)
    lp = Loop(h, batch=50)
    lp.run(_stream(X[:100], Y[:100], 100))
    n = len(h.marks)
    real = h.write_marks
    def bad(m, labels, prov=None):
        m = m.copy(); m[0, 0] = 20            # out-of-range phase: a corrupt mark
        return real(m, labels, prov)
    h.write_marks = bad
    lp.run(_stream(X[100:150], Y[100:150], 50))
    assert lp.rollbacks == 1 and len(h.marks) == n


def test_decay_bounds_memory():
    X, Y = _data(900)
    h = _head(X)
    Loop(h, batch=100, decay=0.01, cap=300).run(_stream(X, Y, 900))
    assert len(h.marks) == 300
    assert h.w.min() > 0


def test_check_catches_golden_drift(tmp_path, monkeypatch):
    import json
    p = tmp_path / "g.json"; json.dump(S.golden(), open(p, "w"))
    X, Y = _data(100)
    lp = Loop(_head(X), batch=50, golden_path=str(p), golden_every=1)
    monkeypatch.setattr(S, "_GOLD", np.uint64(0x9E3779B97F4A7C17))
    lp.run(_stream(X, Y, 100))
    assert lp.rollbacks >= 1
