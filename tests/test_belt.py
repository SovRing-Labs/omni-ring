"""The belt (omni_ring.belt): every mode returns exactly the direct-read answers, for singles and chunked batches."""
import numpy as np
import pytest

from omni_ring.belt import Belt
from omni_ring.heads import RingHead


def _head_and_queries():
    rng = np.random.default_rng(0)
    c = rng.standard_normal((5, 32)).astype(np.float32)
    y = rng.integers(0, 5, 700)
    X = (c[y] + 0.6 * rng.standard_normal((700, 32))).astype(np.float32)
    h = RingHead("t", 32, d=512); h.fit_centre(X[:500]); h.write(X[:500], [f"c{v}" for v in y[:500]])
    return h, X[500:]


@pytest.mark.parametrize("mode", ["fifo", "rhythm", "express", "stagger"])
def test_belt_matches_direct(mode):
    h, Q = _head_and_queries()
    want = h.read(Q)[0]
    b = Belt(h, mode=mode, chunk=16).start()
    try:
        singles = [b.submit_single(q) for q in Q[:40]]
        batch = b.submit_batch(Q[40:])
        got_s = [f.result(timeout=30)[0] for f in singles]
        got_b = batch.result(timeout=30)[0]
    finally:
        b.stop()
    assert got_s == want[:40] and list(got_b) == want[40:]
