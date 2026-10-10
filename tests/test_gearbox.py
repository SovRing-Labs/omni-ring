"""Gearbox: shifting gears live never loses or changes an answer."""
import time
import numpy as np

from omni_ring.belt import Belt
from omni_ring.gearbox import Gearbox
from omni_ring.heads import RingHead


def test_answers_survive_gear_shifts():
    rng = np.random.default_rng(0)
    c = rng.standard_normal((5, 32)).astype(np.float32)
    y = rng.integers(0, 5, 1500)
    X = (c[y] + 0.6 * rng.standard_normal((1500, 32))).astype(np.float32)
    h = RingHead("t", 32, d=512); h.fit_centre(X[:500]); h.write(X[:500], [f"c{v}" for v in y[:500]])
    want = h.read(X[500:])[0]
    gb = Gearbox(Belt(h), window=0.05, up_after=1).start()
    try:
        futs = []
        for k in range(0, 1000, 100):
            futs.append((gb.belt.submit_batch(X[500 + k:600 + k]), k))
            futs += [(gb.belt.submit_single(X[500 + k + i]), ("s", k + i)) for i in range(3)]
            gb._apply((gb.gear + 1) % len(gb.gears), "test")        # force a shift mid-stream every step
            time.sleep(0.01)
        for f, key in futs:
            r = f.result(timeout=60)
            if isinstance(key, tuple):
                assert r[0] == want[key[1]]
            else:
                assert list(r[0]) == want[key:key + 100]
    finally:
        gb.stop()
    assert len(gb.shifts) >= 5
