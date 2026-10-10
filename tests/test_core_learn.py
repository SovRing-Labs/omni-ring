"""CORE-ROUTE-2 gate: program library readable with checks; a tick leaves a mark; the learner answers clean slots."""
import numpy as np
from omni_ring.core_learn import LearningCore, SEQS
from omni_ring.route import encode_prod


def test_library_and_tick():
    MS = (7, 11, 13); D = 1024; rng = np.random.default_rng(0)
    rots = [rng.integers(0, m, D) * (2 * np.pi / m) for m in MS]
    cbs = [np.exp(1j * np.outer(np.arange(m), r)) for m, r in zip(MS, rots)]
    core = LearningCore(cbs, MS, seed=0)
    assert len(core.library) == len(SEQS) == 6
    for vec in core.library.values():
        assert core.pcb.read(vec) is not None
    x = 500; v = encode_prod(x, rots)
    ans, m = core.tick(v, truth={x})
    assert ans == {x} and m.correct and m.program in core.library and len(m.fired) >= 1
    for _ in range(30):
        x = int(rng.integers(0, 1001)); core.tick(encode_prod(x, rots), truth={x})
    assert np.mean([mk.correct for mk in core.marks]) >= 0.95
