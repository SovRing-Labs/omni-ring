"""TORUS-TEL-1: torus record encoder, learned gear table, novelty reader, torus governor wiring."""
import numpy as np
import pytest

from omni_ring import telemetry as TT
from omni_ring import seed as S

F = TT.MACHINE_FIELDS


def sim(a, b):
    return float(np.real(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_encoder_deterministic_and_smooth():
    e1, e2 = TT.TorusEncoder(F, d=1024), TT.TorusEncoder(F, d=1024)
    x = {"util": 0.5, "ghz": 3.0, "temp": 70.0, "dtemp": 0.0}
    assert np.array_equal(e1.phases(e1.encode(x)), e2.phases(e2.encode(x)))           # SuperSeed: same frame everywhere
    near = dict(x, temp=71.0); far = dict(x, temp=99.0)
    assert sim(e1.encode(x), e1.encode(near)) > 0.95 > sim(e1.encode(x), e1.encode(far))
    assert sim(e1.encode(x), e1.encode(far)) < 0.85


def test_missing_sensor_is_not_a_zero():
    e = TT.TorusEncoder(F, d=1024)
    x = {"util": 0.5, "ghz": 3.0, "temp": 70.0, "dtemp": 0.0}
    assert sim(e.encode(x), e.encode(dict(x, dtemp=float("nan")))) > 0.8
    assert not np.array_equal(e.encode(dict(x, dtemp=float("nan"))), e.encode(dict(x, dtemp=-3.0)))


def _state(e, **kw):
    x = {"util": 0.1, "ghz": 3.8, "temp": 60.0, "dtemp": 0.0}; x.update(kw)
    return e.phases(e.encode(x))


def test_gear_memory_learns_and_abstains():
    e = TT.TorusEncoder(F, d=1024)
    m = TT.GearMemory(4, d=1024)
    rng = np.random.default_rng(0)
    for _ in range(20):   # hot + busy: gear 0 earned more than gear 1; cool + busy: gear 1 earned more
        m.write(_state(e, util=0.9 + rng.normal(0, .01), temp=95 + rng.normal(0, .5)), 0, 1.0)
        m.write(_state(e, util=0.9 + rng.normal(0, .01), temp=95 + rng.normal(0, .5)), 1, -1.0)
        m.write(_state(e, util=0.9 + rng.normal(0, .01), temp=62 + rng.normal(0, .5)), 1, 1.0)
        m.write(_state(e, util=0.9 + rng.normal(0, .01), temp=62 + rng.normal(0, .5)), 0, -0.5)
    assert m.read(_state(e, util=0.9, temp=95))[0] == 0
    assert m.read(_state(e, util=0.9, temp=62))[0] == 1
    g, info = m.read(_state(e, util=0.0, ghz=0.5, temp=99, dtemp=3))                    # never seen: abstain
    assert g is None and info["why"] == "no close state"


def test_novelty_flags_stuck_hot_not_idle_or_load():
    e = TT.TorusEncoder(F)
    nv = TT.Novelty()
    rng = np.random.default_rng(1)
    for _ in range(60):   # normal life: idle cool at full clock, busy warm-to-hot
        nv.learn(_state(e, util=rng.uniform(0.05, 0.3), ghz=rng.uniform(3.4, 3.9), temp=rng.uniform(50, 75), dtemp=rng.normal(0, .3)))
        nv.learn(_state(e, util=rng.uniform(0.6, 0.95), ghz=rng.uniform(3.4, 3.9), temp=rng.uniform(80, 99), dtemp=rng.normal(0.5, .5)))
    idle = nv.score(_state(e, util=0.15, ghz=3.7, temp=62, dtemp=0.0))
    load = nv.score(_state(e, util=0.8, ghz=3.6, temp=92, dtemp=0.4))
    stuck = nv.score(_state(e, util=0.05, ghz=0.8, temp=99, dtemp=0.0))                # hot, idle and throttled
    assert stuck > 2 * max(idle, load), (idle, load, stuck)


def test_torus_governor_runs_and_learns(tmp_path):
    from omni_ring.belt import Belt
    from omni_ring.gearbox import Gearbox
    from omni_ring.heads import RingHead
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 64)).astype(np.float32); y = np.array(["a", "b"] * 100)
    h = RingHead("t", 64); h.fit_centre(X); h.write(X, y)
    gb = Gearbox(Belt(h), profile="thermal", window=0.05, governor="torus", memory_path=str(tmp_path / "m.npz")).start()
    futs = [gb.belt.submit_single(X[i]) for i in range(40)] + [gb.belt.submit_batch(X)]
    [f.result(timeout=30) for f in futs]
    import time; time.sleep(0.4)
    gb.stop()
    assert all("decider" in w and "novelty" in w for w in gb.telemetry)
    assert len(gb.memory.marks) >= 3 and (tmp_path / "m.npz").exists()
    # the frame the governor sees is a SuperSeed frame (portable)
    assert gb.memory.marks.shape[1] == S.D
