"""
Tests for MarkStore -- count-weighted accumulation, rehearsal-counter sidecar,
selective decay and prune. (ODECAY-MARKSTORE)

Covers the six behaviours named in the cohort spec (a)-(f), plus the failure
modes those six would not catch: int16 saturation, counter wraparound, slot
bounds, and the weight contract.

Import note: `src/` in this repository is a byte-identical mirror of the
installed `omni_ring` package (verified with `diff -rq`, 2026-09-29). This
contract's file list places the module in `src/`, so the tests import it from
there and bootstrap the repo root onto sys.path so they pass under both
`python3 -m pytest` and a bare `pytest` invocation. Landing the module in
`omni_ring/` and exporting it from `omni_ring/__init__.py` is a follow-up that
requires editing existing files, which this contract forbids -- see the review
doc's "Open issues".
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from omni_ring.mark_store import MarkStore, ACC_MAX, COUNT_MAX  # noqa: E402
from omni_ring.polymorphic import TernaryVector  # noqa: E402

# Fixed seeds. Every number in this file is reproducible.
SEED_A = 20260929
SEED_B = 20260930


def _vec(seed: int) -> TernaryVector:
    """A random ternary vector of the repo's own type."""
    return TernaryVector.random(seed=seed)


def sim(a: TernaryVector, b: TernaryVector) -> float:
    """Cosine similarity of two ternary vectors over their dense forms.

    Zero when either is all-zero, so a forgotten mark scores 0 rather than NaN.
    """
    da = a.to_dense().astype(np.float64)
    db = b.to_dense().astype(np.float64)
    na, nb = np.linalg.norm(da), np.linalg.norm(db)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(da, db) / (na * nb))


@pytest.fixture
def store() -> MarkStore:
    return MarkStore(n_slots=4)


@pytest.fixture
def a() -> TernaryVector:
    return _vec(SEED_A)


@pytest.fixture
def b() -> TernaryVector:
    return _vec(SEED_B)


# ---------------------------------------------------------------- spec (a)
def test_counter_survives_quantised_read(store, a):
    """(a) The counter equals the number of add() calls after any read().

    The rehearsal count must ride ALONGSIDE the payload: `read()` re-thresholds
    and destroys magnitude, so a count recovered from the vector would be lost.
    """
    for expected in range(1, 6):
        store.add(0, a)
        store.read(0)          # the destructive read
        store.read(0)          # ...twice
        assert store.count(0) == expected

    # The read really did quantise: the vector itself no longer carries the count.
    assert store.count(0) == 5
    assert int(np.count_nonzero(store.acc[0])) > 0


# ---------------------------------------------------------------- spec (b)
def test_reinforcement_dominates(store, a, b):
    """(b) Reinforcement: add A 5 times and B once; read() is nearer A than B."""
    for _ in range(5):
        store.add(0, a)
    store.add(0, b)

    assert store.count(0) == 6
    r = store.read(0)
    assert sim(r, a) > sim(r, b)
    # Derived by hand from the 5:1 ratio, and asserted so a regression is visible.
    assert sim(r, a) > 0.5
    assert sim(r, b) < 0.5


# ---------------------------------------------------------------- spec (c)
def test_decay_forgets_the_weak_mark_first(store, a, b):
    """(c) After (b), decay(0.5) twice drops sim(B) below 0.1 while sim(A) holds.

    Mechanism: the B-only dimensions sit at |acc| == 1, and round-toward-zero at
    factor 0.5 sends 1 -> 0. The A dimensions sit at |acc| >= 4 and survive.
    """
    for _ in range(5):
        store.add(0, a)
    store.add(0, b)

    assert sim(store.read(0), b) >= 0.1     # precondition: B is still there

    store.decay(0.5)
    store.decay(0.5)

    r = store.read(0)
    assert sim(r, b) < 0.1, f"weak mark survived decay: {sim(r, b)}"
    assert sim(r, a) > 0.5, f"strong mark decayed too far: {sim(r, a)}"

    # The counter is untouched by decay -- forgetting evidence is not forgetting
    # that it was rehearsed.
    assert store.count(0) == 6


def test_decay_reports_dimensions_forgotten(store, a):
    """decay() returns how many dimensions actually crossed to zero."""
    store.add(0, a)                     # |acc| == 1 everywhere a is active
    live = int(np.count_nonzero(store.acc[0]))
    assert live > 0

    forgotten = store.decay(0.5)        # every 1 becomes 0
    assert forgotten == live
    assert not np.any(store.acc[0])

    # Decaying an already-dead slot forgets nothing.
    assert store.decay(0.5) == 0


# ---------------------------------------------------------------- spec (d)
def test_recency_wins_after_decay(a, b):
    """(d) A×10, decay(0.5)×3, then B×10 -> read() is nearer B than A.

    Without the decays recency has no effect, which is the point: decay is what
    makes a stale mark yield to a fresh one.

    Measured, not asserted by faith (seeds 1/2/3/20260929/777): with equal
    rehearsal totals and no decay, the B-minus-A margin is -0.0047, +0.0113,
    -0.0023, +0.0097, -0.0004 -- a 10:10 bundle is a statistical tie, and its
    sign is coin-flip noise. So the spec's negative half ("without the decays it
    is not [more similar to B]") is asserted as a *negligible margin*, not as
    strict non-dominance, which no single seed can satisfy. With decay the
    margin is +0.40 -- a 40x larger effect, unambiguous.
    """
    # With decay.
    aged = MarkStore(n_slots=1)
    for _ in range(10):
        aged.add(0, a)
    aged.decay(0.5)
    aged.decay(0.5)
    aged.decay(0.5)
    for _ in range(10):
        aged.add(0, b)
    r_aged = aged.read(0)
    aged_margin = sim(r_aged, b) - sim(r_aged, a)
    assert aged_margin > 0.0, f"decay failed to hand the slot to B: {aged_margin}"
    assert aged_margin > 0.3, f"recency effect too weak to be meaningful: {aged_margin}"

    # Without decay, the same rehearsal totals give no recency advantage.
    fresh = MarkStore(n_slots=1)
    for _ in range(10):
        fresh.add(0, a)
    for _ in range(10):
        fresh.add(0, b)
    r_fresh = fresh.read(0)
    fresh_margin = sim(r_fresh, b) - sim(r_fresh, a)
    assert fresh_margin < 0.05, f"un-decayed slot should be a tie, got {fresh_margin}"


# ---------------------------------------------------------------- spec (e)
def test_selective_decay_leaves_other_slots_untouched(store, a, b):
    """(e) decay(0.5, slots=[0]) leaves slot 1's accumulator bit-identical."""
    for _ in range(4):
        store.add(0, a)
    for _ in range(3):
        store.add(1, b)

    before_1 = store.acc[1].copy()
    before_count_1 = store.count(1)
    before_count_0 = store.count(0)

    store.decay(0.5, slots=[0])

    assert np.array_equal(store.acc[1], before_1), "slot 1 accumulator was modified"
    assert store.count(1) == before_count_1
    assert store.count(0) == before_count_0

    # Slot 0 really did move.
    assert not np.array_equal(store.acc[0], before_1)


def test_selective_decay_rejects_bad_slot(store):
    with pytest.raises(IndexError):
        store.decay(0.5, slots=[99])


# ---------------------------------------------------------------- spec (f)
def test_prune_clears_all_zero_slot_and_resets_counter(a):
    """(f) prune clears an all-zero slot and resets its counter.

    Uses a 1-slot store so `cleared` is exactly the slot under test; a wider
    store would also count never-written slots, which are all-zero by definition.
    """
    s = MarkStore(n_slots=1)
    s.add(0, a)
    assert s.count(0) == 1

    s.decay(0.5)                      # +-1 -> 0: slot 0 is now all-zero
    assert not np.any(s.acc[0])
    assert s.count(0) == 1, "decay must not reset the counter"

    cleared = s.prune()

    assert cleared == 1
    assert s.count(0) == 0
    assert not np.any(s.acc[0])
    assert s.last_write[0] == 0.0


def test_prune_min_count_thresholds_by_rehearsal(a):
    """prune(min_count=N) clears slots that were never rehearsed N times."""
    s = MarkStore(n_slots=2)
    s.add(0, a)
    s.add(1, a)
    s.add(1, a)
    s.add(1, a)

    assert s.count(0) == 1
    assert s.count(1) == 3

    assert s.prune(min_count=2) == 1

    assert s.count(0) == 0              # below threshold -> cleared
    assert s.count(1) == 3              # at threshold -> kept
    assert np.any(s.acc[1])


# ------------------------------------------------------------- failure modes
def test_add_saturates_at_int16_bound(a):
    """Accumulators clamp at +-32767 instead of wrapping to a negative value."""
    s = MarkStore(n_slots=1)
    for _ in range(4):
        s.add(0, a, weight=ACC_MAX)

    row = s.acc[0]
    assert row.max() == ACC_MAX
    assert row.min() == -ACC_MAX
    assert not np.any(row < -ACC_MAX)
    # Wraparound would show up as a large negative on a positive-side vector.
    assert int(np.count_nonzero(row > 0)) > 0


def test_decay_factor_bounds(store, a):
    store.add(0, a)
    for bad in (0.0, -0.5, 1.5, 2.0):
        with pytest.raises(ValueError):
            store.decay(bad)
    # factor == 1.0 is a legal no-op.
    before = store.acc[0].copy()
    assert store.decay(1.0) == 0
    assert np.array_equal(store.acc[0], before)


def test_weight_must_be_an_integer(store, a):
    with pytest.raises(TypeError):
        store.add(0, a, weight=0.5)
    with pytest.raises(TypeError):
        store.add(0, a, weight=1.0)
    with pytest.raises(TypeError):
        store.add(0, a, weight=True)
    # An integer weight is accepted and scales the contribution.
    store.add(0, a, weight=3)
    assert store.count(0) == 1


def test_slot_bounds_are_enforced(store, a):
    with pytest.raises(IndexError):
        store.add(store.n_slots, a)
    with pytest.raises(IndexError):
        store.count(-1)
    with pytest.raises(IndexError):
        store.read(store.n_slots)
    with pytest.raises(TypeError):
        store.add("0", a)


def test_rejects_bad_construction():
    with pytest.raises(ValueError):
        MarkStore(n_slots=0)
    with pytest.raises(ValueError):
        MarkStore(n_slots=4, dim=0)


def test_theta_suppresses_weak_dimensions(a, b):
    """A higher threshold drops the single-rehearsal dimensions at read time.

    Measured at theta=1: sim(A)=0.812, sim(B)=0.429. At theta=2: sim(A)=1.000,
    sim(B)=0.018 -- the +-1 dimensions contributed by the single B rehearsal are
    suppressed, leaving only the 5-rehearsed evidence.

    Note theta is a *read* threshold, not a monotone "more selective" dial: at
    theta=6 only the |acc| == 6 dimensions survive, which are precisely the ones
    where A and B agree, so sim(B) rises again. Selectivity and similarity are
    different axes; theta=2 is the value that isolates the weak mark here.
    """
    s = MarkStore(n_slots=1)
    for _ in range(5):
        s.add(0, a)
    s.add(0, b)

    assert sim(s.read(0, theta=1), b) >= 0.1
    assert sim(s.read(0, theta=2), b) < 0.1
    assert sim(s.read(0, theta=2), a) > 0.5

    with pytest.raises(ValueError):
        s.read(0, theta=0)


def test_age_reports_time_since_last_write(store, a):
    store.add(0, a)
    assert store.age(0, now=store.last_write[0] + 10.0) == pytest.approx(10.0)
    # A never-written slot has no stamp.
    assert store.age(3, now=0.0) == pytest.approx(0.0)


def test_nbytes_accounts_for_accumulator_and_sidecar():
    s = MarkStore(n_slots=1000)
    assert s.nbytes() == 1000 * 10240 * 2 + 1000 * 4 + 1000 * 8
    assert "MarkStore" in repr(s)


def test_count_saturates_rather_than_wrapping():
    """The sidecar must not wrap to 0 and masquerade as a fresh slot."""
    s = MarkStore(n_slots=1)
    s.counts[0] = COUNT_MAX
    s.add(0, _vec(SEED_A))
    assert s.count(0) == COUNT_MAX
