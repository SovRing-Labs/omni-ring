"""
Tests for the PEEL cued multi-sweep readout (omni_ring.peel).

Pre-registered bars, from `_Pads/RING/omni-peel.wave.yaml`:
  (a) a single clean mark is read back exactly
  (b) over 20 slots of 24 distinct marks: mean true recovered >= 12, and
      accepted non-members <= 5% of accepted
  (c) over 20 slots of 12 marks: mean true >= 9
  (d) an empty slot (zeros) returns []

rotations are fixed at seed 0. The RING3 harness disclosure records that
unguided settles on the rotation rings are a seed lottery (seeds 3/21/42 decode
a clean single mark 1/15 while 42 is the engine default), so the fixture pins a
cued-good seed and never relies on the engine default.
"""
import time

import numpy as np
import pytest

from omni_ring import chain
from omni_ring.codebook import generate_rotations
from omni_ring.peel import (
    DEFAULT_MIN_CONFIDENCE,
    build_codebooks,
    read_all,
    read_one,
)
from omni_ring.types import DEFAULT_SPEC

ROTATIONS = generate_rotations(DEFAULT_SPEC, seed=0)
SPEC = DEFAULT_SPEC
N_INFO = SPEC.data_capacity
D = SPEC.dimension
CUE_RING = 3  # the 13-ring


def _slot(marks):
    """A crowded slot: the superposition of each mark's exact codeword."""
    return np.sum([chain.encode(int(x), ROTATIONS) for x in marks], axis=0)


def _sample_marks(rng, m):
    return [int(x) for x in rng.choice(N_INFO, size=m, replace=False)]


# --- (a) a single clean mark is read back exactly ----------------------------


@pytest.mark.parametrize("x", [0, 1, 7, 137, 500, 1364])
def test_single_clean_mark_read_back_exactly(x):
    """One mark in the slot -> that mark and nothing else."""
    accepted = read_all(chain.encode(x, ROTATIONS), ROTATIONS)
    assert accepted == [x], f"expected exactly [{x}], got {accepted}"


def test_single_clean_mark_via_read_one():
    """read_one agrees with read_all on the cue that matches the mark."""
    x = 137
    assert read_one(chain.encode(x, ROTATIONS), ROTATIONS, cue=x % 13) == x


def test_read_one_rejects_a_wrong_cue():
    """A wrong cue is not silently answered with a confident-looking fake."""
    x = 137  # true cue is 7
    got = read_one(chain.encode(x, ROTATIONS), ROTATIONS, cue=0)
    assert got != x


def test_read_one_returns_none_on_empty_slot():
    assert read_one(np.zeros(D, dtype=complex), ROTATIONS, cue=0) is None


# --- (b) crowded slot, M = 24 ------------------------------------------------


def test_crowded_slot_24_marks_recall_and_fake_rate():
    """20 slots of 24 distinct marks: mean true >= 12, fakes <= 5% of accepted."""
    rng = np.random.default_rng(20260929)
    n_slots, m = 20, 24

    true_marks = 0
    accepted_total = 0
    elapsed = 0.0

    for _ in range(n_slots):
        marks = _sample_marks(rng, m)
        member = set(marks)

        t0 = time.perf_counter()
        accepted = read_all(_slot(marks), ROTATIONS)
        elapsed += time.perf_counter() - t0

        assert len(set(accepted)) == len(accepted), "read_all returned a duplicate"
        true_marks += sum(1 for x in accepted if x in member)
        accepted_total += len(accepted)

    mean_true = true_marks / n_slots
    fakes = accepted_total - true_marks
    fake_rate = fakes / accepted_total if accepted_total else 0.0
    ms_per_read = 1000.0 * elapsed / n_slots

    print(
        f"\n[M=24] mean true {mean_true:.2f}/{m} | fakes {fakes}/{accepted_total} "
        f"= {fake_rate:.2%} | {ms_per_read:.1f} ms per read_all"
    )

    assert mean_true >= 12, f"mean true recovered {mean_true:.2f} < 12"
    assert fake_rate <= 0.05, f"fake rate {fake_rate:.2%} > 5%"


# --- (c) crowded slot, M = 12 ------------------------------------------------


def test_crowded_slot_12_marks_recall():
    """20 slots of 12 marks: mean true >= 9."""
    rng = np.random.default_rng(12012)
    n_slots, m = 20, 12

    true_marks = 0
    for _ in range(n_slots):
        marks = _sample_marks(rng, m)
        member = set(marks)
        accepted = read_all(_slot(marks), ROTATIONS)
        true_marks += sum(1 for x in accepted if x in member)

    mean_true = true_marks / n_slots
    print(f"\n[M=12] mean true {mean_true:.2f}/{m}")
    assert mean_true >= 9, f"mean true recovered {mean_true:.2f} < 9"


# --- (d) an empty slot is empty ---------------------------------------------


def test_empty_slot_returns_nothing():
    """A zero slot carries no mark; reporting coordinate 0 would be a report."""
    assert read_all(np.zeros(D, dtype=complex), ROTATIONS) == []


def test_explained_away_single_mark_does_not_yield_a_null_settle():
    """Once the last mark is subtracted, sweep 2 must not invent coordinate 0."""
    for x in (0, 137, 1364):
        accepted = read_all(chain.encode(x, ROTATIONS), ROTATIONS)
        assert accepted == [x]


# --- measured cost ----------------------------------------------------------


def test_read_all_timing_at_m24():
    """Record ms per read_all at M = 24 and guard against a blow-up."""
    rng = np.random.default_rng(24024)
    m, reps = 24, 5

    t0 = time.perf_counter()
    for _ in range(reps):
        read_all(_slot(_sample_marks(rng, m)), ROTATIONS)
    ms_per_read = 1000.0 * (time.perf_counter() - t0) / reps

    print(f"\n[timing] {ms_per_read:.1f} ms per read_all at M={m}")
    assert ms_per_read < 5000, f"read_all took {ms_per_read:.0f} ms at M={m}"


# --- codebook contract ------------------------------------------------------


def test_build_codebooks_shapes():
    cbs = build_codebooks(ROTATIONS, SPEC)
    assert len(cbs) == len(SPEC.all_moduli)
    for cb, m in zip(cbs, SPEC.all_moduli):
        assert cb.shape == (m, D)


def test_build_codebooks_requires_one_rotation_per_ring():
    with pytest.raises(ValueError):
        build_codebooks(ROTATIONS[:-1], SPEC)


def test_read_all_rejects_out_of_range_cue_ring():
    with pytest.raises(ValueError):
        read_all(
            chain.encode(1, ROTATIONS), ROTATIONS, cue_ring=len(SPEC.all_moduli)
        )


def test_default_confidence_floor_is_in_the_measured_gap():
    """The floor must sit above the impostor band and below the true band."""
    assert 0.22 < DEFAULT_MIN_CONFIDENCE < 0.9
