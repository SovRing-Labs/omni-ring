"""
MarkStore -- count-weighted mark accumulation with a rehearsal-counter sidecar,
selective decay, and prune, for OMNIRING. (ODECAY-MARKSTORE)

Pillar: Nervous System -- the Reinforce/Decay step of the Loop
(Act -> Mark -> Reinforce/Decay -> Read -> Adjust -> Learn).

Why this exists
---------------
`PolymorphicBus.write_slot` overwrites: a second write to a slot erases the first.
Nothing in the engine accumulated marks, so there was no substrate for decay to
act on -- bundled marks never fade, and the bus fills monotonically with whatever
arrived first (peer review 2026-09-28, section 5.4: "If only one thing is built
from this document, build decay").

Why the counter is a SIDECAR and not part of the vector
-------------------------------------------------------
The bus quantises to K=16 (4 bits) on write, and `bundle` is sum-then-threshold.
Repeated writes therefore *re-threshold* rather than accumulate magnitude, so a
rehearsal count **cannot be recovered from the vector** (peer review section 5.2).
Architectural state has to ride alongside the payload, not inside it. That is what
`counts` is: one uint32 per slot, never quantised, never lost to a read.

Layout (per slot, 10,240-D)
---------------------------
  acc        int16[dim]   saturating count-weighted accumulator, |acc| <= 32767
  counts     uint32       rehearsal sidecar -- number of add() calls
  last_write float64      wall-clock stamp of the last add()

Round-toward-zero in `decay` is the whole forgetting mechanism: at factor 0.5 an
accumulator of +-1 becomes 0, so a mark reinforced once dies before a mark
reinforced five times. Ties are not broken by a heuristic -- the magnitude is the
evidence, and the tie-break is the arithmetic.
"""
from __future__ import annotations

import time
from typing import Iterable, Optional, Sequence

import numpy as np

from .polymorphic import TernaryVector, TERNARY_D

# int16 saturating bound. Using +-32767 (not 32768) keeps every stored value's
# absolute value representable, so `abs()` on the int16 row can never hit the
# -32768 overflow trap, and a decay can never produce an out-of-range value.
ACC_MAX = 32767

# Guard the uint32 sidecar against wraparound. Wrap would silently reset a slot's
# rehearsal history from 4_294_967_295 back to 0, which would look exactly like a
# pristine slot and would defeat the promotion gate this counter exists to feed.
COUNT_MAX = 0xFFFFFFFF

__all__ = ["MarkStore", "ACC_MAX", "COUNT_MAX"]


class MarkStore:
    """
    Count-weighted mark store over N slots of D ternary dimensions.

    The accumulator is deliberately wider than the bus payload (int16 vs 4-bit
    packed) so that reinforcement survives the read: magnitude encodes *how often*
    a mark was rehearsed, and `counts` encodes it losslessly alongside.

    Parameters
    ----------
    n_slots : int
        Number of addressable slots. > 0.
    dim : int
        Dimensions per slot. Defaults to the bus width, 10240. `read()` requires
        `dim == TERNARY_D` because it returns a TernaryVector; a store built at
        another width is accumulation/decay-only.
    """

    __slots__ = ("n_slots", "dim", "acc", "counts", "last_write")

    def __init__(self, n_slots: int, dim: int = TERNARY_D):
        if not isinstance(n_slots, (int, np.integer)) or int(n_slots) <= 0:
            raise ValueError(f"n_slots must be a positive integer, got {n_slots!r}")
        if not isinstance(dim, (int, np.integer)) or int(dim) <= 0:
            raise ValueError(f"dim must be a positive integer, got {dim!r}")

        self.n_slots = int(n_slots)
        self.dim = int(dim)
        # acc is the payload: O(1) constant RAM per slot, never rescaled by a read.
        self.acc = np.zeros((self.n_slots, self.dim), dtype=np.int16)
        # counts is the metadata that rides alongside the payload (peer review 5.2).
        self.counts = np.zeros(self.n_slots, dtype=np.uint32)
        # last_write is the recency stamp `age` reads.
        self.last_write = np.zeros(self.n_slots, dtype=np.float64)

    # ------------------------------------------------------------------ helpers

    def _slot(self, slot: int) -> int:
        """Validate and normalise a slot index."""
        if not isinstance(slot, (int, np.integer)):
            raise TypeError(f"slot must be an integer, got {type(slot).__name__}")
        s = int(slot)
        if s < 0 or s >= self.n_slots:
            raise IndexError(f"slot {s} out of range [0, {self.n_slots})")
        return s

    @staticmethod
    def _weight(weight) -> int:
        """Validate a reinforcement weight.

        Integers only. A float weight would make the saturation boundary
        ambiguous, and a fractional weight on a ternary vector is a category
        error -- magnitude in this store is a *count*, not a score.
        """
        if isinstance(weight, bool) or not isinstance(weight, (int, np.integer)):
            raise TypeError(
                f"weight must be an integer (magnitude is a count, not a score), "
                f"got {type(weight).__name__}"
            )
        return int(weight)

    def nbytes(self) -> int:
        """Total bytes held by the store (accumulator + sidecar + timestamps)."""
        return int(self.acc.nbytes + self.counts.nbytes + self.last_write.nbytes)

    # ---------------------------------------------------------------- reinforce

    def add(self, slot: int, tv: TernaryVector, weight: int = 1) -> int:
        """
        Reinforce `slot` with `tv`, weighted by `weight`.

        Accumulates `tv.to_dense() * weight` into the slot with int16 saturation
        at +-32767, increments the rehearsal counter, and stamps the write time.

        Returns the new rehearsal count for the slot.
        """
        s = self._slot(slot)
        w = self._weight(weight)
        if not isinstance(tv, TernaryVector):
            raise TypeError(f"tv must be a TernaryVector, got {type(tv).__name__}")

        dense = tv.to_dense()
        if len(dense) != self.dim:
            raise ValueError(
                f"vector has {len(dense)} dims but store slot has {self.dim}"
            )

        # Widen to int32 for the accumulate+clip so the intermediate sum cannot
        # wrap before it is clamped. to_dense() is int8 in {-1,0,+1}.
        delta = dense.astype(np.int32) * w
        row = self.acc[s].astype(np.int32) + delta
        np.clip(row, -ACC_MAX, ACC_MAX, out=row)
        self.acc[s] = row.astype(np.int16)

        # Saturate rather than wrap: a wrapped counter is indistinguishable from
        # a fresh slot, which is precisely the failure the sidecar exists to stop.
        if self.counts[s] < COUNT_MAX:
            self.counts[s] += 1
        self.last_write[s] = time.time()
        return int(self.counts[s])

    # --------------------------------------------------------------------- read

    def read(self, slot: int, theta: int = 1) -> TernaryVector:
        """
        Quantise the slot back to a TernaryVector: `sign(acc)` where
        `|acc| >= theta`, else 0.

        This is the quantise-on-write trap made explicit -- the read throws
        magnitude away. That is why `count()` exists as a separate, lossless read.
        """
        s = self._slot(slot)
        if self.dim != TERNARY_D:
            raise ValueError(
                f"read() returns a TernaryVector and requires dim == {TERNARY_D}; "
                f"this store has dim == {self.dim} (accumulate/decay only)"
            )
        if not isinstance(theta, (int, np.integer)) or int(theta) < 1:
            raise ValueError(f"theta must be an integer >= 1, got {theta!r}")

        row = self.acc[s].astype(np.int32)
        dense = np.where(np.abs(row) >= int(theta), np.sign(row), 0).astype(np.int8)
        return TernaryVector.from_dense(dense)

    def count(self, slot: int) -> int:
        """Rehearsal count for `slot` -- survives any number of quantised reads."""
        return int(self.counts[self._slot(slot)])

    def age(self, slot: int, now: Optional[float] = None) -> float:
        """Seconds since the slot's last add(). `now` defaults to wall-clock now."""
        s = self._slot(slot)
        reference = time.time() if now is None else float(now)
        return reference - float(self.last_write[s])

    # --------------------------------------------------------------------- decay

    def decay(self, factor: float, slots: Optional[Sequence[int]] = None) -> int:
        """
        Multiply accumulators by `factor`, rounded toward zero.

        Applies to every slot when `slots` is None, otherwise only to the named
        slots (selective decay -- a two-line mask in the spec, and the cheapest
        real forgetting primitive in the engine).

        Returns the total number of dimensions that transitioned non-zero -> zero,
        i.e. how much evidence was actually forgotten.
        """
        f = float(factor)
        if not (0.0 < f <= 1.0):
            raise ValueError(f"factor must satisfy 0 < factor <= 1, got {factor!r}")

        targets = range(self.n_slots) if slots is None else [self._slot(s) for s in slots]

        forgotten = 0
        for s in targets:
            row = self.acc[s]
            before = int(np.count_nonzero(row))
            # float64 holds every int16 accumulator exactly, and trunc() is
            # round-toward-zero, so +-1 at factor 0.5 lands on 0 (forgotten)
            # while +-5 lands on +-2 (survives). Magnitude is the evidence.
            scaled = np.trunc(row.astype(np.float64) * f)
            np.clip(scaled, -ACC_MAX, ACC_MAX, out=scaled)
            after_row = scaled.astype(np.int16)
            self.acc[s] = after_row
            forgotten += before - int(np.count_nonzero(after_row))
        return forgotten

    # --------------------------------------------------------------------- prune

    def prune(self, min_count: int = 0) -> int:
        """
        Clear slots that hold no evidence: an all-zero accumulator, or a rehearsal
        count below `min_count`.

        Clearing resets the counter and the timestamp with the payload, so a pruned
        slot is indistinguishable from a never-written one -- which is what makes
        `age()` and the promotion gate safe to trust.

        Returns the number of slots cleared.
        """
        if not isinstance(min_count, (int, np.integer)) or int(min_count) < 0:
            raise ValueError(f"min_count must be an integer >= 0, got {min_count!r}")
        mc = int(min_count)

        cleared = 0
        for s in range(self.n_slots):
            if not np.any(self.acc[s]) or int(self.counts[s]) < mc:
                self.acc[s] = 0
                self.counts[s] = 0
                self.last_write[s] = 0.0
                cleared += 1
        return cleared

    def __repr__(self) -> str:
        return (
            f"MarkStore(n_slots={self.n_slots}, dim={self.dim}, "
            f"nbytes={self.nbytes()})"
        )
