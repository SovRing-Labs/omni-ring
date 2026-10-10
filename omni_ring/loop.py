"""
The ring's tick — five lines, continuously fed (RING-SEED plan step 4; J 2026-09-30: "4 + 1", "fuel injector").

Lines (each switchable, so its contribution is measured, not assumed; a line that changes no measured result is cut):
  write   labelled events become marks (learning = writing)
  sift    queries are scored against every mark in one parallel C call (the read)
  peel    the winner is explained away and the runner-up measured: margin = how clearly the answer stands out
  decay   strengths fade by `decay` per tick; marks beyond `cap` (weakest first) are forgotten, so memory stays bounded
  check   (+1) invariants every tick (mark dtype/shape/range, labels, strengths finite) and the seed golden vectors every
          `golden_every` ticks; a failing tick's writes are rolled back.
Feed: a producer thread batches and ENCODES the next batch while the main thread writes/sifts the current one (encode =
float32 BLAS + C sector16; sift = C; both release the GIL), so the cores see one steady stream instead of phase-sized gulps.
Idle: with no events the tick still runs check + decay, at `idle_s` intervals, instead of dropping to zero.
Telemetry: every tick calls `on_tick(dict)` — timings, throughput, load, clock — the hook the governor (TUNE-0) learns from.
Within one batch, writes are applied before reads.
"""
from __future__ import annotations

import ctypes
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional

import numpy as np

from . import seed as S
from .heads import RingHead

LINES = ("write", "sift", "peel", "decay", "check")


@dataclass
class Event:
    kind: str                    # "write" or "read"
    x: np.ndarray                # one input vector (in_dim,)
    label: Optional[str] = None
    id: Optional[str] = None


@dataclass
class Result:
    id: Optional[str]
    label: str
    confidence: float
    margin: float


class throttle:
    """Context manager: OpenMP threads for the C kernels, BLAS threads for the encoder — restored on exit, so a loop never
    leaves its setting behind for other code in the process (found by test_ring_bank's timing budget)."""

    def __init__(self, n: int):
        self.n = int(n)

    def __enter__(self):
        try:
            self.gomp = ctypes.CDLL("libgomp.so.1")
            self.prev = self.gomp.omp_get_max_threads()
            self.gomp.omp_set_num_threads(self.n)
        except OSError:
            self.gomp = None
        try:
            from threadpoolctl import threadpool_limits
            self.blas = threadpool_limits(limits=min(self.n, 4), user_api="blas")   # measured: BLAS peaks at 4 physical cores
        except ImportError:
            self.blas = None
        return self

    def __exit__(self, *exc):
        if self.gomp is not None:
            self.gomp.omp_set_num_threads(self.prev)
        if self.blas is not None:
            self.blas.restore_original_limits()
        return False


def _load_and_clock():
    v = list(map(int, open("/proc/stat").readline().split()[1:]))
    busy, total = sum(v) - v[3] - v[4], sum(v)
    try:
        mhz = np.mean([int(open(f"/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_cur_freq").read())
                       for c in range(os.cpu_count() or 1)]) / 1000
    except OSError:
        mhz = float("nan")
    return busy, total, mhz


@dataclass
class Loop:
    head: RingHead
    batch: int = 256
    depth: int = 2                          # feed queue depth (1 = no overlap: encode then process, in turn)
    threads: int = 8
    lines: tuple = LINES
    decay: float = 0.0                      # strength lost per tick (0 = none)
    cap: Optional[int] = None               # max marks kept by the decay line
    golden_every: int = 50
    golden_path: Optional[str] = None
    idle_s: float = 0.5
    on_tick: Optional[Callable[[dict], None]] = None
    ticks: int = 0
    rollbacks: int = 0
    log: List[dict] = field(default_factory=list)

    # -- feed (producer) -------------------------------------------------------------------------------------------
    def _feed(self, events: Iterable[Event], q: "queue.Queue"):
        buf: List[Event] = []
        for ev in events:
            buf.append(ev)
            if len(buf) >= self.batch:
                q.put(self._encode(buf)); buf = []
        if buf:
            q.put(self._encode(buf))
        q.put(None)

    def _encode(self, evs: List[Event]):
        t = time.perf_counter()
        X = np.stack([e.x for e in evs]).astype(np.float32)
        if self.head.centre is None:
            self.head.fit_centre(X[[e.kind == "write" for e in evs]] if any(e.kind == "write" for e in evs) else X)
        m = S.encode(X, self.head.proj, self.head.centre)
        return evs, m, time.perf_counter() - t

    # -- lines ---------------------------------------------------------------------------------------------------------
    def _check(self) -> bool:
        h = self.head
        ok = (h.marks.dtype == np.uint8 and h.marks.ndim == 2 and h.marks.shape[1] == h.d
              and (len(h.marks) == 0 or int(h.marks.max()) < S.K)
              and len(h.y) == len(h.marks) == len(h.w) and (len(h.y) == 0 or int(h.y.max()) < len(h.classes))
              and bool(np.all(np.isfinite(h.w))))
        if ok and self.golden_path and self.ticks % self.golden_every == 0:
            ok = not list(S.check_golden(self.golden_path))
        return ok

    def _decay(self) -> int:
        h = self.head
        if self.decay > 0 and len(h.w):
            h.w *= np.float32(1.0 - self.decay)
        if self.cap is not None and len(h.w) > self.cap:
            h.forget(np.argsort(h.w, kind="stable")[: len(h.w) - self.cap])   # weakest (oldest under decay) first
            return 1
        return 0

    def _tick(self, evs: List[Event], m: np.ndarray, enc_s: float) -> List[Result]:
        t0 = time.perf_counter()
        h, out = self.head, []
        n0 = len(h.marks)
        wi = [i for i, e in enumerate(evs) if e.kind == "write"]
        ri = [i for i, e in enumerate(evs) if e.kind == "read"]
        if "write" in self.lines and wi:
            h.write_marks(m[wi], [evs[i].label for i in wi], [{"id": evs[i].id} for i in wi])
        if "check" in self.lines and not self._check():
            h.forget(list(range(n0, len(h.marks))))                           # roll back this tick's writes
            self.rollbacks += 1
        t1 = time.perf_counter()
        if "sift" in self.lines and ri and len(h.marks):
            labels, conf, margin = h.read_marks(m[ri], use_strength="decay" in self.lines and self.decay > 0)
            if "peel" not in self.lines:
                margin = np.full(len(ri), np.nan)
            out = [Result(evs[i].id, l, float(c), float(g)) for i, l, c, g in zip(ri, labels, conf, margin)]
        t2 = time.perf_counter()
        if "decay" in self.lines:
            self._decay()
        self.ticks += 1
        tel = {"tick": self.ticks, "n_write": len(wi), "n_read": len(ri), "encode_s": enc_s, "write_s": t1 - t0,
               "sift_s": t2 - t1, "marks": len(h.marks), "batch": self.batch, "depth": self.depth, "threads": self.threads}
        self.log.append(tel)
        if self.on_tick:
            self.on_tick(tel)
        return out

    # -- run -------------------------------------------------------------------------------------------------------
    def run(self, events: Iterable[Event]) -> List[Result]:
        """Stream events through the tick. Returns read results in event order."""
        with throttle(self.threads):
            return self._run(events)

    def _run(self, events: Iterable[Event]) -> List[Result]:
        results: List[Result] = []
        if self.depth <= 1:                                                  # sequential: encode, then process, in turn
            buf: List[Event] = []
            for ev in list(events) + [None]:
                if ev is not None:
                    buf.append(ev)
                if buf and (ev is None or len(buf) >= self.batch):
                    evs, m, es = self._encode(buf)
                    results += self._tick(evs, m, es); buf = []
            return results
        q: "queue.Queue" = queue.Queue(maxsize=self.depth)
        th = threading.Thread(target=self._feed, args=(events, q), daemon=True)
        th.start()
        while True:
            item = q.get()
            if item is None:
                break
            results += self._tick(*item)
        th.join()
        return results

    def idle(self, seconds: float) -> None:
        """No events: keep the engine at a steady idle — check + decay every idle_s."""
        end = time.time() + seconds
        while time.time() < end:
            if "check" in self.lines and not self._check():
                self.rollbacks += 1
            if "decay" in self.lines:
                self._decay()
            self.ticks += 1
            time.sleep(self.idle_s)
