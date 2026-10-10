"""
The belt — serving a ring head to live traffic (J 2026-09-30: "S s s B … the light-weight essential ops always have a route").

Two kinds of request share one head:
  single  one query that wants an answer now (a reflex, a Check, a bus read) — served on the 1-thread path (no thread wake)
  batch   many queries (a heavy job) — cut into bounded CHUNKS so no single ever waits behind a whole job
Modes (BELT-0 decides which ships):
  fifo     baseline dynamic batching: one worker, arrival order; waiting singles are gathered (window_ms / max_gather) into
           one call; a batch job runs whole, at full threads
  rhythm   J's interleave in time: slots s s s B … — each s slot serves the waiting singles (up to max_gather) on 1 thread,
           each B slot serves ONE chunk of the oldest batch job at full threads; empty slots are skipped, never idled
  stagger  J's "S,s S,s S,s S,s B": `single_lanes` independent single lanes (1 thread each), started half a cycle apart so
           one is starting while the other is mid-query (two queries overlapped, not one query split across 2 threads in
           lock-step), plus a batch lane on the remaining threads
  express  lanes in space: an express worker serves singles continuously on `express_threads` threads while the batch
           worker serves chunks on the remaining threads (OpenMP thread count is per calling thread, so both run at once)
Precedent: chunked prefill in vLLM / Sarathi-Serve "stall-free batching" (light decode steps ride with bounded heavy chunks).
All reads go through RingHead.read_marks, so every mode returns exactly the answers of a direct read.
"""
from __future__ import annotations

import ctypes
import threading
import time
from collections import deque
from concurrent.futures import Future
from typing import List, Optional

import numpy as np

from . import seed as S
from .heads import RingHead

try:
    _GOMP = ctypes.CDLL("libgomp.so.1")
except OSError:
    _GOMP = None


def _threads(n: int) -> None:
    """Set THIS thread's OpenMP team size for the C kernels (per-thread setting)."""
    if _GOMP is not None:
        _GOMP.omp_set_num_threads(int(n))


class _Job:
    __slots__ = ("q", "fut", "t", "done", "labels", "conf", "margin")

    def __init__(self, q: np.ndarray, fut: Future):
        self.q, self.fut, self.t, self.done = q, fut, time.perf_counter(), 0
        n = len(q)
        self.labels: List[Optional[str]] = [None] * n
        self.conf, self.margin = np.zeros(n), np.zeros(n)


class Belt:
    def __init__(self, head: RingHead, mode: str = "rhythm", ratio: int = 3, chunk: int = 32, max_gather: int = 8,
                 window_ms: float = 1.0, threads: int = 8, express_threads: int = 1, single_threads: int = 1):
        assert mode in ("fifo", "rhythm", "express", "stagger")
        self.head, self.mode, self.ratio, self.chunk = head, mode, ratio, chunk
        self.max_gather, self.window, self.threads, self.express_threads = max_gather, window_ms / 1000, threads, express_threads
        self.single_threads = single_threads     # rhythm: threads per s slot ("ss" = 2)
        self.single_lanes = 2                     # stagger: independent 1-thread single lanes
        self.stagger_s = 0.002                    # stagger: offset between lane starts (~half a single's service time)
        self.singles: deque = deque()
        self.batches: deque = deque()
        self.cv = threading.Condition()
        self.running = False
        self.workers: List[threading.Thread] = []
        self.slots = {"s": 0, "B": 0, "skip": 0}
        self.latencies: deque = deque(maxlen=512)   # recent single latencies (s) — the gearbox's feel for the road
        self.served = {"single": 0, "batch_q": 0}

    # -- submit ----------------------------------------------------------------------------------------------------
    def _enc(self, X: np.ndarray) -> np.ndarray:
        return S.encode(np.atleast_2d(X), self.head.proj, self.head.centre)

    def submit_single(self, x: np.ndarray) -> Future:
        fut: Future = Future()
        job = _Job(self._enc(x), fut)
        with self.cv:
            self.singles.append(job)
            self.cv.notify_all()
        return fut

    def submit_batch(self, X: np.ndarray) -> Future:
        fut: Future = Future()
        job = _Job(self._enc(X), fut)
        with self.cv:
            self.batches.append(job)
            self.cv.notify_all()
        return fut

    # -- serve -----------------------------------------------------------------------------------------------------
    def _read(self, q: np.ndarray):
        return self.head.read_marks(q)

    def _serve_singles(self, n_threads: int, max_n: Optional[int] = None) -> int:
        """Gather waiting singles (up to max_n or max_gather, waiting at most `window` for company); answer them in one call."""
        cap = max_n or self.max_gather
        with self.cv:
            if not self.singles:
                return 0
            deadline = time.perf_counter() + self.window
            while len(self.singles) < cap and time.perf_counter() < deadline:
                self.cv.wait(timeout=max(0.0, deadline - time.perf_counter()))
            jobs = [self.singles.popleft() for _ in range(min(cap, len(self.singles)))]
        if not jobs:
            return 0
        _threads(n_threads)
        labels, conf, margin = self._read(np.concatenate([j.q for j in jobs]))
        now = time.perf_counter()
        for i, j in enumerate(jobs):
            self.latencies.append(now - j.t)
            j.fut.set_result((labels[i], float(conf[i]), float(margin[i]), now - j.t))
        self.served["single"] += len(jobs)
        return len(jobs)

    def _serve_chunk(self, n_threads: int, whole: bool = False) -> int:
        """One bounded chunk of the oldest batch job (or the whole job in fifo mode) at n_threads."""
        with self.cv:
            if not self.batches:
                return 0
            job = self.batches[0]
            lo = job.done
            hi = len(job.q) if whole else min(len(job.q), lo + self.chunk)
            job.done = hi
            if hi >= len(job.q):
                self.batches.popleft()
        _threads(n_threads)
        labels, conf, margin = self._read(job.q[lo:hi])
        job.labels[lo:hi], job.conf[lo:hi], job.margin[lo:hi] = labels, conf, margin
        self.served["batch_q"] += hi - lo
        if hi >= len(job.q):
            job.fut.set_result((job.labels, job.conf, job.margin, time.perf_counter() - job.t))
        return hi - lo

    def _wait_work(self, need_singles: bool, need_batches: bool) -> bool:
        with self.cv:
            while self.running and not ((need_singles and self.singles) or (need_batches and self.batches)):
                self.cv.wait(timeout=0.05)
            return self.running

    def _run_fifo(self):
        while self._wait_work(True, True):
            # arrival order: whichever head-of-line request is older goes first
            s_t = self.singles[0].t if self.singles else float("inf")
            b_t = self.batches[0].t if self.batches else float("inf")
            if s_t <= b_t:
                self._serve_singles(self.threads)
            else:
                self._serve_chunk(self.threads, whole=True)

    def _run_rhythm(self):
        pattern = ["s"] * self.ratio + ["B"]
        k = 0
        while self._wait_work(True, True):
            slot = pattern[k % len(pattern)]; k += 1
            n = self._serve_singles(self.single_threads) if slot == "s" else self._serve_chunk(self.threads)
            self.slots[slot if n else "skip"] += 1

    def _run_express(self):
        while self._wait_work(True, False):
            self._serve_singles(self.express_threads)

    def _run_batch_lane(self):
        spare = self.single_lanes if self.mode == "stagger" else self.express_threads
        while self._wait_work(False, True):
            self._serve_chunk(max(1, self.threads - spare))

    def _run_single_lane(self, lane: int):
        time.sleep(lane * self.stagger_s)          # stagger the lanes' phase once; they stay offset as they alternate pickups
        per = max(1, self.max_gather // self.single_lanes)
        while self._wait_work(True, False):
            self._serve_singles(1, per)

    def start(self) -> "Belt":
        self.running = True
        targets = {"fifo": [self._run_fifo], "rhythm": [self._run_rhythm],
                   "express": [self._run_express, self._run_batch_lane],
                   "stagger": [lambda i=i: self._run_single_lane(i) for i in range(self.single_lanes)] + [self._run_batch_lane]}[self.mode]
        self.workers = [threading.Thread(target=t, daemon=True) for t in targets]
        for w in self.workers:
            w.start()
        return self

    def backlog(self):
        """(singles waiting, batch queries waiting)."""
        with self.cv:
            return len(self.singles), sum(len(j.q) - j.done for j in self.batches)

    def reconfigure(self, **kw) -> None:
        """Shift gear: finish the in-flight call, swap settings, restart the workers. Queued work is kept."""
        with self.cv:
            self.running = False
            self.cv.notify_all()
        for w in self.workers:
            w.join(timeout=10)
        for k, v in kw.items():
            setattr(self, k, v)
        self.start()

    def stop(self) -> None:
        with self.cv:
            self.running = False
            self.cv.notify_all()
        for w in self.workers:
            w.join(timeout=5)
        _threads(1)
