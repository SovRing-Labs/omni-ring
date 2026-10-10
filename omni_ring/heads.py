"""
Ring heads — reflex classifiers that are MEMORY, not weights (RING-SEED plan step 2; lineage RING-HEAD-1, 2026-09-30).

Learning is writing: every labelled example becomes one mark (its SuperSeed v1 phase vector). Reading is the exhaustive
sift over all marks (AVX2 int8 kernel) and a score-weighted vote of the k best. The top score is the head's confidence, so a
head can abstain when nothing it holds is close. No gradient training anywhere: a new example or a correction is live on the
next read, and the marks are byte-identical in any dialect that passes the seed conformance vectors.

RING-HEAD-1 (float encoder) matched or beat trained ternary MLP heads on 5 of 7 targets; this module is the portable version.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import seed as S


class RingHead:
    def __init__(self, name: str, in_dim: int, k: int = 7, d: int = S.D, seed: int = 0, encoder_id: str = ""):
        self.name, self.k, self.d, self.seed = name, k, d, seed
        self.proj = S.Projection(in_dim, d, seed, "head")
        self.encoder_id = encoder_id          # the embedder that produced x (name + sha); marks are only valid under it
        self.centre: Optional[np.ndarray] = None
        self.classes: List[str] = []
        self.marks = np.zeros((0, d), dtype=np.uint8)
        self.y = np.zeros(0, dtype=np.int32)
        self.w = np.zeros(0, dtype=np.float32)   # per-mark strength (1.0 on write; the loop's Decay line lowers it)
        self.prov: List[dict] = []

    # -- write ---------------------------------------------------------------------------------------------------------
    def fit_centre(self, X: np.ndarray) -> None:
        """Freeze the centre (mean of the first writes). Changing it later means re-encoding every mark."""
        self.centre = np.asarray(X, dtype=np.float32).mean(0).astype(np.float32)

    def _cls(self, label: str) -> int:
        if label not in self.classes:
            self.classes.append(label)
        return self.classes.index(label)

    def write(self, X: np.ndarray, labels: Sequence[str], provenance: Optional[Sequence[dict]] = None) -> int:
        if self.centre is None:
            self.fit_centre(X)
        return self.write_marks(S.encode(X, self.proj, self.centre), labels, provenance)

    def write_marks(self, m: np.ndarray, labels: Sequence[str], provenance: Optional[Sequence[dict]] = None) -> int:
        """Append already-encoded marks (the loop's Write line encodes in its feed stage)."""
        self.marks = np.ascontiguousarray(np.concatenate([self.marks, m]))
        self.y = np.concatenate([self.y, np.array([self._cls(l) for l in labels], dtype=np.int32)])
        self.w = np.concatenate([self.w, np.ones(len(m), dtype=np.float32)])
        self.prov.extend(provenance or [{} for _ in labels])
        return len(m)

    def forget(self, idx: Sequence[int]) -> None:
        keep = np.setdiff1d(np.arange(len(self.y)), np.asarray(idx))
        self.marks, self.y, self.w = np.ascontiguousarray(self.marks[keep]), self.y[keep], self.w[keep]
        self.prov = [self.prov[i] for i in keep]

    # -- read ----------------------------------------------------------------------------------------------------------
    def read(self, X: np.ndarray) -> Tuple[List[str], np.ndarray]:
        """(labels, confidence in [-1, 1]) per row. Confidence = best integer score / self-score of a mark."""
        labels, conf, _ = self.read_marks(S.encode(X, self.proj, self.centre))
        return labels, conf

    def _topk(self, qb: np.ndarray, k: int, cascade: Optional[Tuple[int, int]]):
        """-> per query (mark indices, scores) under the SuperSeed v1 top-k rule (revised 2026-09-30): every mark scoring
        >= the k-th best score votes (ties at the k-th place are ALL included), ordered by score desc then mark index asc,
        and votes are added in that order. Breaking k-th-place ties by index (the earlier rule) biased heads with
        exact-duplicate marks toward whichever duplicates were stored first: same-pen d4.error_class 0.745 -> 0.806 and
        d3.outcome 0.595 -> 0.668 with ties included. cascade=(d0, M): coarse pass on the first d0 dims over every mark,
        rescore the M best on all dims, then the same rule inside the candidates (v1.1)."""
        res = []
        if cascade is None:
            sc = S.sift_batch(qb, self.marks)                 # (b, n) int32
            for row in sc:
                kth = np.partition(row, len(row) - k)[len(row) - k]
                sel = np.nonzero(row >= kth)[0]
                o = np.lexsort((sel, -row[sel].astype(np.int64)))
                res.append((sel[o], row[sel][o]))
            return res
        d0, M = cascade
        if getattr(self, "_packed", None) is None or len(self._packed) != len(self.marks):
            self._packed = S.pack4(self.marks)
        coarse = S.sift_batch_p4(qb, self._packed, d0)
        M = min(M, len(self.marks))
        for i in range(len(qb)):
            mth = np.partition(coarse[i], len(coarse[i]) - M)[len(coarse[i]) - M]
            cand_i = np.nonzero(coarse[i] >= mth)[0]           # the M best + every mark tied at the M-th coarse score
            full = S.sift_rows_p4(qb[i], self._packed, cand_i)
            kth = np.partition(full, len(full) - k)[len(full) - k]
            sel = np.nonzero(full >= kth)[0]
            o = np.lexsort((cand_i[sel], -full[sel].astype(np.int64)))
            res.append((cand_i[sel][o], full[sel][o]))
        return res

    def read_marks(self, q: np.ndarray, use_strength: bool = False,
                   cascade: Optional[Tuple[int, int]] = None) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """Read already-encoded queries. Returns (labels, confidence, margin); margin = (winner - runner-up) / total vote,
        the loop's Peel signal (how clearly the winner stands out once it is explained away)."""
        self_score = float(S.sift_phase(self.marks[0], self.marks[:1])[0]) if len(self.marks) else 1.0
        out, conf, margin = [], np.zeros(len(q)), np.zeros(len(q))
        C = len(self.classes)
        k = min(self.k, len(self.marks))
        for b0 in range(0, len(q), 256):                      # one parallel C call per block: threads stay busy
            for j, (nn, top) in enumerate(self._topk(q[b0:b0 + 256], k, cascade)):
                wt = np.maximum(top.astype(np.float64), 0.0) + 1e-6
                if use_strength:
                    wt = wt * self.w[nn]
                votes = np.zeros(C)
                for c, w in zip(self.y[nn], wt):              # added in rule order (bit-identical across dialects)
                    votes[c] += w
                out.append(self.classes[int(votes.argmax())])
                conf[b0 + j] = float(top.max()) / self_score
                srt = np.sort(votes)
                margin[b0 + j] = (srt[-1] - (srt[-2] if C > 1 else 0)) / max(srt.sum(), 1e-9)
        return out, conf, margin

    # -- persist ---------------------------------------------------------------------------------------------------
    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, "marks.npy"), self.marks)
        np.save(os.path.join(path, "y.npy"), self.y)
        np.save(os.path.join(path, "w.npy"), self.w)
        np.save(os.path.join(path, "centre.npy"), self.centre if self.centre is not None else np.zeros(0, np.float32))
        json.dump({"spec_version": S.SPEC_VERSION, "name": self.name, "k": self.k, "d": self.d, "seed": self.seed,
                   "in_dim": self.proj.in_dim, "projection": self.proj.id, "encoder_id": self.encoder_id,
                   "classes": self.classes, "marks": int(len(self.y))}, open(os.path.join(path, "head.json"), "w"), indent=1)
        with open(os.path.join(path, "provenance.jsonl"), "w") as f:
            for p in self.prov:
                f.write(json.dumps(p) + "\n")

    @classmethod
    def load(cls, path: str) -> "RingHead":
        h = json.load(open(os.path.join(path, "head.json")))
        if h["spec_version"] != S.SPEC_VERSION:
            raise ValueError(f"head {h['name']} is seed v{h['spec_version']}, this build is v{S.SPEC_VERSION}: re-encode")
        r = cls(h["name"], h["in_dim"], h["k"], h["d"], h["seed"], h["encoder_id"])
        r.classes, r.marks = h["classes"], np.load(os.path.join(path, "marks.npy"))
        r.y = np.load(os.path.join(path, "y.npy"))
        wp = os.path.join(path, "w.npy")
        r.w = np.load(wp) if os.path.exists(wp) else np.ones(len(r.y), dtype=np.float32)
        c = np.load(os.path.join(path, "centre.npy"))
        r.centre = c if c.size else None
        r.prov = [json.loads(x) for x in open(os.path.join(path, "provenance.jsonl"))]
        return r
