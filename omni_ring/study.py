"""
Study a format / style (STUDY-0; J 2026-09-30: "have it study a format, style, or language and auto-optimise").

  FormatProfile.study(docs)  learns, from examples only (no gradient):
      profile    mean / spread of style.features plus structure features (headings, bold labels, tables per line)
      template   structural tokens present in >= `support` of the examples (normalised headings and "**Label:**"
                 fields) — what a document of this format must contain
      exemplars  the examples' feature frames on the torus (TorusEncoder), for nearest-example retrieval
  score(doc)   how well a document fits (template coverage + feature closeness); used to recognise formats
  check(doc)   which required template elements are missing (the revise step of draft -> measure -> revise)
  save / load  a PERSONA/FORMAT SHARD on disk (profile.json + exemplars.npy): the head's grown "parameters", swapped in
               when needed — data, not weights (J: "stored in a shard ... swaps them out as needed").
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Dict, List, Optional, Sequence

import numpy as np

from .style import features
from .telemetry import TorusEncoder

STRUCT_FIELDS = [("headings_per_line", 0, 0.5), ("labels_per_line", 0, 0.5), ("table_rows_per_line", 0, 1)]


def _norm(t: str) -> str:
    t = re.sub(r"[`*_#>\[\]()]", " ", t.lower())
    t = re.sub(r"\d[\d.:\-/tz]*", " ", t)                    # dates, versions, counters
    return " ".join(t.split())[:40]


def structure_tokens(doc: str) -> List[str]:
    toks = []
    for line in doc.splitlines():
        m = re.match(r"\s*(#{1,4})\s+(.*)", line)
        if m:
            toks.append(f"H{len(m.group(1))}:{_norm(m.group(2)).split(' — ')[0][:30]}")
        for lab in re.findall(r"\*\*([^*]{2,30}?):?\*\*:?", line):
            toks.append(f"L:{_norm(lab)}")
        if re.match(r"\s*\|", line):
            toks.append("T:table")
    return [t for t in toks if not t.endswith(":")]


def doc_features(doc: str) -> Dict[str, float]:
    lines = max(len([l for l in doc.splitlines() if l.strip()]), 1)
    f = features(doc)
    f["headings_per_line"] = len(re.findall(r"(?m)^\s*#{1,4}\s", doc)) / lines
    f["labels_per_line"] = len(re.findall(r"\*\*[^*]{2,30}?:?\*\*", doc)) / lines
    f["table_rows_per_line"] = len(re.findall(r"(?m)^\s*\|", doc)) / lines
    return f


class FormatProfile:
    def __init__(self, name: str, support: float = 0.6, d: int = 1024):
        from .style import FIELDS
        self.name, self.support, self.d = name, support, d
        self.fields = FIELDS + STRUCT_FIELDS
        self.enc = TorusEncoder(self.fields, d=d)
        self.mean: Dict[str, float] = {}; self.std: Dict[str, float] = {}
        self.template: List[str] = []
        self.exemplars = np.zeros((0, d), np.uint8)

    def study(self, docs: Sequence[str]) -> "FormatProfile":
        feats = [doc_features(d) for d in docs]
        for k in feats[0]:
            v = np.array([f[k] for f in feats]); self.mean[k] = float(v.mean()); self.std[k] = float(v.std() + 1e-6)
        counts = Counter(t for d in docs for t in set(structure_tokens(d)))
        self.template = sorted(t for t, c in counts.items() if c / len(docs) >= self.support)
        self.exemplars = np.stack([self.enc.phases(self.enc.encode(f)) for f in feats])
        return self

    def coverage(self, doc: str) -> float:
        if not self.template:
            return 1.0
        have = set(structure_tokens(doc))
        return sum(t in have for t in self.template) / len(self.template)

    def closeness(self, doc: str) -> float:
        f = doc_features(doc)
        z = np.mean([min(abs(f[k] - self.mean[k]) / self.std[k], 6.0) for k in self.mean])
        return float(np.exp(-z / 2))

    def score(self, doc: str) -> float:
        return 0.6 * self.coverage(doc) + 0.4 * self.closeness(doc)

    def check(self, doc: str) -> List[str]:
        """Required template elements missing from the document (what the revise step must add)."""
        have = set(structure_tokens(doc))
        return [t for t in self.template if t not in have]

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        json.dump({"kind": "format-shard", "name": self.name, "support": self.support, "d": self.d, "mean": self.mean,
                   "std": self.std, "template": self.template, "fields": [f[0] for f in self.fields],
                   "codewords": "seedv1"}, open(os.path.join(path, "profile.json"), "w"), indent=1)
        np.save(os.path.join(path, "exemplars.npy"), self.exemplars)

    @classmethod
    def load(cls, path: str) -> "FormatProfile":
        j = json.load(open(os.path.join(path, "profile.json")))
        p = cls(j["name"], j["support"], j["d"])
        p.mean, p.std, p.template = j["mean"], j["std"], j["template"]
        p.exemplars = np.load(os.path.join(path, "exemplars.npy"))
        return p


def recognise(doc: str, profiles: Sequence[FormatProfile]) -> Optional[str]:
    return max(profiles, key=lambda p: p.score(doc)).name if profiles else None
