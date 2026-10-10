"""
Knowledge-gap detection (PLM-1 / GAP-0; J 2026-09-30: "I have this task, my shards don't hold the knowledge or skills, let
me build a new knowledge base shard for this task").

A gap is declared from signals the system already produces — never from a single low score:
  abstain   a ring head answered below its margin threshold (the shadow's frozen tau, else a default)
  novel     the query sits far from every known state / mark (Novelty reader)
  unproved  the logic spine holds no proof for a claim the answer needs (LOGIC tracker)
Per head (domain), over a window: gap when the abstain rate or the novel rate crosses its threshold on >= min_n queries.
The output is a BUILD BRIEF (markdown) naming the domain, the evidence (query ids, rates, nearest precedents) and the
shard pipeline with a verify command per step. Nothing is built automatically: the brief is the reader of this store and
a person (or the fleet procedure) approves the build.
"""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional


@dataclass
class Gap:
    head: str
    n: int
    abstain_rate: float
    novel_rate: float
    unproved: int
    examples: List[str] = field(default_factory=list)       # query / item ids (no text)
    precedents: Dict[str, int] = field(default_factory=dict)  # nearest mark sources among abstained answers -> count

    @property
    def reasons(self) -> List[str]:
        r = []
        if self.abstain_rate > 0:
            r.append(f"abstain {self.abstain_rate:.0%}")
        if self.novel_rate > 0:
            r.append(f"novel {self.novel_rate:.0%}")
        if self.unproved:
            r.append(f"{self.unproved} unproved claims")
        return r


class GapDetector:
    def __init__(self, taus: Optional[Dict[str, float]] = None, default_tau: float = 0.2, abstain_gate: float = 0.30,
                 novel_gate: float = 0.30, novelty_tau: float = 0.6, min_n: int = 20):
        self.taus, self.default_tau = taus or {}, default_tau
        self.abstain_gate, self.novel_gate, self.novelty_tau, self.min_n = abstain_gate, novel_gate, novelty_tau, min_n

    @classmethod
    def from_shadow(cls, shadow_dir: str, **kw) -> "GapDetector":
        """Use the shadow's frozen per-head margin thresholds when they exist (data/ring-shadow/gates.json)."""
        taus = {}
        p = os.path.join(shadow_dir, "gates.json")
        if os.path.exists(p):
            for h, g in json.load(open(p))["heads"].items():
                if "tau" in g:
                    taus[h] = g["tau"]
        return cls(taus=taus, **kw)

    def scan(self, records: Iterable[Dict], unproved: Optional[Dict[str, int]] = None) -> List[Gap]:
        """records: shadow-style dicts {head, item, margin, [novelty], [because]}. Returns gaps, worst first."""
        by = defaultdict(list)
        for r in records:
            if "pred" in r or "margin" in r:
                by[r["head"]].append(r)
        gaps = []
        for h, rs in by.items():
            if len(rs) < self.min_n:
                continue
            tau = self.taus.get(h, self.default_tau)
            abst = [r for r in rs if r.get("margin", 1.0) < tau]
            novel = [r for r in rs if r.get("novelty", 0.0) >= self.novelty_tau]
            a, n = len(abst) / len(rs), len(novel) / len(rs)
            up = (unproved or {}).get(h, 0)
            if a >= self.abstain_gate or n >= self.novel_gate or up > 0:
                prec = defaultdict(int)
                for r in abst:
                    for b in (r.get("because") or [])[:1]:
                        prec[str(b[0]).split(":")[0]] += 1
                gaps.append(Gap(h, len(rs), a, n, up, [str(r["item"]) for r in (abst or novel)[:10]], dict(prec)))
        return sorted(gaps, key=lambda g: -(g.abstain_rate + g.novel_rate))

    def brief(self, gap: Gap) -> str:
        """A shard-build brief for a person to approve (Plan step of the fleet procedure: each step has a verify)."""
        ds = gap.head.split(".")[0]
        return f"""# GAP — {gap.head}: the ring does not hold this domain well enough ({time.strftime('%Y-%m-%d')})
Evidence: {gap.n} queries; {', '.join(gap.reasons)}. Example items: {', '.join(gap.examples) or '-'}.
Nearest precedent sources among abstained answers: {gap.precedents or '-'}.

## Build (each step with its verify)
1. Gather: export the domain's recent sources (bin/chew-export for {ds}; plus any new source the gap names).
   VERIFY: the export's selftest passes and gitleaks drops are reported.
2. Label: teacher labels on the pod (a labelling script, unpublished), few-shot from the head's current classes.
   VERIFY: label validation rate >= the last chew's.
3. Embed with the server's pen (projects/ring-headd/samepen.py). VERIFY: row count == export rows.
4. Write marks / a shard (save_heads.py; omni_ring.shards SEED_GEN). VERIFY: projects/ring-headd/check.py PASS.
5. Re-ask the gap's queries. VERIFY: abstain rate on them below {self.abstain_gate:.0%}; eval accuracy not lower.
"""
