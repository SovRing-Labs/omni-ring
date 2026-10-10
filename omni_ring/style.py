"""
Style as measurable features, learned from what worked (PLM-3; J 2026-09-30 PLM.txt: "constantly improve its speech
algorithm based on what worked for it that day").

  features(text)   sentence-length mean and variation (the "wavelengths of breath"), paragraph-length variation,
                   words per sentence, a sonority proxy (share of vowels + sonorants l m n r w y among letters, an
                   ORTHOGRAPHIC stand-in for the sonority hierarchy, not phonetics), plosive share, list/table density,
                   and Flesch reading ease (syllables by vowel-group heuristic).
  StyleMemory      marks = torus frames of (context features) with the style preset used and the reward it earned; read =
                   the preset whose close neighbours earned the most (GearMemory — the governor's learned table, reused).
                   Abstains (returns None) when nothing close is known. Rewards come from the reader: accepted as-is = 1,
                   else 1 - normalised edit distance to the final version, minus a follow-up penalty.
Grounding: sentence-length variation and readability are standard stylometry; speech rhythm tracking in the 4-8 Hz band
is real neuroscience; golden-ratio sentence lengths are NOT an established law and are not encoded here.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Sequence

import numpy as np

from .telemetry import GearMemory, TorusEncoder

SONORANT = set("aeiouylmnrw")
PLOSIVE = set("ptkbdg")
FIELDS = [("sent_mean", 0, 40), ("sent_cv", 0, 1.5), ("para_cv", 0, 1.5), ("sonority", 0.4, 0.8), ("plosive", 0.0, 0.3),
          ("list_density", 0, 1), ("reading_ease", -20, 100)]
# Prosody of prose (J 2026-09-30: "pauses, asymmetry, conclusions landed, buildup and closure, rest and movement").
# Measured in the "Golden ratio PLM" essay: breath groups median 6 words; 19/28 sentences land on a shorter final phrase;
# 5 long->short rests; 6 antithesis frames; the close reuses the opening's key words (ring composition).
PROSODY_FIELDS = [("breath_median", 2, 14), ("breath_spread", 0, 12), ("landing", 0, 1), ("rests", 0, 0.4),
                  ("antithesis", 0, 0.3), ("triads", 0, 0.2), ("ring", 0, 0.6)]
_ANTI = re.compile(r"\b(rather than|instead of|not (?:just|only|merely)|doesn't just|no longer|but rather|stops? [a-z]+ing)\b", re.I)
_TRIAD = re.compile(r"\b[\w'-]+(?: [\w'-]+){0,3}, [\w'-]+(?: [\w'-]+){0,3},? and [\w'-]+", re.I)
_PHRASE = re.compile(r"[,;:\u2014()]|\s-\s|--")


LISTISH = re.compile(r"\s*(\||[-*]\s|\d+[.)]\s)")        # table rows (any "|" line), bullets, numbered items


def _syllables(w: str) -> int:
    return max(1, len(re.findall(r"[aeiouy]+", w.lower())))


def features(text: str) -> Dict[str, float]:
    paras = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    lines = [l for l in text.splitlines() if l.strip()]
    listish = sum(bool(LISTISH.match(l)) for l in lines)
    prose = " ".join(l for l in lines if not LISTISH.match(l))
    sents = [s for s in re.split(r"(?<=[.!?])\s+", prose) if re.search(r"\w", s)]
    words = re.findall(r"[A-Za-z']+", text)
    lens = np.array([len(re.findall(r"[A-Za-z']+", s)) for s in sents] or [0], float)
    plen = np.array([len(re.findall(r"[A-Za-z']+", p)) for p in paras] or [0], float)
    letters = [c for c in text.lower() if c.isalpha()]
    nw, ns = max(len(words), 1), max(len(sents), 1)
    syl = sum(_syllables(w) for w in words)
    return {"sent_mean": float(lens.mean()), "sent_cv": float(lens.std() / max(lens.mean(), 1e-9)),
            "para_cv": float(plen.std() / max(plen.mean(), 1e-9)),
            "sonority": sum(c in SONORANT for c in letters) / max(len(letters), 1),
            "plosive": sum(c in PLOSIVE for c in letters) / max(len(letters), 1),
            "list_density": listish / max(len(lines), 1),
            "reading_ease": 206.835 - 1.015 * (nw / ns) - 84.6 * (syl / nw)}


def prosody(text: str) -> Dict[str, float]:
    """Breath groups (phrase lengths between , ; : dashes), landing (share of multi-phrase sentences whose final phrase is
    shorter than the first: long build, short landing), rests (share of transitions from a >= 25-word sentence to a
    <= 12-word one), antithesis and triad frames per sentence, ring (share of the opening's content words that return in
    the close). All measurable, none a fixed ratio: StyleMemory learns which combinations a reader rewards."""
    flat = " ".join(l for l in text.splitlines() if not LISTISH.match(l))
    sents = [s for s in re.split(r"(?<=[.!?])\s+", " ".join(flat.split())) if len(re.findall(r"[A-Za-z']+", s)) > 2]
    W = lambda t: len(re.findall(r"[A-Za-z']+", t))
    phr = [W(p) for s in sents for p in _PHRASE.split(s) if W(p)] or [0]
    multi = [[W(p) for p in _PHRASE.split(s) if W(p)] for s in sents]
    multi = [m for m in multi if len(m) > 1]
    lens = [W(s) for s in sents]
    trans = max(len(lens) - 1, 1)
    words = re.findall(r"[a-z]{5,}", flat.lower())
    k = max(len(words) // 8, 1)
    head, tail = set(words[:k]), set(words[-k:])
    n = max(len(sents), 1)
    return {"breath_median": float(np.median(phr)), "breath_spread": float(np.percentile(phr, 90) - np.percentile(phr, 10)),
            "landing": sum(m[-1] < m[0] for m in multi) / max(len(multi), 1),
            "rests": sum(lens[i - 1] >= 25 and lens[i] <= 12 for i in range(1, len(lens))) / trans,
            "antithesis": len(_ANTI.findall(flat)) / n, "triads": len(_TRIAD.findall(flat)) / n,
            "ring": len(head & tail) / max(len(head), 1)}


def reward(draft: str, final: Optional[str] = None, follow_up: bool = False) -> float:
    """1 = accepted as-is; else similarity of the draft to what the reader kept; a clarifying follow-up costs 0.3."""
    r = 1.0 if final is None or final == draft else SequenceMatcher(None, draft, final).ratio()
    return r - (0.3 if follow_up else 0.0)


class StyleMemory:
    """Learned style table: context (features of the request / conversation) -> preset, from outcomes."""

    def __init__(self, presets: Sequence[str], d: int = 2048, path: Optional[str] = None, **kw):
        self.presets = list(presets)
        self.enc = TorusEncoder(FIELDS, d=d)
        self.mem = GearMemory(len(self.presets), d=d, path=path, **kw)

    def _ph(self, ctx: Dict[str, float]) -> np.ndarray:
        return self.enc.phases(self.enc.encode(ctx))

    def write(self, ctx: Dict[str, float], preset: str, r: float) -> None:
        self.mem.write(self._ph(ctx), self.presets.index(preset), r)

    def choose(self, ctx: Dict[str, float]) -> Optional[str]:
        g, _ = self.mem.read(self._ph(ctx))
        return None if g is None else self.presets[g]
