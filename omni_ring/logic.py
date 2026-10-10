"""
Logic tracking on the ring (LOGIC-0; J 2026-09-30: "a lack of logic and complex logical tracking").

Symbolic spine + holographic muscles:
  Tracker    exact working memory of ground facts, Horn rules with shared variables, forward chaining to a fixpoint, and a
             JUSTIFICATION graph (every belief records every way it was derived: rule + premise facts). From the graph:
             proof(fact) = the tree of why it holds; depth = shortest proof; contradictions = a belief and its explicit
             negation both held (both sides' reasons returned); retract(premise) = DRed (delete, then re-derive what still
             has support), so beliefs change exactly when their reasons do (a truth-maintenance system, Doyle 1979).
  Lexicon    soft unification (Rocktaschel & Riedel 2017, in spirit): a surface name ("is_parent_of", "parents") is encoded
             from its character trigrams as a SuperSeed torus vector and sifted against known surface forms; the answer is
             the canonical predicate, with a margin. Low similarity or low margin -> abstain (None), never a guess.
The ring does what interference is good at (fuzzy matching, retrieval, confidence); the spine does what it is not (exact
variable binding, chaining without decay, remembering why). Facts: tuples (pred, *args); negation: ("not", pred, *args).
Rules: (head, [body...]) where atoms are tuples and strings starting with an upper-case letter are variables.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from . import seed as S

Atom = Tuple[str, ...]
Rule = Tuple[Atom, List[Atom]]


def _is_var(t: str) -> bool:
    return t[:1].isupper()


def _match(pattern: Atom, fact: Atom, env: Dict[str, str]) -> Optional[Dict[str, str]]:
    if len(pattern) != len(fact) or pattern[0] != fact[0]:
        return None
    out = dict(env)
    for p, f in zip(pattern[1:], fact[1:]):
        if _is_var(p):
            if out.setdefault(p, f) != f:
                return None
        elif p != f:
            return None
    return out


def _subst(a: Atom, env: Dict[str, str]) -> Atom:
    return (a[0],) + tuple(env.get(t, t) for t in a[1:])


class Tracker:
    def __init__(self, rules: Iterable[Rule] = ()):
        self.rules: List[Rule] = list(rules)
        self.base: Set[Atom] = set()                              # asserted premises
        self.facts: Set[Atom] = set()                             # current beliefs (base + derived)
        self.just: Dict[Atom, Set[Tuple[int, Tuple[Atom, ...]]]] = defaultdict(set)   # fact -> {(rule, premises)}
        self.by_pred: Dict[str, Set[Atom]] = defaultdict(set)

    # -- beliefs -----------------------------------------------------------------------------------------------------
    def _add(self, f: Atom) -> bool:
        if f in self.facts:
            return False
        self.facts.add(f); self.by_pred[f[0]].add(f)
        return True

    def _drop(self, f: Atom) -> None:
        self.facts.discard(f); self.by_pred[f[0]].discard(f)

    def assert_fact(self, f: Atom) -> None:
        self.base.add(f); self._add(f)
        self.chain()

    def _fire(self, ri: int, delta: Optional[Set[Atom]] = None) -> List[Tuple[Atom, Tuple[Atom, ...]]]:
        """All (conclusion, premises) of rule ri over current beliefs; with delta, at least one premise must be in delta."""
        head, body = self.rules[ri]
        out = []

        def walk(i, env, used, touched):
            if i == len(body):
                if delta is None or touched:
                    out.append((_subst(head, env), tuple(used)))
                return
            pat = body[i]
            cands = self.by_pred.get(pat[0], ())
            for f in list(cands):
                e = _match(pat, f, env)
                if e is not None:
                    walk(i + 1, e, used + [f], touched or (delta is not None and f in delta))

        walk(0, {}, [], False)
        return out

    def chain(self, delta: Optional[Set[Atom]] = None) -> int:
        """Semi-naive forward chaining to a fixpoint; records every justification. Returns the number of new beliefs."""
        new_total = 0
        delta = set(self.facts) if delta is None else set(delta)
        while delta:
            nxt: Set[Atom] = set()
            for ri in range(len(self.rules)):
                for concl, prem in self._fire(ri, delta):
                    self.just[concl].add((ri, prem))
                    if self._add(concl):
                        nxt.add(concl)
            new_total += len(nxt)
            delta = nxt
        return new_total

    # -- truth maintenance ---------------------------------------------------------------------------------------------
    def retract(self, f: Atom) -> Set[Atom]:
        """Remove a premise; DRed: over-delete everything that depended on it, then re-derive what still has support.
        Returns the set of beliefs lost."""
        if f not in self.base:
            return set()
        self.base.discard(f)
        before = set(self.facts)
        # over-delete: transitive dependents via justifications
        dependents = defaultdict(set)
        for c, js in self.just.items():
            for _, prem in js:
                for p in prem:
                    dependents[p].add(c)
        gone, stack = {f}, [f]
        while stack:
            x = stack.pop()
            for c in dependents.get(x, ()):
                if c not in gone and c not in self.base:
                    gone.add(c); stack.append(c)
        for g in gone:
            self._drop(g)
        for c in list(self.just):
            self.just[c] = {j for j in self.just[c] if not (set(j[1]) & gone)}
        # re-derive: anything still derivable from what remains
        self.chain()
        return before - self.facts

    def contradictions(self) -> List[Tuple[Atom, Atom]]:
        """(p, not-p) pairs both believed."""
        return [(f[1:], f) for f in self.by_pred.get("not", ()) if tuple(f[1:]) in self.facts]

    # -- explanation -----------------------------------------------------------------------------------------------------
    def depth(self, f: Atom, _memo=None) -> float:
        _memo = {} if _memo is None else _memo
        if f in _memo:
            return _memo[f]
        if f in self.base:
            return 0
        if f not in self.facts:
            return float("inf")
        _memo[f] = float("inf")                                  # cycle guard
        best = min((1 + max((self.depth(p, _memo) for p in prem), default=0) for _, prem in self.just.get(f, ())),
                   default=float("inf"))
        _memo[f] = best
        return best

    def proof(self, f: Atom, _seen=None) -> Dict:
        """Shortest-depth proof tree: {fact, rule, premises:[...]} (base facts are leaves)."""
        _seen = set() if _seen is None else _seen
        if f in self.base or f in _seen:
            return {"fact": f, "given": f in self.base}
        _seen = _seen | {f}
        js = sorted(self.just.get(f, ()), key=lambda j: max((self.depth(p) for p in j[1]), default=0))
        if not js:
            return {"fact": f, "unsupported": True}
        ri, prem = js[0]
        return {"fact": f, "rule": ri, "premises": [self.proof(p, _seen) for p in prem]}


def _ground(a: Atom) -> bool:
    return not any(_is_var(t) for t in a[1:])


class Prover:
    """Goal-directed (backward) chaining over a Tracker's BASE facts — the progressive, frame-first direction: lock the
    end (the goal) and the start (the given facts), fill the middle. Returns a proof tree or None (no proof: never a guess).
    Memoised; a goal already on the current path counts as unproved (cycle guard). `work` counts rule instances tried."""

    def __init__(self, tracker: "Tracker", max_depth: int = 16):
        self.t, self.max_depth, self.work = tracker, max_depth, 0
        self.memo: Dict[Atom, Optional[Dict]] = {}

    def prove(self, goal: Atom, _path: Tuple[Atom, ...] = ()) -> Optional[Dict]:
        if goal in self.t.base:
            return {"fact": goal, "given": True}
        if goal in self.memo:
            return self.memo[goal]
        if goal in _path or len(_path) >= self.max_depth:
            return None
        path = _path + (goal,)
        for ri, (head, body) in enumerate(self.t.rules):
            env = _match(head, goal, {})
            if env is None:
                continue
            self.work += 1
            sub = self._solve(body, 0, env, path)
            if sub is not None:
                self.memo[goal] = {"fact": goal, "rule": ri, "premises": sub}
                return self.memo[goal]
        self.memo[goal] = None
        return None

    def _solve(self, body: List[Atom], i: int, env: Dict[str, str], path) -> Optional[List[Dict]]:
        if i == len(body):
            return []
        atom = _subst(body[i], env)
        if _ground(atom):
            p = self.prove(atom, path)
            if p is None:
                return None
            rest = self._solve(body, i + 1, env, path)
            return None if rest is None else [p] + rest
        for f in sorted(self.t.by_pred.get(atom[0], ())):          # bind free variables from given facts, in order
            if f not in self.t.base:
                continue
            e = _match(atom, f, env)
            if e is None:
                continue
            rest = self._solve(body, i + 1, e, path)
            if rest is not None:
                return [{"fact": f, "given": True}] + rest
        if any(h[0] == atom[0] for h, _ in self.t.rules):          # derived predicate: try each known constant
            free = [v for v in atom[1:] if _is_var(v)]
            if len(free) == 1:
                for c in self._constants():
                    e = dict(env); e[free[0]] = c
                    g = _subst(atom, e)
                    if g in self.t.base:
                        continue                                    # already tried above
                    p = self.prove(g, path)
                    if p is None:
                        continue
                    rest = self._solve(body, i + 1, e, path)
                    if rest is not None:
                        return [p] + rest
        return None

    def _constants(self) -> List[str]:
        if not hasattr(self, "_consts"):
            self._consts = sorted({a for f in self.t.base for a in f[1:]})
        return self._consts


def check_proof(tracker: "Tracker", tree: Dict) -> bool:
    """Independent re-verification: every leaf is a base fact; every step instantiates its rule exactly."""
    if tree.get("given"):
        return tree["fact"] in tracker.base
    if "rule" not in tree:
        return False
    head, body = tracker.rules[tree["rule"]]
    env = _match(head, tree["fact"], {})
    if env is None or len(body) != len(tree["premises"]):
        return False
    for pat, sub in zip(body, tree["premises"]):
        env = _match(pat, sub["fact"], env)
        if env is None or not check_proof(tracker, sub):
            return False
    return True


# -- soft unification: surface names -> canonical predicates on the ring --------------------------------------------
def trigram_vector(name: str, d: int = S.D, seed: int = 0) -> np.ndarray:
    s = f"#{name.lower()}#"
    grams = [s[i:i + 3] for i in range(len(s) - 2)] or [s]
    PH = np.exp(2j * np.pi * np.arange(S.K) / S.K)
    v = sum(PH[S.codeword("TRI:" + g, seed, d)] for g in grams)
    return (np.round(np.angle(v) / (2 * np.pi / S.K)).astype(np.int64) % S.K).astype(np.uint8)


class Lexicon:
    def __init__(self, d: int = 2048, min_sim: float = 0.30, min_margin: float = 0.05):
        self.d, self.min_sim, self.min_margin = d, min_sim, min_margin
        self.rows: List[np.ndarray] = []; self.canon: List[str] = []; self._lat = None

    def add(self, surface: str, canonical: str) -> None:
        self.rows.append(trigram_vector(surface, self.d)); self.canon.append(canonical); self._lat = None

    def calibrate(self, known: List[Tuple[str, str]], unknown: List[str], grid=None) -> float:
        """Fit min_sim on a calibration set (LOGIC-0b): known = (surface, canonical) pairs NOT in the lexicon, unknown =
        names that must abstain. Picks the threshold maximising resolved + abstained (ties -> the lower threshold)."""
        grid = np.round(np.arange(0.20, 0.801, 0.01), 2) if grid is None else grid
        keep = self.min_sim
        self.min_sim = -1.0
        kr = [(self.resolve(s)[0] == c, *self.resolve(s)[1:]) for s, c in known]
        ur = [self.resolve(s)[1:] for s in unknown]

        def total(t):
            return (sum(ok and s1 >= t and m >= self.min_margin for ok, s1, m in kr) / max(len(kr), 1)
                    + sum(s1 < t or m < self.min_margin for s1, m in ur) / max(len(ur), 1))
        self.min_sim = float(max(grid, key=lambda t: (total(t), -t))) if (kr or ur) else keep
        return self.min_sim

    def resolve(self, surface: str) -> Tuple[Optional[str], float, float]:
        """-> (canonical or None to abstain, best similarity, margin to the best OTHER canonical)."""
        if self._lat is None:
            self._lat = np.ascontiguousarray(np.stack(self.rows))
        sc = S.sift_phase(trigram_vector(surface, self.d), self._lat) / float(127 * 127 * self.d)
        best = {}
        for c, s in zip(self.canon, sc):
            best[c] = max(best.get(c, -1.0), float(s))
        ranked = sorted(best.items(), key=lambda kv: -kv[1])
        top, s1 = ranked[0]
        s2 = ranked[1][1] if len(ranked) > 1 else 0.0
        if s1 < self.min_sim or s1 - s2 < self.min_margin:
            return None, s1, s1 - s2
        return top, s1, s1 - s2
