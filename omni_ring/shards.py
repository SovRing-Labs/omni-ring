"""Shards: compiled, memory-mapped ring memories with hot / warm / cold tiers (SHARDS-0, 2026-09-29).

A shard is a directory:
    header.json            magic, version, kind, dim, seed, meta (source range, provenance), row counts, sha256 per array
    slots.npy  (S, 2, D)   int8 re/im of each slot (records superposed; magnitude kept via per-slot scale)
    scales.npy (S,)        float32 slot scale (quantize_slot)
    pool_<name>.npy/.tsv   uint8 phase lattice of the candidate values of one type + their labels
    index_<name>.npy/.tsv  uint8 phase lattice of whole items (e.g. episode gists) for similarity search + labels
Records: key parts k1..kn and a value v from pool p -> slot[crc32(k1) mod S] += V(k1) * ... * V(kn) * V_p(v).
Reads unbind the key and sift the typed pool (AVX2 kernel via omni_ring.sift.scores). Shards are compiled artifacts: the
source log is the truth and a rebuild is a recompile (SLEEP-0: churn leaves no residue, but hand edits are not supported).
Sizing (SCALE-1a): ~100 records per 5,120-d slot for thresholded answers, ~190 for top-1 answers.
Routing is plain metadata (kind / key prefix / time range), never a sift over plate vectors.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import zlib
from collections import OrderedDict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .sift import THRESHOLD, quantize_slot, scores

MAGIC, VERSION = "omniring-shard", 1
K_PHASE = 16
PHASORS = np.exp(1j * 2 * np.pi * np.arange(K_PHASE) / K_PHASE).astype(np.complex64)
_SECRET = re.compile(r"(sk-[A-Za-z0-9_-]{16,}|AIza[0-9A-Za-z_-]{30,}|gh[pousr]_[A-Za-z0-9]{20,}|xox[abpr]-[A-Za-z0-9-]{10,}"
                     r"|-----BEGIN [A-Z ]*PRIVATE KEY-----|(?i:api[_-]?key|token|secret|password)\s*[:=]\s*\S+"
                     r"|[A-Za-z0-9+/=_-]{40,})")


def redact(text: str) -> str:
    """Scrub likely secrets from a label before it is written (labels map hashes back to text: hashing is not hiding)."""
    return _SECRET.sub("[REDACTED]", text.replace("\t", " ").replace("\n", " "))


LEGACY_GEN, SEED_GEN = "pcg64-crc32", "seedv1"


def symbol(label: str, seed: int = 0, dim: int = 5120, gen: str = LEGACY_GEN) -> np.ndarray:
    """Deterministic 4-bit phase indices for a symbol (the 'encoder' is a hash).
    gen="seedv1": SuperSeed v1 portable codewords (omni_ring.seed.codeword) — every new shard (RING-SEED step 3).
    gen="pcg64-crc32": numpy PCG64 seeded by crc32 — shards built before 2026-09-30 (headers without codeword_gen)."""
    if gen == SEED_GEN:
        from .seed import codeword
        return codeword(label, seed, dim)
    return np.random.default_rng((zlib.crc32(label.encode()) ^ seed) & 0xFFFFFFFF).integers(0, K_PHASE, dim, dtype=np.uint8)


def phase_of(v: np.ndarray) -> np.ndarray:
    """Complex vector -> nearest of the 16 phases per dimension (for index lattices of bundles)."""
    return (np.round(np.angle(v) / (2 * np.pi) * K_PHASE).astype(np.int64) % K_PHASE).astype(np.uint8)


def _sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


class ShardWriter:
    """Accumulate records, pools and indexes in RAM, then write one shard directory."""

    def __init__(self, kind: str, dim: int = 5120, seed: int = 0, per_slot: int = 100, meta: Optional[dict] = None,
                 codeword_gen: str = SEED_GEN):
        self.kind, self.dim, self.seed, self.per_slot, self.meta = kind, dim, seed, per_slot, dict(meta or {})
        self.gen = codeword_gen
        self.records: List[Tuple[Tuple[str, ...], str, str]] = []
        self.pools: Dict[str, List[str]] = {}
        self.indexes: Dict[str, Tuple[List[str], List[np.ndarray]]] = {}
        self.extra: Dict[str, Tuple[np.ndarray, Optional[List[str]]]] = {}

    def add_array(self, name: str, array: np.ndarray, labels: Optional[Sequence[str]] = None) -> None:
        """Store any extra array (e.g. random-indexing context vectors) as array_<name>.npy, with optional row labels."""
        self.extra[name] = (np.ascontiguousarray(array), list(labels) if labels is not None else None)

    def vec(self, label: str) -> np.ndarray:
        return PHASORS[symbol(label, self.seed, self.dim, self.gen)]

    def add(self, key: Sequence[str], pool: str, value: str) -> None:
        self.records.append((tuple(key), pool, value))
        self.pools.setdefault(pool, [])

    def add_index_item(self, index: str, label: str, v: np.ndarray) -> None:
        labels, rows = self.indexes.setdefault(index, ([], []))
        labels.append(label); rows.append(phase_of(v))

    def write(self, path: str) -> str:
        os.makedirs(path, exist_ok=True)
        for _, p, v in self.records:
            self.pools[p].append(v)
        pools = {p: sorted(set(vs)) for p, vs in self.pools.items()}
        pos = {p: {v: i for i, v in enumerate(vs)} for p, vs in pools.items()}
        lat = {p: np.stack([symbol(f"{p}:{v}", self.seed, self.dim, self.gen) for v in vs]) if vs else
               np.zeros((0, self.dim), np.uint8) for p, vs in pools.items()}
        n_slots = max(1, -(-len(self.records) // self.per_slot))
        acc = np.zeros((n_slots, self.dim), np.complex64)
        for key, p, v in self.records:
            t = PHASORS[lat[p][pos[p][v]]]
            for k in key:
                t = t * self.vec(k)
            acc[zlib.crc32(key[0].encode()) % n_slots] += t
        slots = np.zeros((n_slots, 2, self.dim), np.int8)
        scales = np.zeros(n_slots, np.float32)
        for i in range(n_slots):
            slots[i, 0], slots[i, 1], scales[i] = quantize_slot(acc[i])
        arrays = {"slots": slots, "scales": scales}
        for p, a in lat.items():
            arrays[f"pool_{p}"] = np.ascontiguousarray(a)
            _write_labels(os.path.join(path, f"pool_{p}.tsv"), pools[p])
        for name, (labels, rows) in self.indexes.items():
            arrays[f"index_{name}"] = np.ascontiguousarray(np.stack(rows))
            _write_labels(os.path.join(path, f"index_{name}.tsv"), labels)
        for name, (a, labels) in self.extra.items():
            arrays[f"array_{name}"] = a
            if labels is not None:
                _write_labels(os.path.join(path, f"array_{name}.tsv"), labels)
        for name, a in arrays.items():
            np.save(os.path.join(path, f"{name}.npy"), a)
        from .seed import SPEC_VERSION
        header = {"magic": MAGIC, "version": VERSION, "kind": self.kind, "dim": self.dim, "seed": self.seed,
                  "codeword_gen": self.gen, "spec_version": SPEC_VERSION if self.gen == SEED_GEN else 0,
                  "per_slot": self.per_slot, "records": len(self.records), "slots": n_slots, "meta": self.meta,
                  "pools": {p: len(v) for p, v in pools.items()},
                  "indexes": {n: len(v[0]) for n, v in self.indexes.items()},
                  "arrays": {n: list(v[0].shape) for n, v in self.extra.items()},
                  "sha256": {name: _sha(a) for name, a in arrays.items()}}
        with open(os.path.join(path, "header.json"), "w") as f:
            json.dump(header, f, indent=1, sort_keys=True)
        return path


def _write_labels(path: str, labels: Iterable[str]) -> None:
    with open(path, "w") as f:
        for x in labels:
            f.write(redact(str(x)) + "\n")


class Shard:
    """One shard opened warm (mmap) or hot (copied into RAM). verify=True checks every array's sha256 on open."""

    def __init__(self, path: str, hot: bool = False, verify: bool = True):
        self.path = path
        with open(os.path.join(path, "header.json")) as f:
            self.header = json.load(f)
        if self.header.get("magic") != MAGIC:
            raise ValueError(f"not an omni-ring shard: {path}")
        self.dim, self.seed = self.header["dim"], self.header["seed"]
        self.gen = self.header.get("codeword_gen", LEGACY_GEN)      # shards from before 2026-09-30 carry no field
        self.arrays: Dict[str, np.ndarray] = {}
        for name, digest in self.header["sha256"].items():
            a = np.load(os.path.join(path, f"{name}.npy"), mmap_mode="r")
            if verify and _sha(a) != digest:
                raise ValueError(f"shard {path}: {name}.npy failed its sha256 check")
            self.arrays[name] = np.array(a) if hot else a
        self.hot = hot
        self._labels: Dict[str, List[str]] = {}

    @property
    def nbytes(self) -> int:
        return int(sum(a.nbytes for a in self.arrays.values()))

    def labels(self, name: str) -> List[str]:
        if name not in self._labels:
            with open(os.path.join(self.path, f"{name}.tsv")) as f:
                self._labels[name] = [x.rstrip("\n") for x in f]
        return self._labels[name]

    def vec(self, label: str) -> np.ndarray:
        return PHASORS[symbol(label, self.seed, self.dim, self.gen)]

    def slot(self, key0: str) -> np.ndarray:
        i = zlib.crc32(key0.encode()) % self.header["slots"]
        s = self.arrays["slots"][i]
        return (s[0].astype(np.float32) + 1j * s[1].astype(np.float32)) * float(self.arrays["scales"][i])

    def read_scores(self, key: Sequence[str], pool: str) -> np.ndarray:
        u = self.slot(key[0])
        for k in key:
            u = u * np.conj(self.vec(k))
        return scores(u.astype(np.complex64), self.arrays[f"pool_{pool}"])

    def read(self, key: Sequence[str], pool: str, threshold: float = THRESHOLD, top: int = 3) -> List[Tuple[str, float]]:
        sc = self.read_scores(key, pool)
        order = [int(i) for i in np.argsort(-sc)[:top] if sc[i] >= threshold]
        return [(self.labels(f"pool_{pool}")[i], float(sc[i])) for i in order]

    def search(self, index: str, query: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        q = PHASORS[phase_of(query)]
        sc = scores(q, self.arrays[f"index_{index}"])
        return [(self.labels(f"index_{index}")[int(i)], float(sc[i])) for i in np.argsort(-sc)[:k]]


class Library:
    """A directory of shards. Router = header metadata. Tiers: hot (RAM copy, LRU under budget_bytes) / warm (mmap) / cold
    (not opened). Pinned shards are never demoted."""

    def __init__(self, root: str, budget_bytes: int = 512 * 2 ** 20):
        self.root, self.budget = root, budget_bytes
        self.hot: "OrderedDict[str, Shard]" = OrderedDict()
        self.warm: Dict[str, Shard] = {}
        self.pinned: set = set()

    def names(self) -> List[str]:
        return sorted(d for d in os.listdir(self.root) if os.path.exists(os.path.join(self.root, d, "header.json")))

    def route(self, kind: Optional[str] = None, **meta) -> List[str]:
        out = []
        for n in self.names():
            with open(os.path.join(self.root, n, "header.json")) as f:
                h = json.load(f)
            if (kind is None or h["kind"] == kind) and all(h["meta"].get(k) == v for k, v in meta.items()):
                out.append(n)
        return out

    def tier(self, name: str) -> str:
        return "hot" if name in self.hot else "warm" if name in self.warm else "cold"

    def get(self, name: str, hot: bool = False, pin: bool = False) -> Shard:
        if pin:
            self.pinned.add(name); hot = True
        if name in self.hot:
            self.hot.move_to_end(name)
            return self.hot[name]
        if not hot:
            if name not in self.warm:
                self.warm[name] = Shard(os.path.join(self.root, name))
            return self.warm[name]
        s = Shard(os.path.join(self.root, name), hot=True)
        self.warm.pop(name, None)
        self.hot[name] = s
        self._evict()
        return s

    def hot_bytes(self) -> int:
        return sum(s.nbytes for s in self.hot.values())

    def _evict(self) -> None:
        for name in list(self.hot):
            if self.hot_bytes() <= self.budget:
                break
            if name in self.pinned or len(self.hot) == 1:
                continue
            self.hot.pop(name)                                   # demote to warm (already verified on its hot open)
            self.warm[name] = Shard(os.path.join(self.root, name), verify=False)
