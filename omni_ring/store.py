"""APP-0 — the memory-does-the-work store (2026-10-02). Three primitives on one file-backed, mmap-opened store:
  address  : a structural key's residues (3·5·7·13 of hash(key)) are its RAM address; put/get is O(1), no scan (MEM-0 A).
  slot     : one product-encoded superposed slot per group holds up to ~128 ids; one read returns them all (BUS-PEEL).
  lattice  : 4-bit packed phase marks (hash / random-indexing text encoder, no model) with a prefix cascade (MEM-0 C).
Hierarchy: address the slot or shard first, then peel / scan inside it."""
from __future__ import annotations
import hashlib, json, os, re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
from . import seed
from .types import DEFAULT_SPEC
from .codebook import generate_rotations
from .chain import encode
from .peel import read_all_kernel

MODS = (113, 127, 131); NADDR = 113 * 127 * 131      # 1,879,981 addresses (the 3-ring space proved by SIEVE-0 / PART)
_TOK = re.compile(r"[a-z0-9]{2,}")


def residue_address(key: str) -> int:
    h = int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "big")
    return h % NADDR                              # == CRT of (h mod 3, h mod 5, h mod 7, h mod 13)


_PH: Dict[str, np.ndarray] = {}   # gram -> uint8 phase indices (5 KB each; vocabulary repeats across files); capped
_PH_CAP = 40_000
MAX_GRAMS = 4000                     # ~200 MB max; cleared when full (the OOM of 2026-10-02 came from caching 40 KB complex rows)


def _gram_phases(grams: Sequence[str], d: int) -> np.ndarray:
    """Vectorised random indexing: one 32-bit hash per gram, a multiplicative hash over (gram, dim) -> 16-phase codewords
    (n x d uint8). One uint32 multiply + shift per cell (the 2026-10-02 profile: the uint64 splitmix was 3x the bundling)."""
    h = np.array([int.from_bytes(hashlib.blake2b(g.encode(), digest_size=4).digest(), "big") for g in grams], dtype=np.uint32)
    kd = (np.arange(d, dtype=np.uint32) * np.uint32(2654435761) + np.uint32(0x9E3779B9))
    x = (h[:, None] ^ kd[None, :]) * np.uint32(0x85EBCA6B)
    x ^= x >> np.uint32(13); x *= np.uint32(0xC2B2AE35); x ^= x >> np.uint32(16)
    return ((x >> np.uint32(28)) & np.uint32(15)).astype(np.uint8)


def text_vector(text: str, d: int = seed.D) -> np.ndarray:
    """Random-indexing text encoder: bundle the codewords of word unigrams + bigrams (no model), vectorised."""
    toks = _TOK.findall(text.lower())[:4000]
    uni = list(dict.fromkeys(toks)); bi = list(dict.fromkeys(a + "_" + b for a, b in zip(toks, toks[1:])))
    grams = (uni + bi)[:MAX_GRAMS]                            # cap: encoding cost is n_grams x d (APP-0 profile)
    if not grams:
        return np.zeros(d, dtype=np.uint8)
    if len(_PH) > 50_000:                       # cap the cache (~250 MB of uint8 phases); vocabulary beyond that is recomputed
        _PH.clear()
    new = [g for g in grams if g not in _PH]
    if new:
        for g, row in zip(new, _gram_phases(new, d)):
            _PH[g] = row                        # uint8 phase indices: 5 KB per gram
    ph = np.stack([_PH[g] for g in grams])
    acc = np.zeros(d, dtype=np.complex64)
    for a in range(0, len(ph), 2048):           # bounded working set
        acc += np.exp(2j * np.pi * ph[a:a + 2048] / 16).astype(np.complex64).sum(0)
    return seed.sector16(np.rint(acc.real * 127).astype(np.int64), np.rint(acc.imag * 127).astype(np.int64))


class Store:
    def __init__(self, path: str, d: int = seed.D):
        self.path = path; self.d = d; os.makedirs(path, exist_ok=True)
        self.meta_p = os.path.join(path, "meta.json")
        self.meta = json.load(open(self.meta_p)) if os.path.exists(self.meta_p) else {"ids": [], "groups": {}, "payload": {}}
        self.rot = generate_rotations(DEFAULT_SPEC, seed=0)
        self._lat = None; self._slots: Dict[str, np.ndarray] = {}
        lp = os.path.join(path, "lattice.npy")
        if os.path.exists(lp):
            self._lat = np.load(lp, mmap_mode="r")                    # zero-copy open
        sp = os.path.join(path, "slots.npz")
        if os.path.exists(sp):
            z = np.load(sp); keys = self.meta.get("_slot_keys", list(z.files))
            self._slots = {k: z[f"s{i}"] for i, k in enumerate(keys)}
        self._pending: List[np.ndarray] = []

    # -- address: O(1) --------------------------------------------------------------------------------------------------
    def put_at(self, key: str, payload: dict) -> int:
        a = residue_address(key); self.meta["payload"][str(a)] = {"key": key, **payload}; return a

    def get(self, key: str) -> Optional[dict]:
        return self.meta["payload"].get(str(residue_address(key)))

    # -- slots: 128 ids per read ----------------------------------------------------------------------------------------
    def slot_add(self, group: str, item_id: int) -> None:
        v = self._slots.get(group)
        self._slots[group] = (v if v is not None else np.zeros(self.d, dtype=np.complex64)) + encode(item_id % DEFAULT_SPEC.data_capacity, self.rot).astype(np.complex64)
        self.meta["groups"].setdefault(group, []).append(item_id)

    def slot_read(self, group: str) -> List[int]:
        v = self._slots.get(group)
        return [] if v is None else read_all_kernel(v, self.rot, max_marks=160)

    # -- lattice: packed marks + cascade search ---------------------------------------------------------------------------
    def add_text(self, item_id: int, text: str) -> None:
        self._pending.append(text_vector(text, self.d)); self.meta["ids"].append(item_id)

    def commit(self) -> None:
        if self._pending:
            new = seed.pack4(np.stack(self._pending)); self._pending = []
            lat = new if self._lat is None else np.vstack([np.asarray(self._lat), new])
            np.save(os.path.join(self.path, "lattice.npy"), lat); self._lat = np.load(os.path.join(self.path, "lattice.npy"), mmap_mode="r")
        if self._slots:
            # np.savez(file, **kwds): a group literally named "file" (a legal slot key -- e.g. the Weaver
            # field's mark kind "file") collides with savez's own positional parameter name. Keys go in as
            # safe synthetic names (s0, s1, ...); the real names ride in meta["_slot_keys"] (OS-RING-0, 2026-10-02).
            keys = list(self._slots.keys())
            self.meta["_slot_keys"] = keys
            np.savez(os.path.join(self.path, "slots.npz"), **{f"s{i}": self._slots[k] for i, k in enumerate(keys)})
        json.dump(self.meta, open(self.meta_p, "w"))

    def search(self, query: str, k: int = 5, cascade: bool = True) -> List[Tuple[int, int]]:
        if self._lat is None:
            return []
        q = text_vector(query, self.d); lat = np.ascontiguousarray(self._lat)
        if cascade and len(lat) > 256:
            coarse = seed.sift_batch_p4(q[None, :], lat, d_used=self.d // 4)[0]
            top = np.argsort(-coarse)[: max(64, 4 * k)]; fine = seed.sift_rows_p4(q, lat, top)
            order = top[np.argsort(-fine)][:k]; scores = np.sort(fine)[::-1][:k]
        else:
            sc = seed.sift_batch_p4(q[None, :], lat)[0]; order = np.argsort(-sc)[:k]; scores = sc[order]
        return [(self.meta["ids"][int(i)], int(s)) for i, s in zip(order, scores)]
