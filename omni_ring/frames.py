"""
Frames — recursive reduced representations on the ring (Plate 1995 §V.B, V.E, VII.B, X; RING-SEED plan, FRAME-0).

A frame is ONE vector of the ring's dimension:
    frame = label + sum_i role_i (*) filler_i
where (*) is binding (phase addition), and a filler may itself be a frame. So one vector unfolds into a tree of vectors of
the same size: unbind a role (conjugate — exact for our unit-magnitude phasors, Plate §VIII.C "unitary vectors"), then
clean up the noisy result against an item memory that holds atoms AND frames (chunks). A cleaned-up sub-frame can be
unfolded again. Clean-up is the int8 AVX2 sift over the item memory's phase lattice.

Everything is built from SuperSeed v1 codewords (omni_ring.seed.codeword), so frames are portable across dialects.
Normalisation (Plate §X.E): frames are kept as complex sums; the item memory stores their 16-phase projection.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from . import seed as S

PH = np.exp(1j * 2 * np.pi * np.arange(S.K) / S.K).astype(np.complex64)
Filler = Union[str, "Frame"]


class Frame:
    """A structure: a label plus role -> filler bindings; fillers are atom names or nested Frames."""

    def __init__(self, label: str, **roles: Filler):
        self.label, self.roles = label, dict(roles)

    def __repr__(self) -> str:
        inner = ", ".join(f"{r}={v!r}" for r, v in self.roles.items())
        return f"{self.label}({inner})"


def _phase(v: np.ndarray) -> np.ndarray:
    return (np.round(np.angle(v) / (2 * np.pi / S.K)).astype(np.int64) % S.K).astype(np.uint8)


class ItemMemory:
    """Clean-up memory: atoms and chunks (frames), each stored as a 16-phase row; read = exhaustive sift."""

    def __init__(self, d: int = S.D, seed: int = 0):
        self.d, self.seed = d, seed
        self.names: List[str] = []
        self.rows: List[np.ndarray] = []
        self.chunks: Dict[str, Frame] = {}
        self._lat: Optional[np.ndarray] = None

    def atom(self, name: str) -> np.ndarray:
        """Unit-magnitude vector of an atom (SuperSeed codeword)."""
        return PH[S.codeword("A:" + name, self.seed, self.d)]

    def role(self, name: str) -> np.ndarray:
        return PH[S.codeword("R:" + name, self.seed, self.d)]

    def add(self, name: str, v: np.ndarray) -> None:
        self.names.append(name); self.rows.append(_phase(v)); self._lat = None

    def add_atom(self, name: str) -> None:
        if name not in self.names:
            self.add(name, self.atom(name))

    def encode(self, f: Filler, chunk_as: Optional[str] = None) -> np.ndarray:
        """Vector of an atom or (recursively) a frame. A frame registered as a chunk is also stored in the item memory."""
        if isinstance(f, str):
            self.add_atom(f)
            return self.atom(f)
        v = self.atom(f.label).copy()
        self.add_atom(f.label)
        for r, filler in f.roles.items():
            v = v + self.role(r) * self.encode(filler)
        v = v / np.sqrt(1 + len(f.roles))                      # Plate §X: expected length ~1
        if chunk_as:
            self.chunks[chunk_as] = f
            self.add(chunk_as, v)
        return v

    def cleanup(self, v: np.ndarray, top: int = 3) -> List[Tuple[str, float]]:
        if self._lat is None:
            self._lat = np.ascontiguousarray(np.stack(self.rows))
        sc = S.sift_phase(_phase(v), self._lat).astype(np.float64)
        sc /= float(S.sift_phase(self._lat[0], self._lat[:1])[0])
        idx = np.argsort(-sc)[:top]
        return [(self.names[i], float(sc[i])) for i in idx]

    def unbind(self, v: np.ndarray, role: str) -> np.ndarray:
        return v * np.conj(self.role(role))

    def unfold(self, v: np.ndarray, path: Sequence[str], clean_between: bool = True) -> List[Tuple[str, float]]:
        """Follow roles down a path (e.g. ["obj", "agt"] = agent of the object). With clean_between, each intermediate
        result is snapped to the nearest chunk before descending (Plate's slower, stronger method); without, the unbinds
        are chained (the fast method)."""
        for i, r in enumerate(path):
            v = self.unbind(v, r)
            if clean_between and i < len(path) - 1:
                name = self.cleanup(v, top=1)[0][0]
                if name not in self.chunks:
                    return [(name, float("nan"))]
                v = self.encode(self.chunks[name])
        return self.cleanup(v)
