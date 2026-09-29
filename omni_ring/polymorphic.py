"""
Polymorphic Two-Faced Bus implementation for OMNIRING.

Bridges the 10,240-D Ternary VSA representation and the 5,120-D Coprime Phase representation
over identical 2,560-byte slots under seqlock concurrency.
"""
from __future__ import annotations
import os
import mmap
import struct
import numpy as np
from typing import Optional, Tuple, Union, List

from .types import DEFAULT_SPEC
from .avx2 import (
    is_avx2_available,
    pack_nibbles,
    unpack_nibbles,
    similarity_avx2,
    batch_similarity_avx2,
    bind_avx2,
    unbind_avx2,
    transcode_ternary_to_phase_c,
    transcode_phase_to_ternary_c,
    ternary_dot_c,
    ternary_bind_c,
)

SLOT_BYTES = 2560
TERNARY_D = 10240
PHASE_D = 5120
TERNARY_WORDS = 160

# Canonical ternary-pair -> phase mapping.
# The 8 active pairs map bijectively onto the 8 even phases {0,2,4,6,8,10,12,14};
# the neutral pair (0, 0) maps to phase 0, the additive identity, so a null ternary
# block yields a null phase block. Total by construction: all 9 pairs are listed, so
# an unlisted pair raises instead of silently fabricating a phase (which is exactly
# the OSEAM-D1-TRANSCODE defect: (0,0) used to yield a position-dependent
# `(k * 5 + 7) & 0x0F` on the fallback and the same fiction in C).
# This table mirrors TERNARY_TO_PHASE_LUT in c_src/omniring_polymorphic.c.
TERNARY_TO_PHASE_LUT: dict[Tuple[int, int], int] = {
    (1, 0): 0, (1, 1): 2, (0, 1): 4, (-1, 1): 6,
    (-1, 0): 8, (-1, -1): 10, (0, -1): 12, (1, -1): 14,
    (0, 0): 0,
}


def transcode_ternary_to_phase_python(sign: np.ndarray, active: np.ndarray) -> np.ndarray:
    """
    Portable transcode of a 10,240-D ternary bitplane pair into 5,120 packed 4-bit
    phases (2,560 bytes) -- the no-AVX2 fallback.

    Byte-for-byte equivalent to `transcode_ternary_to_phase_c`, and deliberately
    callable on an AVX2 host so both paths can be asserted equal in tests.
    """
    sign = np.ascontiguousarray(sign, dtype=np.uint64)
    active = np.ascontiguousarray(active, dtype=np.uint64)

    dense = np.zeros(TERNARY_D, dtype=np.int8)
    for w in range(TERNARY_WORDS):
        s = int(sign[w])
        a = int(active[w])
        for bit in range(64):
            if (a >> bit) & 1:
                dense[w * 64 + bit] = -1 if ((s >> bit) & 1) else 1

    phases = np.empty(PHASE_D, dtype=np.uint8)
    for k in range(PHASE_D):
        phases[k] = TERNARY_TO_PHASE_LUT[(int(dense[2 * k]), int(dense[2 * k + 1]))]

    return pack_nibbles(phases)


class TernaryVector:
    """
    10,240-dimensional ternary hypervector {-1, 0, +1} stored as two 1280-byte bitplanes:
    - sign: 160 x uint64
    - active: 160 x uint64
    Total size: 2,560 bytes.
    """
    __slots__ = ("sign", "active")

    def __init__(self, sign: Optional[np.ndarray] = None, active: Optional[np.ndarray] = None):
        if sign is None:
            self.sign = np.zeros(TERNARY_WORDS, dtype=np.uint64)
        else:
            self.sign = np.ascontiguousarray(sign, dtype=np.uint64)
            
        if active is None:
            self.active = np.zeros(TERNARY_WORDS, dtype=np.uint64)
        else:
            self.active = np.ascontiguousarray(active, dtype=np.uint64)

        if len(self.sign) != TERNARY_WORDS or len(self.active) != TERNARY_WORDS:
            raise ValueError(f"TernaryVector requires {TERNARY_WORDS} 64-bit words per plane.")

    @classmethod
    def random(cls, seed: Optional[int] = None, sparsity: float = 0.5) -> TernaryVector:
        """Generate a random ternary hypervector."""
        rng = np.random.default_rng(seed)
        # Random uint64s for active mask and sign mask
        active_words = rng.integers(0, 0xFFFFFFFFFFFFFFFF, TERNARY_WORDS, dtype=np.uint64)
        sign_words = rng.integers(0, 0xFFFFFFFFFFFFFFFF, TERNARY_WORDS, dtype=np.uint64)
        return cls(sign=sign_words, active=active_words)

    @classmethod
    def from_dense(cls, arr: np.ndarray) -> TernaryVector:
        """Convert a 1D array of {-1, 0, +1} (length 10,240) into a TernaryVector."""
        assert len(arr) == TERNARY_D, f"Expected {TERNARY_D} elements, got {len(arr)}"
        sign = np.zeros(TERNARY_WORDS, dtype=np.uint64)
        active = np.zeros(TERNARY_WORDS, dtype=np.uint64)

        for w in range(TERNARY_WORDS):
            chunk = arr[w * 64:(w + 1) * 64]
            act_mask = 0
            sgn_mask = 0
            for bit, val in enumerate(chunk):
                if val != 0:
                    act_mask |= (1 << bit)
                    if val < 0:
                        sgn_mask |= (1 << bit)
            sign[w] = np.uint64(sgn_mask)
            active[w] = np.uint64(act_mask)

        return cls(sign=sign, active=active)

    def to_dense(self) -> np.ndarray:
        """Convert back to a 1D int8 array of shape (10,240,) in {-1, 0, +1}."""
        dense = np.zeros(TERNARY_D, dtype=np.int8)
        for w in range(TERNARY_WORDS):
            s = int(self.sign[w])
            a = int(self.active[w])
            for bit in range(64):
                if (a >> bit) & 1:
                    dense[w * 64 + bit] = -1 if ((s >> bit) & 1) else 1
        return dense

    def dot(self, other: TernaryVector) -> int:
        """Ternary dot product in [-10240, +10240]."""
        if is_avx2_available():
            return ternary_dot_c(self.sign, self.active, other.sign, other.active)
        else:
            active_both = self.active & other.active
            diff_sign = self.sign ^ other.sign
            pos = active_both & (~diff_sign)
            neg = active_both & diff_sign
            # popcounts
            pos_cnt = sum(bin(int(x)).count('1') for x in pos)
            neg_cnt = sum(bin(int(x)).count('1') for x in neg)
            return pos_cnt - neg_cnt

    def bind(self, other: TernaryVector) -> TernaryVector:
        """Ternary binding: XOR signs, AND active."""
        if is_avx2_available():
            s, a = ternary_bind_c(self.sign, self.active, other.sign, other.active)
            return TernaryVector(s, a)
        else:
            return TernaryVector(self.sign ^ other.sign, self.active & other.active)

    def to_bytes(self) -> bytes:
        """Serialize to 2,560 bytes (sign plane first, then active plane)."""
        return struct.pack("<160Q160Q", *self.sign, *self.active)

    @classmethod
    def from_bytes(cls, data: bytes) -> TernaryVector:
        """Deserialize from 2,560 bytes."""
        assert len(data) == SLOT_BYTES, f"Expected {SLOT_BYTES} bytes, got {len(data)}"
        unpacked = struct.unpack("<160Q160Q", data)
        return cls(
            sign=np.array(unpacked[:160], dtype=np.uint64),
            active=np.array(unpacked[160:], dtype=np.uint64)
        )

    def to_phase(self) -> PhaseVectorPacked:
        """
        Transcode to 5,120-D PhaseVectorPacked.

        The 8 active ternary pairs map bijectively onto the even phases
        {0, 2, 4, 6, 8, 10, 12, 14}; the neutral pair (0, 0) maps to phase 0, so the
        null ternary vector transcodes to the null phase vector. The C kernel and the
        fallback share one definition of that mapping (see TERNARY_TO_PHASE_LUT) and
        therefore always agree. (OSEAM-D1-TRANSCODE)
        """
        if is_avx2_available():
            packed = transcode_ternary_to_phase_c(self.sign, self.active)
            return PhaseVectorPacked(packed)
        else:
            return PhaseVectorPacked(transcode_ternary_to_phase_python(self.sign, self.active))


class PhaseVectorPacked:
    """
    5,120-dimensional phase vector (K=16, 4 bits per dimension),
    packed 2 dimensions per byte into 2,560 bytes.
    """
    __slots__ = ("data",)

    def __init__(self, data: Optional[Union[np.ndarray, bytes]] = None):
        if data is None:
            self.data = np.zeros(SLOT_BYTES, dtype=np.uint8)
        elif isinstance(data, bytes):
            assert len(data) == SLOT_BYTES
            self.data = np.frombuffer(data, dtype=np.uint8).copy()
        else:
            self.data = np.ascontiguousarray(data, dtype=np.uint8)
            assert len(self.data) == SLOT_BYTES

    @classmethod
    def from_phases(cls, phases: np.ndarray) -> PhaseVectorPacked:
        """Create from 5,120 phase indices in [0, 15]."""
        return cls(pack_nibbles(phases))

    def to_phases(self) -> np.ndarray:
        """Unpack to 5,120 uint8 phase indices in [0, 15]."""
        return unpack_nibbles(self.data)

    def similarity(self, other: PhaseVectorPacked) -> int:
        """AVX2 1-cycle table-lookup cosine similarity score."""
        return similarity_avx2(self.data, other.data)

    def bind(self, other: PhaseVectorPacked) -> PhaseVectorPacked:
        """Hadamard phase binding: (a + b) mod 16."""
        return PhaseVectorPacked(bind_avx2(self.data, other.data))

    def unbind(self, other: PhaseVectorPacked) -> PhaseVectorPacked:
        """Phase unbinding: (a - b) mod 16."""
        return PhaseVectorPacked(unbind_avx2(self.data, other.data))

    def to_bytes(self) -> bytes:
        return self.data.tobytes()

    @classmethod
    def from_bytes(cls, data: bytes) -> PhaseVectorPacked:
        return cls(data)

    def to_ternary(self) -> TernaryVector:
        """Transcode to 10,240-D TernaryVector."""
        if is_avx2_available():
            sign, active = transcode_phase_to_ternary_c(self.data)
            return TernaryVector(sign=sign, active=active)
        else:
            phases = self.to_phases()
            p_to_t0 = [1, 1, 1, 0, 0, 0, -1, -1, -1, -1, -1, 0, 0, 0, 1, 1]
            p_to_t1 = [0, 1, 1, 1, 1, 1, 1, 0, 0, -1, -1, -1, -1, -1, -1, 0]
            dense = np.empty(TERNARY_D, dtype=np.int8)
            for k in range(PHASE_D):
                p = phases[k]
                dense[2 * k] = p_to_t0[p]
                dense[2 * k + 1] = p_to_t1[p]
            return TernaryVector.from_dense(dense)


class PolymorphicSlot:
    """
    A single 2,560-byte memory slot capable of polymorphic interpretation.
    Can be treated as either a 10,240-D TernaryVector or a 5,120-D PhaseVectorPacked.
    """
    def __init__(self, raw_buffer: Optional[bytearray] = None):
        if raw_buffer is None:
            self._buffer = bytearray(SLOT_BYTES)
        else:
            assert len(raw_buffer) == SLOT_BYTES
            self._buffer = raw_buffer

    @property
    def raw_bytes(self) -> bytes:
        return bytes(self._buffer)

    @property
    def data(self) -> bytearray:
        """Direct access to the 2,560-byte underlying slot buffer."""
        return self._buffer

    def as_ternary(self) -> TernaryVector:
        """View/deserialize slot memory as a 10,240-D TernaryVector."""
        return TernaryVector.from_bytes(bytes(self._buffer))

    def as_phase(self) -> PhaseVectorPacked:
        """View/deserialize slot memory as a 5,120-D PhaseVectorPacked."""
        return PhaseVectorPacked.from_bytes(bytes(self._buffer))

    def write_ternary(self, tv: TernaryVector) -> None:
        """Write a 10,240-D TernaryVector into the slot."""
        self._buffer[:] = tv.to_bytes()

    def write_phase(self, pv: PhaseVectorPacked) -> None:
        """Write a 5,120-D PhaseVectorPacked into the slot."""
        self._buffer[:] = pv.to_bytes()

    def transcode_to_phase_in_place(self) -> PhaseVectorPacked:
        """Read the slot as Ternary, transcode to Phase, and overwrite the slot."""
        tv = self.as_ternary()
        pv = tv.to_phase()
        self.write_phase(pv)
        return pv

    def transcode_to_ternary_in_place(self) -> TernaryVector:
        """Read the slot as Phase, transcode to Ternary, and overwrite the slot."""
        pv = self.as_phase()
        tv = pv.to_ternary()
        self.write_ternary(tv)
        return tv


class PolymorphicBus:
    """
    Production-grade shared-memory bus client supporting dual-face polymorphic slots
    and seqlock synchronization on /dev/shm.
    """
    def __init__(
        self,
        num_slots: int = 1000,
        bus_path: str = "/dev/shm/vsa_matrix_bus",
        seq_path: str = "/dev/shm/vsa_matrix_seq",
        create: bool = False
    ):
        self.num_slots = num_slots
        self.bus_path = bus_path
        self.seq_path = seq_path
        self.bus_size = num_slots * SLOT_BYTES
        self.seq_size = num_slots * 8

        if create:
            if not os.path.exists(self.bus_path):
                with open(self.bus_path, "wb") as f:
                    f.write(b"\x00" * self.bus_size)
            if not os.path.exists(self.seq_path):
                with open(self.seq_path, "wb") as f:
                    f.write(b"\x00" * self.seq_size)

        self.bus_fd = os.open(self.bus_path, os.O_RDWR)
        self.bus_map = mmap.mmap(self.bus_fd, self.bus_size, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE)

        self.seq_fd = os.open(self.seq_path, os.O_RDWR)
        self.seq_map = mmap.mmap(self.seq_fd, self.seq_size, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE)

    def _get_seq(self, slot_idx: int) -> int:
        offset = slot_idx * 8
        return struct.unpack("<Q", self.seq_map[offset:offset + 8])[0]

    def _set_seq(self, slot_idx: int, val: int) -> None:
        offset = slot_idx * 8
        self.seq_map[offset:offset + 8] = struct.pack("<Q", val)

    def read_slot(self, slot_idx: int) -> PolymorphicSlot:
        """Read a 2,560-byte slot with seqlock torn-read protection."""
        if slot_idx < 0 or slot_idx >= self.num_slots:
            raise IndexError(f"Slot {slot_idx} out of range [0, {self.num_slots})")

        offset = slot_idx * SLOT_BYTES
        while True:
            s0 = self._get_seq(slot_idx)
            if s0 & 1:
                continue  # Writer active, spin
            data = bytes(self.bus_map[offset:offset + SLOT_BYTES])
            s1 = self._get_seq(slot_idx)
            if s0 == s1 and (s1 & 1) == 0:
                return PolymorphicSlot(bytearray(data))

    def write_slot(self, slot_idx: int, slot: PolymorphicSlot) -> None:
        """Write a 2,560-byte slot under seqlock protection."""
        if slot_idx < 0 or slot_idx >= self.num_slots:
            raise IndexError(f"Slot {slot_idx} out of range [0, {self.num_slots})")

        offset = slot_idx * SLOT_BYTES
        s = self._get_seq(slot_idx)
        self._set_seq(slot_idx, s + 1)  # Odd = writing
        self.bus_map[offset:offset + SLOT_BYTES] = slot.raw_bytes
        self._set_seq(slot_idx, s + 2)  # Even = finished

    def read_ternary(self, slot_idx: int) -> TernaryVector:
        return self.read_slot(slot_idx).as_ternary()

    def write_ternary(self, slot_idx: int, tv: TernaryVector) -> None:
        slot = PolymorphicSlot()
        slot.write_ternary(tv)
        self.write_slot(slot_idx, slot)

    def read_phase(self, slot_idx: int) -> PhaseVectorPacked:
        return self.read_slot(slot_idx).as_phase()

    def write_phase(self, slot_idx: int, pv: PhaseVectorPacked) -> None:
        slot = PolymorphicSlot()
        slot.write_phase(pv)
        self.write_slot(slot_idx, slot)

    def close(self):
        if hasattr(self, "bus_map") and not self.bus_map.closed:
            self.bus_map.close()
        if hasattr(self, "bus_fd"):
            try:
                os.close(self.bus_fd)
            except OSError:
                pass
        if hasattr(self, "seq_map") and not self.seq_map.closed:
            self.seq_map.close()
        if hasattr(self, "seq_fd"):
            try:
                os.close(self.seq_fd)
            except OSError:
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
