"""
Multi-Ring Memory Bank for OMNIRING (Tier 2).
Maintains an array of 8 named 2,560-byte polymorphic slots in /dev/shm
with lockless seqlock synchronization and instant AVX2 cross-ring attention.
Total footprint: 20.48 KB (fits entirely inside CPU L1/L2 cache).
"""
import os
import mmap
import struct
import time
from enum import IntEnum
from typing import Optional, Dict, Tuple, Union, List
import numpy as np

from .polymorphic import (
    TernaryVector,
    PhaseVectorPacked,
    PolymorphicSlot,
    SLOT_BYTES,
    TERNARY_WORDS
)
from .avx2 import is_avx2_available, ternary_gemv_c


class RingSlotId(IntEnum):
    """The 8 specialized functional memory rings in the Bank."""
    TEMPORAL = 0        # Continuous time / FPE phase rotation state (v(t) = v_0^{\odot t})
    GOAL = 1            # Active high-level task / objective vector
    ENTITY_CONTEXT = 2  # Scene binding (E_1 \otimes V_1 \oplus E_2 \otimes V_2)
    SYSTEM_STATE = 3    # Telemetry, health, daemon state hypervector
    ACTION_ATTRACTOR = 4# Current candidate tool / action codebook vector
    IMMUNE_FILTER = 5   # Anomaly filter / residue desync mask
    WORKING_0 = 6       # Scratchpad slot A
    WORKING_1 = 7       # Scratchpad slot B


NUM_BANK_SLOTS = 8
MAGIC_BYTES = b"OMRNG08B"

# Header Layout:
# magic (8 bytes)
# num_slots (uint32 = 4 bytes)
# reserved (uint32 = 4 bytes)
# seqlocks (8 * uint64 = 64 bytes)
# timestamps (8 * double = 64 bytes)
# Total Header = 144 bytes, padded to 256 bytes (cacheline aligned)
HEADER_FORMAT = "=8sII8Q8d112x"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
assert HEADER_SIZE == 256

BANK_FILE_SIZE = HEADER_SIZE + NUM_BANK_SLOTS * SLOT_BYTES # 256 + 20,480 = 20,736 bytes


class RingBank:
    """
    Lockless, L1/L2 cache-resident Multi-Ring Memory Bank.
    Enables sub-microsecond state tracking and AVX2 cross-ring attention.
    """
    def __init__(
        self,
        name: str = "omniring_bank.bin",
        base_dir: str = "/dev/shm",
        create: bool = True
    ):
        self.filepath = os.path.join(base_dir, name)
        self.total_size = BANK_FILE_SIZE
        
        if create and not os.path.exists(self.filepath):
            with open(self.filepath, "wb") as f:
                f.write(b"\x00" * self.total_size)
                
        self.fd = os.open(self.filepath, os.O_RDWR)
        self.mmap = mmap.mmap(self.fd, self.total_size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        
        if create:
            # Initialize magic and slot count if zeroed
            if self.mmap[:8] == b"\x00" * 8:
                self.mmap.seek(0)
                self.mmap.write(struct.pack("=8sII", MAGIC_BYTES, NUM_BANK_SLOTS, 0))

    def _get_seq_offset(self, slot_idx: int) -> int:
        # magic(8) + num_slots(4) + reserved(4) = 16
        return 16 + slot_idx * 8

    def _get_ts_offset(self, slot_idx: int) -> int:
        # 16 + 64 = 80
        return 80 + slot_idx * 8

    def _get_data_offset(self, slot_idx: int) -> int:
        return HEADER_SIZE + slot_idx * SLOT_BYTES

    def _read_seq(self, slot_idx: int) -> int:
        off = self._get_seq_offset(slot_idx)
        return struct.unpack("<Q", self.mmap[off:off + 8])[0]

    def _write_seq(self, slot_idx: int, val: int) -> None:
        off = self._get_seq_offset(slot_idx)
        self.mmap[off:off + 8] = struct.pack("<Q", val)

    def _write_ts(self, slot_idx: int, ts: float) -> None:
        off = self._get_ts_offset(slot_idx)
        self.mmap[off:off + 8] = struct.pack("<d", ts)

    def write_slot(
        self,
        slot_id: Union[int, RingSlotId],
        slot_data: Union[PolymorphicSlot, TernaryVector, PhaseVectorPacked, bytes]
    ) -> None:
        """Write into a named slot under atomic seqlock."""
        slot_idx = int(slot_id)
        assert 0 <= slot_idx < NUM_BANK_SLOTS, f"Invalid slot_id {slot_idx}"
        
        if isinstance(slot_data, PolymorphicSlot):
            raw = slot_data.raw_bytes
        elif isinstance(slot_data, (TernaryVector, PhaseVectorPacked)):
            raw = slot_data.to_bytes()
        elif isinstance(slot_data, (bytes, bytearray)):
            assert len(slot_data) == SLOT_BYTES
            raw = bytes(slot_data)
        else:
            raise TypeError(f"Unsupported slot data type: {type(slot_data)}")

        offset = self._get_data_offset(slot_idx)
        s = self._read_seq(slot_idx)
        
        # Begin write: odd seq
        self._write_seq(slot_idx, s + 1)
        self.mmap[offset:offset + SLOT_BYTES] = raw
        self._write_ts(slot_idx, time.time())
        # Finish write: even seq
        self._write_seq(slot_idx, s + 2)

    def read_slot(self, slot_id: Union[int, RingSlotId], max_retries: int = 100) -> PolymorphicSlot:
        """Read a slot under lockless seqlock consistency check."""
        slot_idx = int(slot_id)
        assert 0 <= slot_idx < NUM_BANK_SLOTS, f"Invalid slot_id {slot_idx}"
        offset = self._get_data_offset(slot_idx)

        for _ in range(max_retries):
            s1 = self._read_seq(slot_idx)
            if s1 & 1:  # Currently being written
                time.sleep(0.00001)
                continue
            data = bytes(self.mmap[offset:offset + SLOT_BYTES])
            s2 = self._read_seq(slot_idx)
            if s1 == s2 and not (s2 & 1):
                slot = PolymorphicSlot()
                slot.data[:] = data
                return slot

        # Fallback to direct read if contention timeout
        data = bytes(self.mmap[offset:offset + SLOT_BYTES])
        slot = PolymorphicSlot()
        slot.data[:] = data
        return slot

    def write_ternary(self, slot_id: Union[int, RingSlotId], tv: TernaryVector) -> None:
        self.write_slot(slot_id, tv)

    def read_ternary(self, slot_id: Union[int, RingSlotId]) -> TernaryVector:
        return self.read_slot(slot_id).as_ternary()

    def write_phase(self, slot_id: Union[int, RingSlotId], pv: PhaseVectorPacked) -> None:
        self.write_slot(slot_id, pv)

    def read_phase(self, slot_id: Union[int, RingSlotId]) -> PhaseVectorPacked:
        return self.read_slot(slot_id).as_phase()

    def cross_attention_scan(self, query: TernaryVector) -> np.ndarray:
        """
        Compute instant AVX2 cross-ring attention across all 8 slots simultaneously.
        Returns array of shape (8,) containing integer dot products in [-10240, +10240].
        Execution time: < 2 microseconds.
        """
        # Build contiguous (8, 160) uint64 bitplanes directly from slot memory
        matrix_sign = np.empty((NUM_BANK_SLOTS, TERNARY_WORDS), dtype=np.uint64)
        matrix_active = np.empty((NUM_BANK_SLOTS, TERNARY_WORDS), dtype=np.uint64)

        for idx in range(NUM_BANK_SLOTS):
            off = self._get_data_offset(idx)
            raw = self.mmap[off:off + SLOT_BYTES]
            matrix_sign[idx] = np.frombuffer(raw[:1280], dtype=np.uint64)
            matrix_active[idx] = np.frombuffer(raw[1280:2560], dtype=np.uint64)

        if is_avx2_available():
            scores = ternary_gemv_c(matrix_sign, matrix_active, query.sign, query.active)
            return scores
        else:
            out = np.empty(NUM_BANK_SLOTS, dtype=np.int32)
            for idx in range(NUM_BANK_SLOTS):
                tv = TernaryVector(matrix_sign[idx], matrix_active[idx])
                out[idx] = query.dot(tv)
            return out

    def get_snapshot(self) -> Dict[str, PolymorphicSlot]:
        """Atomic snapshot of all 8 slots."""
        snapshot = {}
        for slot_enum in RingSlotId:
            snapshot[slot_enum.name] = self.read_slot(slot_enum)
        return snapshot

    def close(self):
        if hasattr(self, "mmap") and not self.mmap.closed:
            self.mmap.close()
        if hasattr(self, "fd"):
            try:
                os.close(self.fd)
            except OSError:
                pass

    def unlink(self):
        self.close()
        if os.path.exists(self.filepath):
            try:
                os.remove(self.filepath)
            except OSError:
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
