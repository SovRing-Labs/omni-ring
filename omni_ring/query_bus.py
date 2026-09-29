"""
Bidirectional lockless query/response ring buffer over /dev/shm.
Implements Fractional Power Encoding (FPE) for continuous time/analog variables.
Maintains the 2,560-byte slot invariant.
"""
import os
import mmap
import struct
import numpy as np
from typing import Tuple, Optional

from .polymorphic import PhaseVectorPacked, SLOT_BYTES

MAGIC_BYTES = b"OMRINGQB"
# Header: 
# magic (8 bytes)
# head_query (uint32)
# tail_query (uint32)
# head_resp (uint32)
# tail_resp (uint32)
# Total Header = 24 bytes, padded to 32 bytes
HEADER_FORMAT = "=8sIIII8x"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

# Slot structure:
# seq (uint64) = 8 bytes
# data = 2560 bytes
# Total Slot = 2568 bytes, padded to 2576 (multiple of 8)
SLOT_FORMAT = "=Q2560s8x"
SLOT_SIZE = struct.calcsize(SLOT_FORMAT)

DEFAULT_CAPACITY = 16

class QueryBus:
    """
    Bidirectional, lockless query/response ring buffer over /dev/shm.
    Protects the 2,560-byte slot invariant.
    """
    def __init__(self, capacity: int = DEFAULT_CAPACITY, name: str = "omniring_query.bin", base_dir: str = "/dev/shm", create: bool = True):
        self.capacity = capacity
        self.filepath = os.path.join(base_dir, name)
        self.ring_size = capacity * SLOT_SIZE
        self.total_size = HEADER_SIZE + 2 * self.ring_size
        
        if create and not os.path.exists(self.filepath):
            with open(self.filepath, "wb") as f:
                f.write(b"\x00" * self.total_size)
                
        self.fd = os.open(self.filepath, os.O_RDWR)
        self.mmap = mmap.mmap(self.fd, self.total_size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        
        if create:
            self.mmap.seek(0)
            self.mmap.write(struct.pack("=8s", MAGIC_BYTES))
            self._write_hq(0)
            self._write_tq(0)
            self._write_hr(0)
            self._write_tr(0)

    # Memory layout:
    # 0..7: magic
    # 8..11: hq
    # 12..15: tq
    # 16..19: hr
    # 20..23: tr
    
    def _read_hq(self) -> int: return struct.unpack("I", self.mmap[8:12])[0]
    def _read_tq(self) -> int: return struct.unpack("I", self.mmap[12:16])[0]
    def _read_hr(self) -> int: return struct.unpack("I", self.mmap[16:20])[0]
    def _read_tr(self) -> int: return struct.unpack("I", self.mmap[20:24])[0]
    
    def _write_hq(self, val: int): self.mmap[8:12] = struct.pack("I", val)
    def _write_tq(self, val: int): self.mmap[12:16] = struct.pack("I", val)
    def _write_hr(self, val: int): self.mmap[16:20] = struct.pack("I", val)
    def _write_tr(self, val: int): self.mmap[20:24] = struct.pack("I", val)
    
    def send_query(self, pv: PhaseVectorPacked, seq: int) -> bool:
        hq = self._read_hq()
        tq = self._read_tq()
        next_hq = (hq + 1) % self.capacity
        if next_hq == tq:
            return False
            
        offset = HEADER_SIZE + hq * SLOT_SIZE
        self.mmap[offset:offset+SLOT_SIZE] = struct.pack(SLOT_FORMAT, seq, pv.to_bytes())
        
        self._write_hq(next_hq)
        return True
        
    def recv_query(self) -> Optional[Tuple[PhaseVectorPacked, int]]:
        hq = self._read_hq()
        tq = self._read_tq()
        if hq == tq:
            return None
            
        offset = HEADER_SIZE + tq * SLOT_SIZE
        slot_bytes = self.mmap[offset:offset+SLOT_SIZE]
        unpacked = struct.unpack(SLOT_FORMAT, slot_bytes)
        seq = unpacked[0]
        pv = PhaseVectorPacked.from_bytes(unpacked[1])
        
        self._write_tq((tq + 1) % self.capacity)
        return pv, seq

    def send_response(self, pv: PhaseVectorPacked, seq: int) -> bool:
        hr = self._read_hr()
        tr = self._read_tr()
        next_hr = (hr + 1) % self.capacity
        if next_hr == tr:
            return False
            
        offset = HEADER_SIZE + self.ring_size + hr * SLOT_SIZE
        self.mmap[offset:offset+SLOT_SIZE] = struct.pack(SLOT_FORMAT, seq, pv.to_bytes())
        
        self._write_hr(next_hr)
        return True
        
    def recv_response(self) -> Optional[Tuple[PhaseVectorPacked, int]]:
        hr = self._read_hr()
        tr = self._read_tr()
        if hr == tr:
            return None
            
        offset = HEADER_SIZE + self.ring_size + tr * SLOT_SIZE
        slot_bytes = self.mmap[offset:offset+SLOT_SIZE]
        unpacked = struct.unpack(SLOT_FORMAT, slot_bytes)
        seq = unpacked[0]
        pv = PhaseVectorPacked.from_bytes(unpacked[1])
        
        self._write_tr((tr + 1) % self.capacity)
        return pv, seq

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

def fractional_power_encoding(base: PhaseVectorPacked, t: float) -> PhaseVectorPacked:
    """
    Fractional Power Encoding (FPE) for continuous time/analog variables.
    Computes v(t) = v_0^{\\odot t} for t in R.
    Math runs in Python (numpy) cleanly outside the AVX2 fast path.
    Outputs a standard integer-packed 2,560-byte PhaseVectorPacked.
    """
    phases = base.to_phases()
    scaled = np.round(phases.astype(np.float32) * float(t)).astype(np.int32)
    new_phases = (scaled % 16).astype(np.uint8)
    return PhaseVectorPacked.from_phases(new_phases)
