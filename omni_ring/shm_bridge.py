"""
Shared Memory Fast-Bus (IPC) for OMNIRING.
Enables sub-50µs lockless state sharing via /dev/shm.
"""
import os
import mmap
import struct
import time
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from .types import RingSpec, DEFAULT_SPEC

MAGIC_BYTES = b"OMRING01"
# Header specification:
# magic (8 bytes)
# counter (8 bytes, uint64)
# timestamp (8 bytes, double)
# residues (5 * 4 bytes = 20 bytes, 5 x uint32)
# confidence (4 bytes, float32)
# Total = 48 bytes (evenly aligned to 8 and 16 bytes)
HEADER_FORMAT = "=8sQd5If"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)


class ShmBridge:
    """
    High-performance memory-mapped IPC bridge for OMNIRING phase vectors and state.
    """
    def __init__(
        self,
        spec: RingSpec = DEFAULT_SPEC,
        name: str = "omniring_bus.bin",
        base_dir: str = "/dev/shm",
        create: bool = True
    ):
        self.spec = spec
        self.filepath = os.path.join(base_dir, name)
        self.data_size = spec.dimension * 4  # 5120 float32 values = 20,480 bytes
        self.total_size = HEADER_SIZE + self.data_size
        self._counter = 0

        if create and not os.path.exists(self.filepath):
            with open(self.filepath, "wb") as f:
                f.write(b"\x00" * self.total_size)

        self.fd = os.open(self.filepath, os.O_RDWR)
        self.mmap = mmap.mmap(
            self.fd,
            self.total_size,
            flags=mmap.MAP_SHARED,
            prot=mmap.PROT_READ | mmap.PROT_WRITE
        )

    def write(self, phases: np.ndarray, residues: List[int], confidence: float,
              counter: Optional[int] = None) -> int:
        """
        Write state and phase vector to shared memory.
        
        Args:
            phases: 1D array of float32 phase angles (length D).
            residues: Sequence of 5 residue values [r3, r5, r7, r13, r17].
            confidence: Scalar confidence value in [0.0, 1.0].
            counter: Optional explicit sequence number (a writer that owns its own clock,
                e.g. one that resumes after a restart, passes it so the bus header and the
                writer agree). Default: previous counter + 1.
            
        Returns:
            The sequence counter for this write.
        """
        self._counter = int(counter) if counter is not None else self._counter + 1
        res_5 = [0] * 5
        for i, val in enumerate(residues[:5]):
            res_5[i] = int(val)

        header = struct.pack(
            HEADER_FORMAT,
            MAGIC_BYTES,
            self._counter,
            time.time(),
            res_5[0], res_5[1], res_5[2], res_5[3], res_5[4],
            float(confidence)
        )

        self.mmap.seek(0)
        self.mmap.write(header)

        # Cast to contiguous float32 buffer
        phases_f32 = np.ascontiguousarray(phases, dtype=np.float32)
        self.mmap.write(phases_f32.tobytes())
        return self._counter

    def read(self) -> Tuple[Dict[str, Any], np.ndarray]:
        """
        Read the latest state and phase vector from shared memory.

        Returns:
            header: Dict containing magic, counter, timestamp, residues, confidence.
            phases: 1D float32 array of shape (D,).
        """
        self.mmap.seek(0)
        header_bytes = self.mmap.read(HEADER_SIZE)
        unpacked = struct.unpack(HEADER_FORMAT, header_bytes)

        header = {
            "magic": unpacked[0],
            "counter": unpacked[1],
            "timestamp": unpacked[2],
            "residues": list(unpacked[3:8]),
            "confidence": unpacked[8]
        }

        phases_bytes = self.mmap.read(self.data_size)
        phases = np.frombuffer(phases_bytes, dtype=np.float32).copy()

        return header, phases

    def close(self):
        """Close memory map and file descriptor."""
        if hasattr(self, "mmap") and not self.mmap.closed:
            self.mmap.close()
        if hasattr(self, "fd"):
            try:
                os.close(self.fd)
            except OSError:
                pass

    def unlink(self):
        """Remove backing file from shared memory."""
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
