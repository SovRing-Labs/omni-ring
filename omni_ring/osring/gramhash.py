"""Gram-hash kernel (OS-RING-0 step 2): a C port of store.py's `_gram_phases` hot loop (n_grams x d, the one
slow leg of APP-0's text_vector), built beside libomniring_avx2.so via build_kernel.sh. Python fallback is the
same arithmetic store.py already runs; `gram_phases` picks the C kernel when the .so carries the symbol, else
falls back — bit-exact equality between the two paths is tests/test_gramhash.py."""
from __future__ import annotations

import ctypes
import os

import numpy as np
import numpy.ctypeslib as npct

_LIB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "libomniring_avx2.so")

try:
    _lib = ctypes.CDLL(_LIB_PATH)
    _u32_1d = npct.ndpointer(dtype=np.uint32, ndim=1, flags="C_CONTIGUOUS")
    _u8_1d = npct.ndpointer(dtype=np.uint8, ndim=1, flags="C_CONTIGUOUS")
    _lib.omniring_gram_phases.argtypes = [_u32_1d, ctypes.c_size_t, ctypes.c_uint32, _u8_1d]
    _lib.omniring_gram_phases.restype = None
    _HAVE_C = True
except (OSError, AttributeError):
    _lib = None
    _HAVE_C = False


def is_c_kernel_available() -> bool:
    return _HAVE_C


def gram_phases_py(h: np.ndarray, d: int) -> np.ndarray:
    """Pure-numpy reference — identical arithmetic to store.py's `_gram_phases`, taking the hash array directly."""
    h = h.astype(np.uint32)
    kd = np.arange(d, dtype=np.uint32) * np.uint32(2654435761) + np.uint32(0x9E3779B9)
    x = (h[:, None] ^ kd[None, :]) * np.uint32(0x85EBCA6B)
    x ^= x >> np.uint32(13)
    x *= np.uint32(0xC2B2AE35)
    x ^= x >> np.uint32(16)
    return ((x >> np.uint32(28)) & np.uint32(15)).astype(np.uint8)


def gram_phases(h: np.ndarray, d: int, backend: str = "auto") -> np.ndarray:
    """n x d uint8 phase table for a gram-hash array `h` (uint32). `backend`: "auto" (C if built, else numpy),
    "c" (force C, raises if unavailable), or "numpy" (force the fallback)."""
    if backend == "numpy" or (backend == "auto" and not _HAVE_C):
        return gram_phases_py(h, d)
    if not _HAVE_C:
        raise RuntimeError("omniring_gram_phases C kernel is not built — run build_kernel.sh")
    h = np.ascontiguousarray(h, dtype=np.uint32)
    n = len(h)
    out = np.empty(n * d, dtype=np.uint8)
    _lib.omniring_gram_phases(h, n, d, out)
    return out.reshape(n, d)
