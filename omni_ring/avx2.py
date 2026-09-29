"""
AVX2 SIMD acceleration wrapper for OMNIRING 4-bit nibble-packed vectors and polymorphic operations.
"""
import os
import ctypes
import numpy as np
import numpy.ctypeslib as npct
from typing import Optional, Tuple

_LIB_PATH = os.path.join(os.path.dirname(__file__), "libomniring_avx2.so")

try:
    _lib = ctypes.CDLL(_LIB_PATH)
    
    _array_1d_uint8 = npct.ndpointer(dtype=np.uint8, ndim=1, flags="C_CONTIGUOUS")
    _array_2d_uint8 = npct.ndpointer(dtype=np.uint8, ndim=2, flags="C_CONTIGUOUS")
    _array_1d_int8 = npct.ndpointer(dtype=np.int8, ndim=1, flags="C_CONTIGUOUS")
    _array_1d_int32 = npct.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS")
    _array_1d_uint64 = npct.ndpointer(dtype=np.uint64, ndim=1, flags="C_CONTIGUOUS")
    _array_2d_uint64 = npct.ndpointer(dtype=np.uint64, ndim=2, flags="C_CONTIGUOUS")

    # int32_t omniring_similarity_avx2(const uint8_t* a, const uint8_t* b);
    _lib.omniring_similarity_avx2.argtypes = [_array_1d_uint8, _array_1d_uint8]
    _lib.omniring_similarity_avx2.restype = ctypes.c_int32

    # void omniring_batch_similarity_avx2(...)
    _lib.omniring_batch_similarity_avx2.argtypes = [
        _array_1d_uint8,
        _array_2d_uint8,
        ctypes.c_size_t,
        _array_1d_int32
    ]
    _lib.omniring_batch_similarity_avx2.restype = None

    # void omniring_bind_avx2(const uint8_t* a, const uint8_t* b, uint8_t* out);
    _lib.omniring_bind_avx2.argtypes = [_array_1d_uint8, _array_1d_uint8, _array_1d_uint8]
    _lib.omniring_bind_avx2.restype = None

    # void omniring_unbind_avx2(const uint8_t* a, const uint8_t* b, uint8_t* out);
    _lib.omniring_unbind_avx2.argtypes = [_array_1d_uint8, _array_1d_uint8, _array_1d_uint8]
    _lib.omniring_unbind_avx2.restype = None

    # void omniring_pack_nibbles(const uint8_t* in, uint8_t* out, size_t n_dims);
    _lib.omniring_pack_nibbles.argtypes = [_array_1d_uint8, _array_1d_uint8, ctypes.c_size_t]
    _lib.omniring_pack_nibbles.restype = None

    # void omniring_unpack_nibbles(const uint8_t* in, uint8_t* out, size_t n_dims);
    _lib.omniring_unpack_nibbles.argtypes = [_array_1d_uint8, _array_1d_uint8, ctypes.c_size_t]
    _lib.omniring_unpack_nibbles.restype = None

    # void omniring_transcode_ternary_to_phase(const uint64_t* sign, const uint64_t* active, uint8_t* out_packed_phases);
    _lib.omniring_transcode_ternary_to_phase.argtypes = [_array_1d_uint64, _array_1d_uint64, _array_1d_uint8]
    _lib.omniring_transcode_ternary_to_phase.restype = None

    # void omniring_transcode_phase_to_ternary(const uint8_t* packed_phases, uint64_t* out_sign, uint64_t* out_active);
    _lib.omniring_transcode_phase_to_ternary.argtypes = [_array_1d_uint8, _array_1d_uint64, _array_1d_uint64]
    _lib.omniring_transcode_phase_to_ternary.restype = None

    # int32_t omniring_ternary_dot(const uint64_t* sign_a, const uint64_t* active_a, const uint64_t* sign_b, const uint64_t* active_b);
    _lib.omniring_ternary_dot.argtypes = [_array_1d_uint64, _array_1d_uint64, _array_1d_uint64, _array_1d_uint64]
    _lib.omniring_ternary_dot.restype = ctypes.c_int32

    # void omniring_ternary_bind(const uint64_t* sign_a, const uint64_t* active_a, const uint64_t* sign_b, const uint64_t* active_b, uint64_t* out_sign, uint64_t* out_active);
    _lib.omniring_ternary_bind.argtypes = [
        _array_1d_uint64, _array_1d_uint64,
        _array_1d_uint64, _array_1d_uint64,
        _array_1d_uint64, _array_1d_uint64
    ]
    _lib.omniring_ternary_bind.restype = None

    # void omniring_set_num_threads(int32_t n_threads);
    _lib.omniring_set_num_threads.argtypes = [ctypes.c_int32]
    _lib.omniring_set_num_threads.restype = None

    # void omniring_ternary_gemv(...)
    _lib.omniring_ternary_gemv.argtypes = [
        _array_2d_uint64,
        _array_2d_uint64,
        _array_1d_uint64,
        _array_1d_uint64,
        ctypes.c_size_t,
        ctypes.c_size_t,
        _array_1d_int32
    ]
    _lib.omniring_ternary_gemv.restype = None

    # void omniring_gemv_ternary_int8(...)
    _lib.omniring_gemv_ternary_int8.argtypes = [
        _array_2d_uint64,
        _array_2d_uint64,
        _array_1d_int8,
        ctypes.c_size_t,
        ctypes.c_size_t,
        _array_1d_int32
    ]
    _lib.omniring_gemv_ternary_int8.restype = None

    _AVX2_AVAILABLE = True
except Exception:
    _lib = None
    _AVX2_AVAILABLE = False


def is_avx2_available() -> bool:
    """Return whether the AVX2 native library is loaded and available."""
    return _AVX2_AVAILABLE


def pack_nibbles(phases: np.ndarray) -> np.ndarray:
    """
    Pack an array of 4-bit phase indices (0..15) into 2-nibbles-per-byte format.
    Input shape (D,) -> Output shape (D // 2,) of uint8.
    """
    n_dims = len(phases)
    assert n_dims % 2 == 0, "Dimension count must be even"
    in_arr = phases if (phases.flags.c_contiguous and phases.dtype == np.uint8) else np.ascontiguousarray(phases, dtype=np.uint8)
    out_arr = np.empty(n_dims // 2, dtype=np.uint8)

    if _AVX2_AVAILABLE:
        _lib.omniring_pack_nibbles(in_arr, out_arr, n_dims)
    else:
        out_arr[:] = (in_arr[0::2] & 0x0F) | ((in_arr[1::2] & 0x0F) << 4)

    return out_arr


def unpack_nibbles(packed: np.ndarray) -> np.ndarray:
    """
    Unpack 2-nibbles-per-byte format into individual 4-bit phase indices.
    Input shape (D // 2,) -> Output shape (D,) of uint8.
    """
    n_packed = len(packed)
    n_dims = n_packed * 2
    in_arr = packed if (packed.flags.c_contiguous and packed.dtype == np.uint8) else np.ascontiguousarray(packed, dtype=np.uint8)
    out_arr = np.empty(n_dims, dtype=np.uint8)

    if _AVX2_AVAILABLE:
        _lib.omniring_unpack_nibbles(in_arr, out_arr, n_dims)
    else:
        out_arr[0::2] = in_arr & 0x0F
        out_arr[1::2] = (in_arr >> 4) & 0x0F

    return out_arr


def similarity_avx2(a_packed: np.ndarray, b_packed: np.ndarray) -> int:
    """
    Compute similarity score using 1-cycle AVX2 vpshufb table lookups.
    a_packed, b_packed: 2560 bytes (5120 dimensions).
    """
    if not _AVX2_AVAILABLE:
        raise RuntimeError("AVX2 native library is not available.")
    
    assert len(a_packed) == 2560 and len(b_packed) == 2560
    a = a_packed if (a_packed.flags.c_contiguous and a_packed.dtype == np.uint8) else np.ascontiguousarray(a_packed, dtype=np.uint8)
    b = b_packed if (b_packed.flags.c_contiguous and b_packed.dtype == np.uint8) else np.ascontiguousarray(b_packed, dtype=np.uint8)
    return int(_lib.omniring_similarity_avx2(a, b))


def batch_similarity_avx2(
    query_packed: np.ndarray,
    matrix_packed: np.ndarray,
    out: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute similarity scores between one query and a matrix of candidate vectors.
    query_packed: shape (2560,), uint8
    matrix_packed: shape (N, 2560), uint8
    out: Optional preallocated int32 output array of shape (N,)
    
    Returns:
        scores: shape (N,), int32
    """
    if not _AVX2_AVAILABLE:
        raise RuntimeError("AVX2 native library is not available.")
        
    n_vectors = matrix_packed.shape[0]
    if out is None:
        out = np.empty(n_vectors, dtype=np.int32)
        
    q = query_packed if (query_packed.flags.c_contiguous and query_packed.dtype == np.uint8) else np.ascontiguousarray(query_packed, dtype=np.uint8)
    m = matrix_packed if (matrix_packed.flags.c_contiguous and matrix_packed.dtype == np.uint8) else np.ascontiguousarray(matrix_packed, dtype=np.uint8)

    _lib.omniring_batch_similarity_avx2(q, m, n_vectors, out)
    return out


def bind_avx2(a_packed: np.ndarray, b_packed: np.ndarray, out: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute vector binding: (a + b) mod 16 using AVX2.
    """
    if not _AVX2_AVAILABLE:
        raise RuntimeError("AVX2 native library is not available.")

    assert len(a_packed) == 2560 and len(b_packed) == 2560
    if out is None:
        out = np.empty(2560, dtype=np.uint8)

    a = a_packed if (a_packed.flags.c_contiguous and a_packed.dtype == np.uint8) else np.ascontiguousarray(a_packed, dtype=np.uint8)
    b = b_packed if (b_packed.flags.c_contiguous and b_packed.dtype == np.uint8) else np.ascontiguousarray(b_packed, dtype=np.uint8)

    _lib.omniring_bind_avx2(a, b, out)
    return out


def unbind_avx2(a_packed: np.ndarray, b_packed: np.ndarray, out: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute vector unbinding: (a - b) mod 16 using AVX2.
    """
    if not _AVX2_AVAILABLE:
        raise RuntimeError("AVX2 native library is not available.")

    assert len(a_packed) == 2560 and len(b_packed) == 2560
    if out is None:
        out = np.empty(2560, dtype=np.uint8)

    a = a_packed if (a_packed.flags.c_contiguous and a_packed.dtype == np.uint8) else np.ascontiguousarray(a_packed, dtype=np.uint8)
    b = b_packed if (b_packed.flags.c_contiguous and b_packed.dtype == np.uint8) else np.ascontiguousarray(b_packed, dtype=np.uint8)

    _lib.omniring_unbind_avx2(a, b, out)
    return out


def transcode_ternary_to_phase_c(sign: np.ndarray, active: np.ndarray) -> np.ndarray:
    """
    Transcode 10,240-D ternary vector into 5,120-D 4-bit packed phases (2,560 bytes).
    """
    if not _AVX2_AVAILABLE:
        raise RuntimeError("AVX2 native library is not available.")
    assert len(sign) == 160 and len(active) == 160
    out = np.empty(2560, dtype=np.uint8)
    s = sign if (sign.flags.c_contiguous and sign.dtype == np.uint64) else np.ascontiguousarray(sign, dtype=np.uint64)
    a = active if (active.flags.c_contiguous and active.dtype == np.uint64) else np.ascontiguousarray(active, dtype=np.uint64)
    _lib.omniring_transcode_ternary_to_phase(s, a, out)
    return out


def transcode_phase_to_ternary_c(packed_phases: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transcode 5,120-D 4-bit packed phases into 10,240-D ternary vector (sign, active).
    """
    if not _AVX2_AVAILABLE:
        raise RuntimeError("AVX2 native library is not available.")
    assert len(packed_phases) == 2560
    out_sign = np.empty(160, dtype=np.uint64)
    out_active = np.empty(160, dtype=np.uint64)
    p = packed_phases if (packed_phases.flags.c_contiguous and packed_phases.dtype == np.uint8) else np.ascontiguousarray(packed_phases, dtype=np.uint8)
    _lib.omniring_transcode_phase_to_ternary(p, out_sign, out_active)
    return out_sign, out_active


def ternary_dot_c(sign_a: np.ndarray, active_a: np.ndarray, sign_b: np.ndarray, active_b: np.ndarray) -> int:
    """
    Compute ternary dot product using 64-bit popcounts.
    """
    if not _AVX2_AVAILABLE:
        raise RuntimeError("AVX2 native library is not available.")
    return int(_lib.omniring_ternary_dot(
        sign_a if (sign_a.flags.c_contiguous and sign_a.dtype == np.uint64) else np.ascontiguousarray(sign_a, dtype=np.uint64),
        active_a if (active_a.flags.c_contiguous and active_a.dtype == np.uint64) else np.ascontiguousarray(active_a, dtype=np.uint64),
        sign_b if (sign_b.flags.c_contiguous and sign_b.dtype == np.uint64) else np.ascontiguousarray(sign_b, dtype=np.uint64),
        active_b if (active_b.flags.c_contiguous and active_b.dtype == np.uint64) else np.ascontiguousarray(active_b, dtype=np.uint64)
    ))


def ternary_bind_c(sign_a: np.ndarray, active_a: np.ndarray, sign_b: np.ndarray, active_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute ternary binding (XOR sign, AND active).
    """
    if not _AVX2_AVAILABLE:
        raise RuntimeError("AVX2 native library is not available.")
    out_sign = np.empty(160, dtype=np.uint64)
    out_active = np.empty(160, dtype=np.uint64)
    _lib.omniring_ternary_bind(
        sign_a if (sign_a.flags.c_contiguous and sign_a.dtype == np.uint64) else np.ascontiguousarray(sign_a, dtype=np.uint64),
        active_a if (active_a.flags.c_contiguous and active_a.dtype == np.uint64) else np.ascontiguousarray(active_a, dtype=np.uint64),
        sign_b if (sign_b.flags.c_contiguous and sign_b.dtype == np.uint64) else np.ascontiguousarray(sign_b, dtype=np.uint64),
        active_b if (active_b.flags.c_contiguous and active_b.dtype == np.uint64) else np.ascontiguousarray(active_b, dtype=np.uint64),
        out_sign,
        out_active
    )
    return out_sign, out_active


def set_num_threads(n_threads: int) -> None:
    """Set OpenMP thread count for parallel SIMD loops."""
    if _AVX2_AVAILABLE and _lib is not None:
        _lib.omniring_set_num_threads(ctypes.c_int32(n_threads))


def ternary_gemv_c(
    matrix_sign: np.ndarray,
    matrix_active: np.ndarray,
    vec_sign: np.ndarray,
    vec_active: np.ndarray,
    out: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Batch Ternary GEMV:
    matrix (rows, n_words) x vector (n_words) -> out (rows)
    Supports arbitrary dimensions where n_dims = n_words * 64.
    """
    if not _AVX2_AVAILABLE:
        raise RuntimeError("AVX2 native library is not available.")
    
    ms = matrix_sign if (matrix_sign.flags.c_contiguous and matrix_sign.dtype == np.uint64) else np.ascontiguousarray(matrix_sign, dtype=np.uint64)
    ma = matrix_active if (matrix_active.flags.c_contiguous and matrix_active.dtype == np.uint64) else np.ascontiguousarray(matrix_active, dtype=np.uint64)
    vs = vec_sign if (vec_sign.flags.c_contiguous and vec_sign.dtype == np.uint64) else np.ascontiguousarray(vec_sign, dtype=np.uint64)
    va = vec_active if (vec_active.flags.c_contiguous and vec_active.dtype == np.uint64) else np.ascontiguousarray(vec_active, dtype=np.uint64)

    rows, n_words = ms.shape
    assert ma.shape == (rows, n_words), "matrix_sign and matrix_active shape mismatch"
    assert len(vs) == n_words and len(va) == n_words, "vec length must equal n_words"

    if out is None:
        out = np.empty(rows, dtype=np.int32)
    elif not (out.flags.c_contiguous and out.dtype == np.int32 and len(out) == rows):
        out = np.empty(rows, dtype=np.int32)

    _lib.omniring_ternary_gemv(ms, ma, vs, va, rows, n_words, out)
    return out


def gemv_ternary_int8_c(
    matrix_sign: np.ndarray,
    matrix_active: np.ndarray,
    vec_int8: np.ndarray,
    out: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Ternary-Weight by INT8-Activation GEMV:
    y = W x where W is ternary bitplanes (rows, n_words) and x is int8 (n_words * 64).
    """
    if not _AVX2_AVAILABLE:
        raise RuntimeError("AVX2 native library is not available.")
    
    ms = matrix_sign if (matrix_sign.flags.c_contiguous and matrix_sign.dtype == np.uint64) else np.ascontiguousarray(matrix_sign, dtype=np.uint64)
    ma = matrix_active if (matrix_active.flags.c_contiguous and matrix_active.dtype == np.uint64) else np.ascontiguousarray(matrix_active, dtype=np.uint64)
    x = vec_int8 if (vec_int8.flags.c_contiguous and vec_int8.dtype == np.int8) else np.ascontiguousarray(vec_int8, dtype=np.int8)

    rows, n_words = ms.shape
    assert ma.shape == (rows, n_words), "matrix_sign and matrix_active shape mismatch"
    assert len(x) == n_words * 64, f"vec_int8 length {len(x)} must equal n_words * 64 ({n_words * 64})"

    if out is None:
        out = np.empty(rows, dtype=np.int32)
    elif not (out.flags.c_contiguous and out.dtype == np.int32 and len(out) == rows):
        out = np.empty(rows, dtype=np.int32)

    _lib.omniring_gemv_ternary_int8(ms, ma, x, rows, n_words, out)
    return out
