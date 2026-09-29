"""
High-Performance Ternary Linear Layer and Weight Matrix for OMNIRING.
Enables sub-microsecond GEMV forward passes over 1.58-bit packed bitplanes
using AVX2 and popcount SIMD kernels.
"""
from __future__ import annotations
import numpy as np
from typing import Optional, Union, Tuple
from .polymorphic import TernaryVector, TERNARY_WORDS
from .avx2 import is_avx2_available, ternary_gemv_c, gemv_ternary_int8_c


class TernaryMatrix:
    """
    Ternary weight matrix stored in native 1.58-bit bitplanes:
    - sign: (rows, n_words) uint64
    - active: (rows, n_words) uint64
    Memory footprint: Exactly 2 bits per weight (0.25 bytes/param).
    """
    def __init__(
        self,
        sign: np.ndarray,
        active: np.ndarray,
        in_features: int,
        out_features: int
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.n_words = in_features // 64
        
        assert in_features % 64 == 0, f"in_features ({in_features}) must be a multiple of 64"
        assert sign.shape == (out_features, self.n_words)
        assert active.shape == (out_features, self.n_words)

        self.sign = np.ascontiguousarray(sign, dtype=np.uint64)
        self.active = np.ascontiguousarray(active, dtype=np.uint64)

    @classmethod
    def from_dense(cls, dense_matrix: np.ndarray) -> TernaryMatrix:
        """
        Create from a dense 2D array of {-1, 0, +1} values.
        dense_matrix shape: (out_features, in_features).
        """
        out_features, in_features = dense_matrix.shape
        assert in_features % 64 == 0, f"in_features ({in_features}) must be a multiple of 64"
        n_words = in_features // 64
        
        sign = np.zeros((out_features, n_words), dtype=np.uint64)
        active = np.zeros((out_features, n_words), dtype=np.uint64)
        
        for w in range(n_words):
            chunk = dense_matrix[:, w * 64:(w + 1) * 64] # shape (out_features, 64)
            # Find positive and negative elements
            is_pos = (chunk > 0).astype(np.uint64)
            is_neg = (chunk < 0).astype(np.uint64)
            
            powers = (1 << np.arange(64, dtype=object)).astype(np.uint64)
            s_word = np.sum(is_neg * powers, axis=1, dtype=np.uint64)
            a_word = np.sum((is_pos | is_neg) * powers, axis=1, dtype=np.uint64)
            
            sign[:, w] = s_word
            active[:, w] = a_word
            
        return cls(sign, active, in_features, out_features)

    @classmethod
    def from_float_weights(cls, weights: np.ndarray, threshold: Optional[float] = None) -> TernaryMatrix:
        """
        Quantize standard floating-point weights into ternary {-1, 0, +1}.
        If threshold is None, uses mean absolute value * 0.7 (standard BitNet b1.58 quantizer).
        """
        if threshold is None:
            threshold = float(np.mean(np.abs(weights)) * 0.7)
            
        dense = np.zeros_like(weights, dtype=np.int8)
        dense[weights > threshold] = 1
        dense[weights < -threshold] = -1
        return cls.from_dense(dense)

    def to_dense(self) -> np.ndarray:
        """Unpack bitplanes back into dense {-1, 0, +1} int8 matrix."""
        dense = np.zeros((self.out_features, self.in_features), dtype=np.int8)
        for w in range(self.n_words):
            s = self.sign[:, w]
            a = self.active[:, w]
            for bit in range(64):
                mask = np.uint64(1 << bit)
                is_active = (a & mask) != 0
                is_neg = (s & mask) != 0
                
                col = w * 64 + bit
                dense[is_active & (~is_neg), col] = 1
                dense[is_active & is_neg, col] = -1
        return dense

    @property
    def nbytes(self) -> int:
        """Total memory consumed by weight bitplanes in bytes."""
        return self.sign.nbytes + self.active.nbytes


class TernaryLinear:
    """
    Drop-in Linear Projection Layer for Ternary Architectures.
    Supports:
    1. Ternary Activations -> Ternary Weights (Pure popcount GEMV)
    2. INT8 Activations -> Ternary Weights (AVX2 Accumulation GEMV)
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        weights: Optional[TernaryMatrix] = None,
        bias: Optional[np.ndarray] = None
    ):
        self.in_features = in_features
        self.out_features = out_features
        
        if weights is not None:
            assert weights.in_features == in_features
            assert weights.out_features == out_features
            self.weights = weights
        else:
            # Initialize empty or random
            n_words = in_features // 64
            sign = np.zeros((out_features, n_words), dtype=np.uint64)
            active = np.zeros((out_features, n_words), dtype=np.uint64)
            self.weights = TernaryMatrix(sign, active, in_features, out_features)
            
        self.bias = bias if bias is None else np.ascontiguousarray(bias, dtype=np.int32)

    def forward_ternary(
        self,
        x: Union[TernaryVector, Tuple[np.ndarray, np.ndarray], List[TernaryVector]],
        out: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Forward pass for ternary activations x in {-1, 0, +1}^N."""
        if isinstance(x, list):
            res = np.empty((len(x), self.out_features), dtype=np.int32)
            for i, tv in enumerate(x):
                res[i] = ternary_gemv_c(self.weights.sign, self.weights.active, tv.sign, tv.active)
                if self.bias is not None:
                    res[i] += self.bias
            return res

        if isinstance(x, TernaryVector):
            v_sign, v_active = x.sign, x.active
        else:
            v_sign, v_active = x

        scores = ternary_gemv_c(
            self.weights.sign,
            self.weights.active,
            v_sign,
            v_active,
            out=out
        )
        if self.bias is not None:
            scores += self.bias
        return scores

    def forward_int8(
        self,
        x: np.ndarray,
        out: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Forward pass for INT8 activations x in [-128, 127]^N or (seq_len, N)."""
        if x.ndim == 2:
            seq_len, in_f = x.shape
            assert in_f == self.in_features
            res = np.empty((seq_len, self.out_features), dtype=np.int32)
            for i in range(seq_len):
                res[i] = gemv_ternary_int8_c(self.weights.sign, self.weights.active, x[i])
                if self.bias is not None:
                    res[i] += self.bias
            return res

        scores = gemv_ternary_int8_c(
            self.weights.sign,
            self.weights.active,
            x,
            out=out
        )
        if self.bias is not None:
            scores += self.bias
        return scores

    def __call__(self, x: Union[TernaryVector, np.ndarray]) -> np.ndarray:
        """Unified dispatch: automatically chooses ternary or int8 GEMV path."""
        if isinstance(x, TernaryVector):
            return self.forward_ternary(x)
        elif x.dtype == np.int8:
            return self.forward_int8(x)
        elif np.issubdtype(x.dtype, np.floating):
            # Quantize float input to INT8
            max_val = np.max(np.abs(x))
            scale = 127.0 / (max_val + 1e-6)
            x_int8 = np.clip(np.round(x * scale), -128, 127).astype(np.int8)
            return self.forward_int8(x_int8)
        else:
            raise TypeError(f"Unsupported activation type: {type(x)} with dtype {getattr(x, 'dtype', None)}")
