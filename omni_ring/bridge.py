"""
Bridge between neural model activations/embeddings and the OMNIRING Polymorphic Bus.
Provides high-speed projection and quantization from dense/ternary embeddings to 10,240-D Ternary and 5,120-D Phase.
"""
from typing import Optional, Union, Tuple
import numpy as np
from .polymorphic import TernaryVector, PhaseVectorPacked, TERNARY_D, PHASE_D
from .avx2 import pack_nibbles


class ActivationProjector:
    """
    Projects arbitrary-dimensional neural activations (e.g., 768, 1024, 2048, 4096)
    into 10,240-D TernaryVectors and 5,120-D PhaseVectorPacked representations.
    """
    def __init__(
        self,
        input_dim: int,
        ternary_dim: int = TERNARY_D,
        phase_dim: int = PHASE_D,
        sparsity: float = 0.5,
        seed: int = 42
    ):
        self.input_dim = input_dim
        self.ternary_dim = ternary_dim
        self.phase_dim = phase_dim
        self.sparsity = sparsity

        # Generate orthogonal sparse projection matrix for ternary face
        rng = np.random.default_rng(seed)
        # Rademacher or sparse random projection
        raw_matrix = rng.standard_normal((input_dim, ternary_dim), dtype=np.float32)
        # Normalize columns
        self.proj_weights = raw_matrix / np.linalg.norm(raw_matrix, axis=0, keepdims=True)

    def project_to_ternary(self, embedding: np.ndarray, threshold: Optional[float] = None) -> TernaryVector:
        """
        Project a continuous dense embedding into a 10,240-D TernaryVector.
        Uses adaptive thresholding to enforce desired sparsity (default ~50% zeros).
        """
        assert embedding.shape[-1] == self.input_dim, f"Expected input dim {self.input_dim}, got {embedding.shape[-1]}"
        
        # 1D projection: (input_dim,) @ (input_dim, ternary_dim) -> (ternary_dim,)
        projected = np.dot(embedding.astype(np.float32), self.proj_weights)

        if threshold is None:
            # Adaptive threshold based on quantile of absolute values
            abs_vals = np.abs(projected)
            threshold = float(np.quantile(abs_vals, self.sparsity))

        dense = np.zeros(self.ternary_dim, dtype=np.int8)
        dense[projected > threshold] = 1
        dense[projected < -threshold] = -1

        return TernaryVector.from_dense(dense)

    def project_to_phase(self, embedding: np.ndarray) -> PhaseVectorPacked:
        """
        Project a continuous dense embedding into a 5,120-D PhaseVectorPacked.
        Maps the 10,240-D projected ternary vector into phase space via IQ constellation.
        """
        tv = self.project_to_ternary(embedding)
        return tv.to_phase()

    def direct_ternary_to_bus(self, ternary_activations: np.ndarray) -> TernaryVector:
        """
        If the model is natively ternary (BitNet / Bonsai), adapt activations directly.
        Expands or maps activations into 10,240 dimensions.
        """
        n_act = len(ternary_activations)
        dense = np.zeros(self.ternary_dim, dtype=np.int8)
        
        if n_act >= self.ternary_dim:
            dense[:] = ternary_activations[:self.ternary_dim]
        else:
            # Tile or distribute
            reps = (self.ternary_dim + n_act - 1) // n_act
            tiled = np.tile(ternary_activations, reps)
            dense[:] = tiled[:self.ternary_dim]

        return TernaryVector.from_dense(dense)
