"""
Ring-Attentive Transformer (Tier 3 Capstone).
Replaces standard O(N) expanding KV-cache with constant-memory attention
directed at the 20.48 KB Multi-Ring Memory Bank in /dev/shm.
Linear projections run via AVX2 TernaryLinear (Tier 1),
and semantic outputs are clamped by ResonatorAttractorClamp (Tier 2).
"""
from __future__ import annotations
import time
from typing import Optional, List, Tuple, Dict, Union
import numpy as np

from .types import DEFAULT_SPEC
from .polymorphic import TernaryVector, PhaseVectorPacked, TERNARY_D, TERNARY_WORDS
from .ring_bank import RingBank, RingSlotId, NUM_BANK_SLOTS
from .ternary_layer import TernaryMatrix, TernaryLinear
from .resonator_clamp import ResonatorAttractorClamp, ClampResult
from .bridge import ActivationProjector
from .query_bus import fractional_power_encoding


def rms_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Root Mean Square Layer Normalization."""
    variance = np.mean(np.square(x), axis=-1, keepdims=True)
    return x / np.sqrt(variance + eps)


class RingCrossAttention:
    """
    Constant-Memory Cross-Attention over the 8-Slot Ring Bank.
    Instead of attending across an expanding history of past tokens (O(N) memory),
    the query vector attends directly to the 8 cache-resident memory rings (O(1) memory).
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        seed: int = 42
    ):
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert d_model % 64 == 0, f"d_model ({d_model}) must be a multiple of 64"

        # Projection layers (Q, O)
        rng = np.random.default_rng(seed)
        w_q = rng.choice([-1, 0, 1], size=(d_model, d_model), p=[0.25, 0.5, 0.25]).astype(np.int8)
        w_o = rng.choice([-1, 0, 1], size=(d_model, d_model), p=[0.25, 0.5, 0.25]).astype(np.int8)
        
        self.q_proj = TernaryLinear(d_model, d_model, weights=TernaryMatrix.from_dense(w_q))
        self.out_proj = TernaryLinear(d_model, d_model, weights=TernaryMatrix.from_dense(w_o))
        
        # Projector between model dimension and 10,240-D ternary bus space
        self.bus_projector = ActivationProjector(input_dim=d_model, seed=seed)

        # D2 down-projection: full 10,240-D slot -> d_model. Fixed +-1 Rademacher
        # matrix (seed 42 stream, same rng style as w_q/w_o above), unit-norm
        # columns. Replaces the [:d_model] truncation that dropped 97.5% of
        # every ring. ActivationProjector only projects UP (d_model -> 10,240),
        # so it does not fit here; this dedicated matrix is the down direction.
        down_raw = rng.choice([-1, 1], size=(TERNARY_D, self.d_model)).astype(np.float32)
        self.down_proj = down_raw / np.sqrt(TERNARY_D)

    def forward(self, x: np.ndarray, bank: RingBank) -> np.ndarray:
        """
        x: hidden state of shape (d_model,) or (seq_len, d_model).
        bank: the active /dev/shm Multi-Ring Bank.
        """
        is_1d = (x.ndim == 1)
        if is_1d:
            x_seq = x.reshape(1, -1)
        else:
            x_seq = x

        seq_len, _ = x_seq.shape
        outputs = np.empty_like(x_seq)

        for i in range(seq_len):
            token_x = x_seq[i]
            # 1. Project query
            q = self.q_proj(token_x)
            
            # 2. Map query into 10,240-D Ternary Vector for AVX2 cross-ring scan
            q_ternary = self.bus_projector.project_to_ternary(q.astype(np.float32))
            
            # 3. Compute instant AVX2 attention scores over all 8 slots in <150µs
            raw_scores = bank.cross_attention_scan(q_ternary) # shape (8,)
            # Softmax normalization over the 8 memory rings
            scaled = raw_scores.astype(np.float32) / np.sqrt(self.head_dim)
            exp_scores = np.exp(scaled - np.max(scaled))
            attn_weights = exp_scores / np.sum(exp_scores) # shape (8,)

            # 4. Read slot values from bank and form context
            # Project bank values back to d_model
            context = np.zeros(self.d_model, dtype=np.float32)
            for slot_idx in range(NUM_BANK_SLOTS):
                w = attn_weights[slot_idx]
                if w > 0.01: # Sparsity filter
                    slot_tv = bank.read_ternary(slot_idx)
                    # Dense int8 vector of 10,240
                    slot_dense = slot_tv.to_dense().astype(np.float32)
                    # Downproject the full 10,240-D slot to d_model through the
                    # fixed +-1 matrix (replaces [:d_model] truncation).
                    context += w * (slot_dense @ self.down_proj)

            # 5. Output projection
            norm_ctx = rms_norm(context)
            out = self.out_proj(norm_ctx)
            outputs[i] = out

        return outputs[0] if is_1d else outputs


class RingTransformerBlock:
    """A single Transformer Block with RingCrossAttention and AVX2 Ternary FFN."""
    def __init__(self, d_model: int, d_ffn: int, seed: int = 42):
        self.d_model = d_model
        self.attn = RingCrossAttention(d_model=d_model, seed=seed)
        
        # FFN: d_model -> d_ffn -> d_model
        assert d_ffn % 64 == 0
        rng = np.random.default_rng(seed + 1)
        w_ffn1 = rng.choice([-1, 0, 1], size=(d_ffn, d_model), p=[0.25, 0.5, 0.25]).astype(np.int8)
        w_ffn2 = rng.choice([-1, 0, 1], size=(d_model, d_ffn), p=[0.25, 0.5, 0.25]).astype(np.int8)
        
        self.ffn1 = TernaryLinear(d_model, d_ffn, weights=TernaryMatrix.from_dense(w_ffn1))
        self.ffn2 = TernaryLinear(d_ffn, d_model, weights=TernaryMatrix.from_dense(w_ffn2))

    def forward(self, x: np.ndarray, bank: RingBank) -> np.ndarray:
        # Pre-LN Attention
        norm_x = rms_norm(x.astype(np.float32))
        attn_out = self.attn.forward(norm_x, bank)
        x = x + attn_out

        # Pre-LN FFN with SwiGLU / ReLU approximation
        norm_x2 = rms_norm(x.astype(np.float32))
        h = self.ffn1(norm_x2)
        h_act = np.maximum(h, 0) # ReLU
        ffn_out = self.ffn2(h_act.astype(np.float32))
        x = x + ffn_out
        return x


class RingTransformer:
    """
    Ring-Attentive Transformer Cortex.
    Maintains infinite context depth in constant 20.48 KB /dev/shm memory.
    Linear operations use AVX2 integer GEMV, and outputs can be clamped
    to 100% verified discrete codebook attractors.
    """
    def __init__(
        self,
        vocab_size: int = 1000,
        d_model: int = 256,
        d_ffn: int = 512,
        n_layers: int = 2,
        seed: int = 42
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.n_layers = n_layers

        rng = np.random.default_rng(seed)
        # Token embeddings
        self.embeddings = rng.standard_normal((vocab_size, d_model), dtype=np.float32)

        # Transformer layers
        self.layers = [
            RingTransformerBlock(d_model=d_model, d_ffn=d_ffn, seed=seed + 10 * l)
            for l in range(n_layers)
        ]

        # LM Head
        w_head = rng.choice([-1, 0, 1], size=(vocab_size, d_model), p=[0.25, 0.5, 0.25]).astype(np.int8)
        self.lm_head = TernaryLinear(d_model, vocab_size, weights=TernaryMatrix.from_dense(w_head))

        # Semantic Attractor Clamp
        self.clamp = ResonatorAttractorClamp(spec=DEFAULT_SPEC, seed=seed)
        self.projector = ActivationProjector(input_dim=d_model, seed=seed)

    def forward(
        self,
        input_ids: Union[List[int], np.ndarray],
        bank: RingBank,
        t: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass over input sequence.
        Returns:
            (logits, final_hidden_state)
        """
        ids = np.asarray(input_ids, dtype=np.int32)
        x = self.embeddings[ids] # shape (seq_len, d_model)

        # Update Temporal ring if t is provided
        if t != 0.0:
            temporal_pv = bank.read_phase(RingSlotId.TEMPORAL)
            rotated_pv = fractional_power_encoding(temporal_pv, t)
            bank.write_phase(RingSlotId.TEMPORAL, rotated_pv)

        # Pass through Ring Transformer blocks
        for layer in self.layers:
            x = layer.forward(x, bank)

        final_hidden = x[-1] if x.ndim == 2 else x
        norm_hidden = rms_norm(final_hidden.astype(np.float32))
        logits = self.lm_head(norm_hidden)
        return logits, final_hidden

    def step_decode(
        self,
        token_id: int,
        bank: RingBank,
        t_delta: float = 1.0
    ) -> Tuple[int, np.ndarray]:
        """
        Single token decode step in constant O(1) memory.
        No KV-cache expansion.
        """
        logits, hidden = self.forward([token_id], bank, t=t_delta)
        next_token = int(np.argmax(logits))
        return next_token, hidden

    def clamp_action(self, hidden_state: np.ndarray) -> ClampResult:
        """
        Route hidden state through Tier 2 Resonator Attractor Clamp.
        This clamps to a verified discrete attractor when one exists; it is an
        integrity check, not an accuracy guarantee. It cannot tell a correct
        answer from a confidently wrong one, and the weights it runs on are
        seeded pseudo-random, not trained. Abstain when no attractor verifies.
        See reviews/2026-09-28 — OMNIRING-PEER-REVIEW-RETRIEVAL-AND-CONSOLIDATION.md §6.
        """
        return self.clamp.clamp_from_embedding(hidden_state, self.projector)

    def generate(
        self,
        prompt_ids: List[int],
        bank: RingBank,
        max_tokens: int = 16
    ) -> List[int]:
        """Autoregressive generation without KV-cache explosion."""
        curr_ids = list(prompt_ids)
        # Prefill prompt
        logits, _ = self.forward(curr_ids, bank, t=0.0)
        next_tok = int(np.argmax(logits))
        curr_ids.append(next_tok)

        for step in range(1, max_tokens):
            next_tok, _ = self.step_decode(next_tok, bank, t_delta=1.0)
            curr_ids.append(next_tok)

        return curr_ids
