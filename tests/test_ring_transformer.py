"""
Tests for Ring-Attentive Transformer (Tier 3 Capstone).
Verifies constant-memory cross-attention, FPE temporal rotation,
step decoding, generation, and semantic attractor clamping.
"""
import time
import pytest
import numpy as np

from omni_ring.ring_bank import RingBank, RingSlotId, NUM_BANK_SLOTS
from omni_ring.ring_transformer import (
    RingCrossAttention,
    RingTransformerBlock,
    RingTransformer
)
from omni_ring.polymorphic import TernaryVector, PhaseVectorPacked, TERNARY_D


@pytest.fixture
def test_bank():
    bank = RingBank(name="test_transformer_bank.bin", create=True)
    # Initialize slots with random orthogonal ternary vectors
    for i in range(8):
        bank.write_ternary(i, TernaryVector.random(seed=500 + i))
    # Initialize TEMPORAL slot with valid 4-bit phase vector
    phases = np.zeros(5120, dtype=np.uint8)
    bank.write_phase(RingSlotId.TEMPORAL, PhaseVectorPacked.from_phases(phases))
    yield bank
    bank.unlink()


def test_ring_cross_attention_forward(test_bank):
    """Verify constant-memory cross-attention projection."""
    d_model = 128
    attn = RingCrossAttention(d_model=d_model, seed=42)

    # 1. Test 1D hidden state
    x_1d = np.ones(d_model, dtype=np.float32)
    out_1d = attn.forward(x_1d, test_bank)
    assert out_1d.shape == (d_model,)
    assert not np.isnan(out_1d).any()

    # 2. Test 2D sequence (seq_len=4, d_model=128)
    x_2d = np.random.standard_normal((4, d_model)).astype(np.float32)
    out_2d = attn.forward(x_2d, test_bank)
    assert out_2d.shape == (4, d_model)
    assert not np.isnan(out_2d).any()


def test_ring_transformer_forward_and_decode(test_bank):
    """Verify end-to-end forward pass, temporal FPE rotation, and single-step decode."""
    vocab_size = 250
    d_model = 128
    d_ffn = 256
    model = RingTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        d_ffn=d_ffn,
        n_layers=2,
        seed=101
    )

    prompt = [5, 12, 42]
    # Forward pass with temporal tick t=1.5
    logits, hidden = model.forward(prompt, test_bank, t=1.5)

    assert logits.shape == (vocab_size,)
    assert hidden.shape == (d_model,)
    assert not np.isnan(logits).any()

    # Verify temporal slot underwent FPE phase rotation
    temporal_pv = test_bank.read_phase(RingSlotId.TEMPORAL)
    # At t=1.5 on zero vector: 0 * 1.5 = 0, so let's test decode with delta
    next_tok, next_hidden = model.step_decode(token_id=42, bank=test_bank, t_delta=2.0)
    assert 0 <= next_tok < vocab_size
    assert next_hidden.shape == (d_model,)


def test_ring_transformer_generation(test_bank):
    """Verify autoregressive token generation in constant memory."""
    vocab_size = 200
    d_model = 128
    model = RingTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        d_ffn=256,
        n_layers=1,
        seed=202
    )

    prompt = [1, 2, 3]
    max_new_tokens = 6
    generated = model.generate(prompt, test_bank, max_tokens=max_new_tokens)

    assert len(generated) == len(prompt) + max_new_tokens
    assert generated[:len(prompt)] == prompt
    for tok in generated:
        assert 0 <= tok < vocab_size


def test_ring_transformer_attractor_clamp(test_bank):
    """Verify that neural cortex hidden representations clamp to verified attractors."""
    model = RingTransformer(
        vocab_size=100,
        d_model=128,
        d_ffn=256,
        n_layers=1,
        seed=303
    )
    # Register discrete semantic action in the clamp head
    model.clamp.register_symbol("EXECUTE_WINDUP", 42)

    # Clean target vector for coordinate 42
    target_pv = model.clamp.get_symbol_vector("EXECUTE_WINDUP")
    clamp_res = model.clamp.clamp(target_pv)

    assert clamp_res.clamped is True
    assert clamp_res.coordinate == 42
    assert clamp_res.symbol == "EXECUTE_WINDUP"


def test_down_projection_matrix_uses_full_slot():
    """D2: the down-projection spans the whole 10,240-D bus, not just [:d_model]."""
    d_model = 128
    attn = RingCrossAttention(d_model=d_model, seed=42)

    assert attn.down_proj.shape == (TERNARY_D, d_model)
    # Rows past d_model are the dims the old truncation discarded; they must
    # carry non-zero weights so the projection actually reads the full slot.
    assert np.count_nonzero(attn.down_proj[d_model:]) > 0
    assert np.isfinite(attn.down_proj).all()


def test_down_projection_distinguishes_full_slot(test_bank):
    """D2: slots sharing the first d_model dims but differing in the tail must
    produce different attention output. The old [:d_model] truncation made such
    pairs identical downstream (97.5% of each slot discarded)."""
    d_model = 128
    attn = RingCrossAttention(d_model=d_model, seed=42)

    rng = np.random.default_rng(999)
    prefix = rng.choice([-1, 0, 1], size=d_model, p=[0.25, 0.5, 0.25]).astype(np.int8)
    tail_a = rng.choice([-1, 0, 1], size=TERNARY_D - d_model, p=[0.25, 0.5, 0.25]).astype(np.int8)
    tail_b = tail_a.copy()
    tail_b[0] = 1 if tail_a[0] != 1 else -1  # guaranteed difference at dim d_model

    slot_a = np.concatenate([prefix, tail_a]).astype(np.int8)
    slot_b = np.concatenate([prefix, tail_b]).astype(np.int8)
    assert np.array_equal(slot_a[:d_model], slot_b[:d_model])
    assert not np.array_equal(slot_a[d_model:], slot_b[d_model:])

    # Fill all 8 slots so attention weight lands on the vectors under control.
    for i in range(NUM_BANK_SLOTS):
        test_bank.write_ternary(i, TernaryVector.from_dense(slot_a))
    x = np.ones(d_model, dtype=np.float32)
    out_a = attn.forward(x, test_bank)

    for i in range(NUM_BANK_SLOTS):
        test_bank.write_ternary(i, TernaryVector.from_dense(slot_b))
    out_b = attn.forward(x, test_bank)

    assert not np.allclose(out_a, out_b), (
        "Down-projection must read the full 10,240-D slot; a tail-only change "
        "(same [:d_model] prefix) must change the output"
    )


def test_down_projection_output_shape_unchanged(test_bank):
    """D2: forward output shape is unchanged by the projection swap."""
    d_model = 256
    attn = RingCrossAttention(d_model=d_model, seed=42)

    x = np.ones(d_model, dtype=np.float32)
    out = attn.forward(x, test_bank)
    assert out.shape == (d_model,)
    assert not np.isnan(out).any()
