"""
OMNIRING — Coprime Resonator Hyperdimensional Engine.

"We are resonators; everything we build then becomes another resonator to build off from. Develop us well."
"""

import os

# Limit BLAS threading overhead for small matrix/vector operations
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

try:
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1, user_api="blas")
except Exception:
    pass

from .types import PhaseVector, ResidueTuple, RingSpec, DEFAULT_SPEC, ResonatorResult
from .codebook import (
    generate_rotations,
    generate_codebooks,
    generate_quantized_codebook,
    generate_direct_codebooks,
)
from .chain import bind, unbind, encode, get_tick, advance
from .resonator import decode_soft
from .crt import modular_inverse, solve_crt, direct_probe
from .check import check_anomaly, run_full_error_detection_test, ResidueDesyncError
from .shm_bridge import ShmBridge
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
    set_num_threads,
    ternary_gemv_c,
    gemv_ternary_int8_c,
)
from .polymorphic import (
    TernaryVector,
    PhaseVectorPacked,
    PolymorphicSlot,
    PolymorphicBus,
    SLOT_BYTES,
    TERNARY_D,
    PHASE_D,
)
from .bridge import ActivationProjector
from .retrieval import GatedMemoryEngine, GatedRecallResult
from .ternary_layer import TernaryMatrix, TernaryLinear
from .affinity import set_physical_affinity, configure_pacing, pinned_pacing
from .query_bus import QueryBus, fractional_power_encoding
from .ring_bank import RingBank, RingSlotId
from .resonator_clamp import ResonatorAttractorClamp, ClampResult
from .ring_transformer import RingCrossAttention, RingTransformerBlock, RingTransformer

__all__ = [
    "PhaseVector",
    "ResidueTuple",
    "RingSpec",
    "DEFAULT_SPEC",
    "ResonatorResult",
    "generate_rotations",
    "generate_codebooks",
    "generate_quantized_codebook",
    "generate_direct_codebooks",
    "bind",
    "unbind",
    "encode",
    "get_tick",
    "advance",
    "decode_soft",
    "modular_inverse",
    "solve_crt",
    "direct_probe",
    "check_anomaly",
    "run_full_error_detection_test",
    "ResidueDesyncError",
    "ShmBridge",
    "is_avx2_available",
    "pack_nibbles",
    "unpack_nibbles",
    "similarity_avx2",
    "batch_similarity_avx2",
    "bind_avx2",
    "unbind_avx2",
    "transcode_ternary_to_phase_c",
    "transcode_phase_to_ternary_c",
    "ternary_dot_c",
    "ternary_bind_c",
    "set_num_threads",
    "ternary_gemv_c",
    "gemv_ternary_int8_c",
    "TernaryVector",
    "PhaseVectorPacked",
    "PolymorphicSlot",
    "PolymorphicBus",
    "SLOT_BYTES",
    "TERNARY_D",
    "PHASE_D",
    "ActivationProjector",
    "GatedMemoryEngine",
    "GatedRecallResult",
    "TernaryMatrix",
    "TernaryLinear",
    "set_physical_affinity",
    "configure_pacing",
    "pinned_pacing",
    "QueryBus",
    "fractional_power_encoding",
    "RingBank",
    "RingSlotId",
    "ResonatorAttractorClamp",
    "ClampResult",
    "RingCrossAttention",
    "RingTransformerBlock",
    "RingTransformer",
]
