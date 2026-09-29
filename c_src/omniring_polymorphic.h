#ifndef OMNIRING_POLYMORPHIC_H
#define OMNIRING_POLYMORPHIC_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#define OMNIRING_SLOT_BYTES 2560
#define OMNIRING_TERNARY_D  10240
#define OMNIRING_PHASE_D    5120
#define OMNIRING_TERNARY_WORDS 160 // 10240 / 64

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Polymorphic Slot Union: Exactly 2,560 bytes.
 * Matches VSA_HV_BYTES in vsa_kernel.h and /dev/shm/vsa_matrix_bus.
 */
typedef union {
    // Face 1: Ternary Representation (10,240 dimensions)
    // Stored as sign plane (1280 bytes) followed by active/zero plane (1280 bytes)
    struct {
        uint64_t sign[OMNIRING_TERNARY_WORDS];
        uint64_t active[OMNIRING_TERNARY_WORDS];
    } ternary;

    // Face 2: Phase Representation (5,120 dimensions)
    // 4-bit nibbles (K=16), 2 per byte = 2,560 bytes
    struct {
        uint8_t packed_phases[OMNIRING_SLOT_BYTES];
    } phase;

    // Raw byte buffer representation
    uint8_t bytes[OMNIRING_SLOT_BYTES];
} __attribute__((aligned(64))) omniring_polymorphic_slot_t;

/*
 * Transcoding Functions:
 * Convert between 10,240-D Ternary and 5,120-D 4-bit Phase formats.
 */

// Transcode Ternary (sign + active) into 5,120 packed 4-bit phases (2,560 bytes)
// Each pair of trits (2*k, 2*k+1) forms an IQ phasor mapped to {0, 2, 4, 6, 8, 10, 12, 14}.
// The 8 active pairs map bijectively onto those 8 even phases.
// Quiescent (0, 0) trits -- the neutral pair -- map to phase 0, the additive identity,
// so the null ternary block yields the null phase block regardless of its position.
// (OSEAM-D1-TRANSCODE: replaced a position-dependent pseudo-random phase with 0.)
// Consequence: phase 0 is shared with the active pair (+1, 0), so a (0, 0) block
// round-trips back through omniring_transcode_phase_to_ternary as (+1, 0).
void omniring_transcode_ternary_to_phase(
    const uint64_t* sign,
    const uint64_t* active,
    uint8_t* out_packed_phases
);

// Transcode 5,120 packed 4-bit phases into 10,240-D Ternary (sign + active)
// Phase angles are mapped to the nearest constellation points in {-1, 0, +1}^2.
void omniring_transcode_phase_to_ternary(
    const uint8_t* packed_phases,
    uint64_t* out_sign,
    uint64_t* out_active
);

// Fast Ternary Dot Product (popcount-based)
// returns score in [-10240, +10240]
int32_t omniring_ternary_dot(
    const uint64_t* sign_a, const uint64_t* active_a,
    const uint64_t* sign_b, const uint64_t* active_b
);

// Fast Ternary Bind (XOR sign, AND active)
void omniring_ternary_bind(
    const uint64_t* sign_a, const uint64_t* active_a,
    const uint64_t* sign_b, const uint64_t* active_b,
    uint64_t* out_sign, uint64_t* out_active
);

// Configure OpenMP worker threads for SIMD GEMV operations
void omniring_set_num_threads(int32_t n_threads);

// Batch Ternary GEMV: matrix (rows x n_words) multiplied by vector (1 x n_words)
// Computes ternary dot product for each row using AVX2/popcount.
void omniring_ternary_gemv(
    const uint64_t* matrix_sign,
    const uint64_t* matrix_active,
    const uint64_t* vec_sign,
    const uint64_t* vec_active,
    size_t rows,
    size_t n_words,
    int32_t* out_scores
);

// Ternary-Weight by INT8-Activation GEMV:
// Computes y[i] = sum_{j} W[i, j] * X[j] where W is ternary bitplanes and X is int8_t.
// n_dims = n_words * 64.
void omniring_gemv_ternary_int8(
    const uint64_t* matrix_sign,
    const uint64_t* matrix_active,
    const int8_t* vec_int8,
    size_t rows,
    size_t n_words,
    int32_t* out_scores
);

#ifdef __cplusplus
}
#endif

#endif // OMNIRING_POLYMORPHIC_H
