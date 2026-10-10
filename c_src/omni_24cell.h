#ifndef OMNI_24CELL_H
#define OMNI_24CELL_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// 24-cell group order (Binary Tetrahedral Group 2T)
#define OMNI_2T_ORDER 24

// Coprime ring moduli for the 1D/5-Torus Residue Clock
#define OMNI_NUM_MODULI 5
static const int32_t OMNI_COPRIME_MODULI[OMNI_NUM_MODULI] = {3, 5, 7, 13, 17};
#define OMNI_CRT_PERIOD 23205

// 24x24 Cayley multiplication table for non-commutative causal composition
uint8_t omni_24cell_mult(uint8_t a, uint8_t b);

// Batch non-commutative sequence composition
void omni_24cell_batch_mult(
    const uint8_t* a,
    const uint8_t* b,
    uint8_t* out,
    size_t n
);

// AVX2 Unitary Complex Phasor Harmonic Generator for CRT Clock
// Computes cos(2*pi*j*(pos % p)/p) and sin(2*pi*j*(pos % p)/p)
void omni_phasor_harmonics_avx2(
    const int32_t* positions,
    size_t n_pos,
    float* out_sin_cos,
    size_t dim
);

// AVX2 Superposed Bus Slot Peeler with Extreme-Value Spark Margin Test
// Returns true if (s1 - s2) >= margin_threshold; sets out_best_idx and out_best_margin
bool omni_bus_peel_margin_avx2(
    const int8_t* slot_matrix,
    size_t n_marks,
    size_t mark_dim,
    const int8_t* query,
    float margin_threshold,
    int32_t* out_best_idx,
    float* out_best_margin
);

#ifdef __cplusplus
}
#endif

#endif // OMNI_24CELL_H
