#include "omni_24cell.h"
#include <immintrin.h>
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// 24x24 Cayley multiplication table for binary tetrahedral group 2T
static const uint8_t CAYLEY_2T[24][24] = {
    { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
    { 1,  0,  3,  2,  5,  4,  7,  6, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10,  9,  8},
    { 2,  3,  1,  0,  6,  7,  5,  4, 18, 16, 19, 17, 10,  8, 11,  9, 22, 20, 23, 21, 14, 12, 15, 13},
    { 3,  2,  0,  1,  7,  6,  4,  5, 13, 15, 12, 14, 21, 23, 20, 22,  9, 11,  8, 10, 17, 19, 16, 18},
    { 4,  5,  7,  6,  1,  0,  2,  3, 17, 21,  9, 13, 16, 20,  8, 12, 19, 23, 11, 15, 18, 22, 10, 14},
    { 5,  4,  6,  7,  0,  1,  3,  2, 14, 10, 22, 18, 15, 11, 23, 19, 12,  8, 20, 16, 13,  9, 21, 17},
    { 6,  7,  4,  5,  3,  2,  1,  0, 20, 12, 16,  8, 22, 14, 18, 10, 21, 13, 17,  9, 23, 15, 19, 11},
    { 7,  6,  5,  4,  2,  3,  0,  1, 11, 19, 15, 23,  9, 17, 13, 21, 10, 18, 14, 22,  8, 16, 12, 20},
    { 8, 23, 17, 14, 20, 11, 18, 13, 16,  4,  2,  9,  6, 12, 10,  0,  1, 21, 19,  7, 22,  3,  5, 15},
    { 9, 22, 19, 12, 16, 15, 10, 21,  2, 17, 11,  7,  8,  4,  0, 13, 18,  1,  5, 23,  6, 20, 14,  3},
    {10, 21, 16, 15, 12, 19, 22,  9,  6,  8, 18,  2, 14,  0,  5, 11, 20,  4,  1, 17,  3, 13, 23,  7},
    {11, 20, 18, 13,  8, 23, 14, 17, 10,  2,  5, 19,  0,  9, 15,  7,  6, 16, 22,  1, 12,  4,  3, 21},
    {12, 19,  9, 22, 21, 10, 16, 15,  4, 13,  8,  0, 20,  3,  6, 14, 17,  7,  2, 11,  1, 23, 18,  5},
    {13, 18, 11, 20, 17, 14,  8, 23,  9,  7,  0, 15,  4, 21, 12,  3,  2, 19, 10,  5, 16,  1,  6, 22},
    {14, 17,  8, 23, 13, 18, 20, 11, 12,  0,  6, 10,  3, 15, 22,  5,  4,  9, 16,  2, 21,  7,  1, 19},
    {15, 16, 10, 21,  9, 22, 12, 19,  0, 11, 14,  5, 13,  7,  3, 23,  8,  2,  6, 18,  4, 17, 20,  1},
    {16, 15, 21, 10, 22,  9, 19, 12,  1, 20, 17,  4, 18,  6,  2,  8, 23,  3,  7, 13,  5, 14, 11,  0},
    {17, 14, 23,  8, 18, 13, 11, 20, 19,  1,  7, 21,  2, 16,  9,  4,  5, 22, 15,  3, 10,  6,  0, 12},
    {18, 13, 20, 11, 14, 17, 23,  8, 22,  6,  1, 16,  5, 10, 19,  2,  3, 12, 21,  4, 15,  0,  7,  9},
    {19, 12, 22,  9, 10, 21, 15, 16,  5, 18, 23,  1, 11,  2,  7, 17, 14,  6,  3, 20,  0,  8, 13,  4},
    {20, 11, 13, 18, 23,  8, 17, 14, 21,  3,  4, 12,  1, 22, 16,  6,  7, 15,  9,  0, 19,  5,  2, 10},
    {21, 10, 15, 16, 19, 12,  9, 22,  7, 23, 13,  3, 17,  1,  4, 20, 11,  5,  0, 14,  2, 18,  8,  6},
    {22,  9, 12, 19, 15, 16, 21, 10,  3, 14, 20,  6, 23,  5,  1, 18, 13,  0,  4,  8,  7, 11, 17,  2},
    {23,  8, 14, 17, 11, 20, 13, 18, 15,  5,  3, 22,  7, 19, 21,  1,  0, 10, 12,  6,  9,  2,  4, 16}
};

uint8_t omni_24cell_mult(uint8_t a, uint8_t b) {
    if (a >= 24 || b >= 24) return 0;
    return CAYLEY_2T[a][b];
}

void omni_24cell_batch_mult(
    const uint8_t* a,
    const uint8_t* b,
    uint8_t* out,
    size_t n
) {
    for (size_t i = 0; i < n; i++) {
        uint8_t va = a[i];
        uint8_t vb = b[i];
        out[i] = (va < 24 && vb < 24) ? CAYLEY_2T[va][vb] : 0;
    }
}

void omni_phasor_harmonics_avx2(
    const int32_t* positions,
    size_t n_pos,
    float* out_sin_cos,
    size_t dim
) {
    // dim must be even; allocates channels across coprime moduli {3, 5, 7, 13, 17}
    size_t pairs_per_mod = (dim / 2) / OMNI_NUM_MODULI;
    if (pairs_per_mod < 1) pairs_per_mod = 1;

    for (size_t i = 0; i < n_pos; i++) {
        int32_t pos = positions[i];
        float* row = out_sin_cos + i * dim;
        size_t ch = 0;

        for (int m = 0; m < OMNI_NUM_MODULI && ch < dim; m++) {
            int32_t p = OMNI_COPRIME_MODULI[m];
            int32_t res = pos % p;
            if (res < 0) res += p;

            for (size_t h = 1; h <= pairs_per_mod && (ch + 1) < dim; h++) {
                double theta = (2.0 * M_PI * (double)(h * res)) / (double)p;
                row[ch++] = (float)cos(theta);
                row[ch++] = (float)sin(theta);
            }
        }
        // Fill any remaining residual channels with global low-frequency phase
        while (ch + 1 < dim) {
            double theta = (2.0 * M_PI * (double)pos) / (double)OMNI_CRT_PERIOD;
            row[ch++] = (float)cos(theta);
            row[ch++] = (float)sin(theta);
        }
    }
}

bool omni_bus_peel_margin_avx2(
    const int8_t* slot_matrix,
    size_t n_marks,
    size_t mark_dim,
    const int8_t* query,
    float margin_threshold,
    int32_t* out_best_idx,
    float* out_best_margin
) {
    if (n_marks == 0) {
        if (out_best_idx) *out_best_idx = -1;
        if (out_best_margin) *out_best_margin = 0.0f;
        return false;
    }

    int32_t top1_score = -2147483647;
    int32_t top2_score = -2147483647;
    int32_t top1_idx = -1;

    for (size_t m = 0; m < n_marks; m++) {
        const int8_t* mark = slot_matrix + m * mark_dim;
        __m256i acc = _mm256_setzero_si256();

        size_t d = 0;
        for (; d + 32 <= mark_dim; d += 32) {
            __m256i q_vec = _mm256_loadu_si256((const __m256i*)(query + d));
            __m256i m_vec = _mm256_loadu_si256((const __m256i*)(mark + d));

            // Unpack low 16 bytes to epi16 and multiply
            __m256i q_low = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(q_vec));
            __m256i m_low = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(m_vec));
            __m256i prod_low = _mm256_madd_epi16(q_low, m_low);
            acc = _mm256_add_epi32(acc, prod_low);

            // Unpack high 16 bytes to epi16 and multiply
            __m256i q_high = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(q_vec, 1));
            __m256i m_high = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(m_vec, 1));
            __m256i prod_high = _mm256_madd_epi16(q_high, m_high);
            acc = _mm256_add_epi32(acc, prod_high);
        }

        int32_t buf[8];
        _mm256_storeu_si256((__m256i*)buf, acc);
        int32_t score = buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7];

        for (; d < mark_dim; d++) {
            score += ((int32_t)query[d]) * ((int32_t)mark[d]);
        }

        if (score > top1_score) {
            top2_score = top1_score;
            top1_score = score;
            top1_idx = (int32_t)m;
        } else if (score > top2_score) {
            top2_score = score;
        }
    }

    float norm_factor = (float)mark_dim * 127.0f * 127.0f;  /* seat fix 2026-10-03: int8 x int8 products scale by 127^2, not 127 -> margin was ~127x too large and the spark gate never closed */
    float norm_top1 = (float)top1_score / norm_factor;
    float norm_top2 = (n_marks > 1) ? ((float)top2_score / norm_factor) : 0.0f;
    float margin = norm_top1 - norm_top2;

    if (out_best_idx) *out_best_idx = top1_idx;
    if (out_best_margin) *out_best_margin = margin;

    return (margin >= margin_threshold);
}
