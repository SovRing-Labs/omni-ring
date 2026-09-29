#include "omniring_polymorphic.h"
#include <immintrin.h>
#include <string.h>
#include <omp.h>

// 2D lookup table: [t0 + 1][t1 + 1] -> phase angle in [0..15]
// t0, t1 in {-1, 0, +1}
// The 8 active pairs map bijectively onto the 8 even phases {0,2,4,6,8,10,12,14}.
// The neutral pair (0, 0) maps to phase 0 -- the additive identity -- so the null
// ternary block yields the null phase block, deterministically and independently of
// its position in the vector. This is the single source of truth for the mapping;
// src/polymorphic.py:TERNARY_TO_PHASE_LUT mirrors it exactly for the no-AVX2 fallback.
// (OSEAM-D1-TRANSCODE: (0,0) was 0xFF, a sentinel that triggered a position-dependent
//  pseudo-random phase; that fabricated a non-zero phase for the null vector.)
static const uint8_t TERNARY_TO_PHASE_LUT[3][3] = {
    // t1 = -1,    t1 = 0,    t1 = +1
    {    10,          8,          6     }, // t0 = -1
    {    12,          0,          4     }, // t0 =  0   (0,0) -> 0
    {    14,          0,          2     }  // t0 = +1
};

// 16-entry lookup table for phase -> ternary (t0, t1)
// Perfectly inverts the 8 active constellation points
static const int8_t PHASE_TO_T0[16] = {
    1,  1,  1,  0,  0,  0, -1, -1, -1, -1, -1,  0,  0,  0,  1,  1
};

static const int8_t PHASE_TO_T1[16] = {
    0,  1,  1,  1,  1,  1,  1,  0,  0, -1, -1, -1, -1, -1, -1,  0
};

void omniring_transcode_ternary_to_phase(
    const uint64_t* sign,
    const uint64_t* active,
    uint8_t* out_packed_phases
) {
    size_t out_byte_idx = 0;

    // 160 uint64 words = 10,240 bits = 5,120 pairs
    // Each word has 32 pairs of trits -> 16 bytes of output (2 nibbles per byte)
    for (size_t w = 0; w < OMNIRING_TERNARY_WORDS; w++) {
        uint64_t s = sign[w];
        uint64_t a = active[w];

        for (size_t j = 0; j < 32; j += 2) {
            // First pair (dim 2*k, 2*k+1) -> low nibble
            uint64_t act0 = (a >> (2 * j)) & 1ULL;
            uint64_t sgn0 = (s >> (2 * j)) & 1ULL;
            int t0 = act0 ? (sgn0 ? -1 : 1) : 0;

            uint64_t act1 = (a >> (2 * j + 1)) & 1ULL;
            uint64_t sgn1 = (s >> (2 * j + 1)) & 1ULL;
            int t1 = act1 ? (sgn1 ? -1 : 1) : 0;

            uint8_t phase_low = TERNARY_TO_PHASE_LUT[t0 + 1][t1 + 1];

            // Second pair (dim 2*k+2, 2*k+3) -> high nibble
            size_t j2 = j + 1;
            uint64_t act2 = (a >> (2 * j2)) & 1ULL;
            uint64_t sgn2 = (s >> (2 * j2)) & 1ULL;
            int t2 = act2 ? (sgn2 ? -1 : 1) : 0;

            uint64_t act3 = (a >> (2 * j2 + 1)) & 1ULL;
            uint64_t sgn3 = (s >> (2 * j2 + 1)) & 1ULL;
            int t3 = act3 ? (sgn3 ? -1 : 1) : 0;

            uint8_t phase_high = TERNARY_TO_PHASE_LUT[t2 + 1][t3 + 1];

            out_packed_phases[out_byte_idx++] = (phase_low & 0x0F) | ((phase_high & 0x0F) << 4);
        }
    }
}

void omniring_transcode_phase_to_ternary(
    const uint8_t* packed_phases,
    uint64_t* out_sign,
    uint64_t* out_active
) {
    memset(out_sign, 0, OMNIRING_TERNARY_WORDS * sizeof(uint64_t));
    memset(out_active, 0, OMNIRING_TERNARY_WORDS * sizeof(uint64_t));

    size_t in_byte_idx = 0;
    for (size_t w = 0; w < OMNIRING_TERNARY_WORDS; w++) {
        uint64_t s = 0;
        uint64_t a = 0;

        for (size_t j = 0; j < 32; j += 2) {
            uint8_t byte_val = packed_phases[in_byte_idx++];
            uint8_t phase_low = byte_val & 0x0F;
            uint8_t phase_high = (byte_val >> 4) & 0x0F;

            // Recover pair 1: t0, t1
            int8_t t0 = PHASE_TO_T0[phase_low];
            int8_t t1 = PHASE_TO_T1[phase_low];

            if (t0 != 0) {
                a |= (1ULL << (2 * j));
                if (t0 < 0) s |= (1ULL << (2 * j));
            }
            if (t1 != 0) {
                a |= (1ULL << (2 * j + 1));
                if (t1 < 0) s |= (1ULL << (2 * j + 1));
            }

            // Recover pair 2: t2, t3
            size_t j2 = j + 1;
            int8_t t2 = PHASE_TO_T0[phase_high];
            int8_t t3 = PHASE_TO_T1[phase_high];

            if (t2 != 0) {
                a |= (1ULL << (2 * j2));
                if (t2 < 0) s |= (1ULL << (2 * j2));
            }
            if (t3 != 0) {
                a |= (1ULL << (2 * j2 + 1));
                if (t3 < 0) s |= (1ULL << (2 * j2 + 1));
            }
        }
        out_sign[w] = s;
        out_active[w] = a;
    }
}

int32_t omniring_ternary_dot(
    const uint64_t* sign_a, const uint64_t* active_a,
    const uint64_t* sign_b, const uint64_t* active_b
) {
    int32_t dot = 0;
    for (size_t w = 0; w < OMNIRING_TERNARY_WORDS; w++) {
        uint64_t active_both = active_a[w] & active_b[w];
        uint64_t diff_sign = sign_a[w] ^ sign_b[w];

        uint64_t pos_matches = active_both & (~diff_sign);
        uint64_t neg_matches = active_both & diff_sign;

        dot += (int32_t)_mm_popcnt_u64(pos_matches) - (int32_t)_mm_popcnt_u64(neg_matches);
    }
    return dot;
}

void omniring_ternary_bind(
    const uint64_t* sign_a, const uint64_t* active_a,
    const uint64_t* sign_b, const uint64_t* active_b,
    uint64_t* out_sign, uint64_t* out_active
) {
    for (size_t w = 0; w < OMNIRING_TERNARY_WORDS; w++) {
        out_sign[w] = sign_a[w] ^ sign_b[w];
        out_active[w] = active_a[w] & active_b[w];
    }
}

void omniring_set_num_threads(int32_t n_threads) {
    if (n_threads > 0) {
        omp_set_num_threads(n_threads);
    }
}

void omniring_ternary_gemv(
    const uint64_t* matrix_sign,
    const uint64_t* matrix_active,
    const uint64_t* vec_sign,
    const uint64_t* vec_active,
    size_t rows,
    size_t n_words,
    int32_t* out_scores
) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < rows; i++) {
        int32_t dot = 0;
        size_t row_offset = i * n_words;
        for (size_t w = 0; w < n_words; w++) {
            uint64_t active_both = matrix_active[row_offset + w] & vec_active[w];
            uint64_t diff_sign = matrix_sign[row_offset + w] ^ vec_sign[w];

            uint64_t pos_matches = active_both & (~diff_sign);
            uint64_t neg_matches = active_both & diff_sign;

            dot += (int32_t)_mm_popcnt_u64(pos_matches) - (int32_t)_mm_popcnt_u64(neg_matches);
        }
        out_scores[i] = dot;
    }
}

static const int8_t SHUF_MASK_256[32] __attribute__((aligned(32))) = {
    0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1,
    2, 2, 2, 2, 2, 2, 2, 2,
    3, 3, 3, 3, 3, 3, 3, 3
};
static const int8_t BIT_TEST_256[32] __attribute__((aligned(32))) = {
    1, 2, 4, 8, 16, 32, 64, (int8_t)128,
    1, 2, 4, 8, 16, 32, 64, (int8_t)128,
    1, 2, 4, 8, 16, 32, 64, (int8_t)128,
    1, 2, 4, 8, 16, 32, 64, (int8_t)128
};

static inline __m256i expand_32_bits_to_masks(uint32_t val, __m256i shuf, __m256i bit_test) {
    __m256i v = _mm256_set1_epi32((int)val);
    __m256i broadcast = _mm256_shuffle_epi8(v, shuf);
    __m256i tested = _mm256_and_si256(broadcast, bit_test);
    return _mm256_cmpeq_epi8(tested, bit_test);
}

void omniring_gemv_ternary_int8(
    const uint64_t* matrix_sign,
    const uint64_t* matrix_active,
    const int8_t* vec_int8,
    size_t rows,
    size_t n_words,
    int32_t* out_scores
) {
    __m256i shuf = _mm256_load_si256((const __m256i*)SHUF_MASK_256);
    __m256i bit_test = _mm256_load_si256((const __m256i*)BIT_TEST_256);
    __m256i ones = _mm256_set1_epi16(1);

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < rows; i++) {
        __m256i acc = _mm256_setzero_si256();
        size_t row_offset = i * n_words;

        for (size_t w = 0; w < n_words; w++) {
            uint64_t sign = matrix_sign[row_offset + w];
            uint64_t active = matrix_active[row_offset + w];
            uint64_t pos = active & (~sign);
            uint64_t neg = active & sign;

            size_t base = w * 64;

            // Chunk 0 (low 32 trits)
            uint32_t pos0 = (uint32_t)pos;
            uint32_t neg0 = (uint32_t)neg;
            __m256i vx0 = _mm256_loadu_si256((const __m256i*)(vec_int8 + base));
            __m256i m_pos0 = expand_32_bits_to_masks(pos0, shuf, bit_test);
            __m256i m_neg0 = expand_32_bits_to_masks(neg0, shuf, bit_test);
            __m256i vp0 = _mm256_and_si256(vx0, m_pos0);
            __m256i vn0 = _mm256_and_si256(vx0, m_neg0);

            __m256i p0_lo = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vp0, 0));
            __m256i n0_lo = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vn0, 0));
            __m256i d0_lo = _mm256_sub_epi16(p0_lo, n0_lo);

            __m256i p0_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vp0, 1));
            __m256i n0_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vn0, 1));
            __m256i d0_hi = _mm256_sub_epi16(p0_hi, n0_hi);

            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(d0_lo, ones));
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(d0_hi, ones));

            // Chunk 1 (high 32 trits)
            uint32_t pos1 = (uint32_t)(pos >> 32);
            uint32_t neg1 = (uint32_t)(neg >> 32);
            __m256i vx1 = _mm256_loadu_si256((const __m256i*)(vec_int8 + base + 32));
            __m256i m_pos1 = expand_32_bits_to_masks(pos1, shuf, bit_test);
            __m256i m_neg1 = expand_32_bits_to_masks(neg1, shuf, bit_test);
            __m256i vp1 = _mm256_and_si256(vx1, m_pos1);
            __m256i vn1 = _mm256_and_si256(vx1, m_neg1);

            __m256i p1_lo = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vp1, 0));
            __m256i n1_lo = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vn1, 0));
            __m256i d1_lo = _mm256_sub_epi16(p1_lo, n1_lo);

            __m256i p1_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vp1, 1));
            __m256i n1_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vn1, 1));
            __m256i d1_hi = _mm256_sub_epi16(p1_hi, n1_hi);

            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(d1_lo, ones));
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(d1_hi, ones));
        }

        // Horizontal sum of acc
        __m128i hi128 = _mm256_extracti128_si256(acc, 1);
        __m128i lo128 = _mm256_castsi256_si128(acc);
        __m128i sum128 = _mm_add_epi32(hi128, lo128);
        sum128 = _mm_hadd_epi32(sum128, sum128);
        sum128 = _mm_hadd_epi32(sum128, sum128);
        out_scores[i] = _mm_cvtsi128_si32(sum128);
    }
}
