// Exhaustive lattice sift (2026-09-29, TORUS/EXH): score every legal codeword against one slot in a single pass.
// Lattice: one byte per dimension holding the codeword's 4-bit phase index (0..15), row-major (n x d).
// Query:   the slot WITH magnitude, as int8 real/imag parts (phase-only slots lose capacity: 96 vs 384 marks measured).
// Score:   S[n] = sum_d qr[d]*C8[p] + qi[d]*S8[p], with C8/S8 = round(127*cos/sin(2*pi*p/16));
//          Re(conj(w) v) = S * scale / (127*127) where scale = max|v| / 127 used to quantise the query.
#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>

static const int8_t C8[16] = {127, 117, 90, 49, 0, -49, -90, -117, -127, -117, -90, -49, 0, 49, 90, 117};
static const int8_t S8[16] = {0, 49, 90, 117, 127, 117, 90, 49, 0, -49, -90, -117, -127, -117, -90, -49};

static inline int32_t hsum_epi32(__m256i v) {
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
    return _mm_cvtsi128_si32(s);
}

// d must be a multiple of 32. out[n] receives the integer score of codeword n.
void omniring_sift_cplx(const uint8_t *lattice, size_t n, size_t d,
                        const int8_t *qr, const int8_t *qi, int32_t *out) {
    const __m256i lutc = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)C8));
    const __m256i luts = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)S8));
    const __m256i ones = _mm256_set1_epi16(1);
#pragma omp parallel for schedule(static)
    for (long k = 0; k < (long)n; k++) {
        const uint8_t *row = lattice + (size_t)k * d;
        __m256i acc = _mm256_setzero_si256();
        for (size_t i = 0; i < d; i += 32) {
            __m256i p = _mm256_loadu_si256((const __m256i *)(row + i));
            __m256i c = _mm256_shuffle_epi8(lutc, p);
            __m256i s = _mm256_shuffle_epi8(luts, p);
            __m256i r = _mm256_loadu_si256((const __m256i *)(qr + i));
            __m256i m = _mm256_loadu_si256((const __m256i *)(qi + i));
            // u8 x s8 products via |lut| and sign-transferred query (sign_epi8 zeroes where lut == 0: product 0)
            __m256i pc = _mm256_maddubs_epi16(_mm256_abs_epi8(c), _mm256_sign_epi8(r, c));
            __m256i ps = _mm256_maddubs_epi16(_mm256_abs_epi8(s), _mm256_sign_epi8(m, s));
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(pc, ones));
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(ps, ones));
        }
        out[k] = hsum_epi32(acc);
    }
}
