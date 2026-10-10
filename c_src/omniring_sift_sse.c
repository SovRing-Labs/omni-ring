// Sandy Bridge dialect of the SuperSeed sift (2026-09-30; J's OptiPlex 790 farm: AVX but NO AVX2 / FMA). Same arithmetic
// as omniring_sift_cplx / omniring_sift_p4, at 128 bits with SSSE3 (pshufb lookup, maddubs, sign) — must produce identical
// integer scores (golden). Build: gcc -O3 -march=sandybridge -fopenmp -fPIC -shared omniring_sift_sse.c -o libomniring_sse.so
#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>

static const int8_t C8[16] = {127, 117, 90, 49, 0, -49, -90, -117, -127, -117, -90, -49, 0, 49, 90, 117};
static const int8_t S8[16] = {0, 49, 90, 117, 127, 117, 90, 49, 0, -49, -90, -117, -127, -117, -90, -49};

static inline int32_t hsum128(__m128i v) {
    v = _mm_add_epi32(v, _mm_shuffle_epi32(v, 0x4E)); v = _mm_add_epi32(v, _mm_shuffle_epi32(v, 0xB1));
    return _mm_cvtsi128_si32(v);
}

static inline __m128i acc16(__m128i p, const int8_t *qr, const int8_t *qi, __m128i lc, __m128i ls, __m128i ones, __m128i acc) {
    __m128i c = _mm_shuffle_epi8(lc, p), s = _mm_shuffle_epi8(ls, p);
    __m128i r = _mm_loadu_si128((const __m128i *)qr), m = _mm_loadu_si128((const __m128i *)qi);
    acc = _mm_add_epi32(acc, _mm_madd_epi16(_mm_maddubs_epi16(_mm_abs_epi8(c), _mm_sign_epi8(r, c)), ones));
    return _mm_add_epi32(acc, _mm_madd_epi16(_mm_maddubs_epi16(_mm_abs_epi8(s), _mm_sign_epi8(m, s)), ones));
}

// unpacked lattice (one phase per byte), d % 16 == 0
void omniring_sift_cplx_sse(const uint8_t *lattice, size_t n, size_t d, const int8_t *qr, const int8_t *qi, int32_t *out) {
    const __m128i lc = _mm_loadu_si128((const __m128i *)C8), ls = _mm_loadu_si128((const __m128i *)S8), ones = _mm_set1_epi16(1);
#pragma omp parallel for schedule(static)
    for (long k = 0; k < (long)n; k++) {
        const uint8_t *row = lattice + (size_t)k * d; __m128i acc = _mm_setzero_si128();
        for (size_t i = 0; i < d; i += 16) acc = acc16(_mm_loadu_si128((const __m128i *)(row + i)), qr + i, qi + i, lc, ls, ones, acc);
        out[k] = hsum128(acc);
    }
}

// packed v1.1 lattice (block-64 split: byte j of a block = phase[64b+j] | phase[64b+32+j] << 4), d_used % 64 == 0
void omniring_sift_p4_sse(const uint8_t *packed, size_t n, size_t stride, size_t d_used, const int8_t *qr, const int8_t *qi, int32_t *out) {
    const __m128i lc = _mm_loadu_si128((const __m128i *)C8), ls = _mm_loadu_si128((const __m128i *)S8), ones = _mm_set1_epi16(1);
    const __m128i lo = _mm_set1_epi8(0x0F);
#pragma omp parallel for schedule(static)
    for (long k = 0; k < (long)n; k++) {
        const uint8_t *row = packed + (size_t)k * stride; __m128i acc = _mm_setzero_si128();
        for (size_t b = 0; b < d_used; b += 64)
            for (int h = 0; h < 2; h++) {                    // two 16-byte loads cover the block's 32 bytes
                __m128i v = _mm_loadu_si128((const __m128i *)(row + b / 2 + 16 * h));
                acc = acc16(_mm_and_si128(v, lo), qr + b + 16 * h, qi + b + 16 * h, lc, ls, ones, acc);
                acc = acc16(_mm_and_si128(_mm_srli_epi16(v, 4), lo), qr + b + 32 + 16 * h, qi + b + 32 + 16 * h, lc, ls, ones, acc);
            }
        out[k] = hsum128(acc);
    }
}
