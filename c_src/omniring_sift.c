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

// SuperSeed v1 16-sector phase of integer (re, im) pairs, n elements (2026-09-30). Must equal seed.sector16 exactly.
// sub = number of quadrant boundaries (floor(tan * 2^16) at 11.25/33.75/56.25/78.75 deg) that |im|/|re| exceeds.
void omniring_sector16(const int32_t *re, const int32_t *im, size_t n, uint8_t *out) {
    static const int64_t T[4] = {13035, 43789, 98081, 329471};
#pragma omp parallel for schedule(static)
    for (long i = 0; i < (long)n; i++) {
        int64_t x = re[i], y = im[i];
        int64_t ax = x < 0 ? -x : x, ay = y < 0 ? -y : y;
        int sub = 0;
        for (int k = 0; k < 4; k++) sub += (ay << 16) > ax * T[k];
        int p;
        if (x >= 0 && y >= 0) p = sub;
        else if (x < 0 && y >= 0) p = 8 - sub;
        else if (x < 0) p = 8 + sub;
        else p = 16 - sub;
        out[i] = (uint8_t)(p & 15);
    }
}

// Batched phase-query sift (2026-09-30): nq queries (phase indices, nq x d) against a lattice (n x d) in ONE parallel region,
// so threads stay busy instead of waking per query. out is nq x n int32. Same arithmetic as omniring_sift_cplx.
void omniring_sift_batch(const uint8_t *lattice, size_t n, size_t d, const uint8_t *queries, size_t nq, int32_t *out) {
    const __m256i lutc = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)C8));
    const __m256i luts = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)S8));
    const __m256i ones = _mm256_set1_epi16(1);
#pragma omp parallel for schedule(dynamic, 1)
    for (long qn = 0; qn < (long)nq; qn++) {
        const uint8_t *qp = queries + (size_t)qn * d;
        int8_t qr[8192], qi[8192];            // d <= 8192
        for (size_t i = 0; i < d; i++) { qr[i] = C8[qp[i] & 15]; qi[i] = S8[qp[i] & 15]; }
        for (size_t k = 0; k < n; k++) {
            const uint8_t *row = lattice + k * d;
            __m256i acc = _mm256_setzero_si256();
            for (size_t i = 0; i < d; i += 32) {
                __m256i p = _mm256_loadu_si256((const __m256i *)(row + i));
                __m256i c = _mm256_shuffle_epi8(lutc, p);
                __m256i s = _mm256_shuffle_epi8(luts, p);
                __m256i r = _mm256_loadu_si256((const __m256i *)(qr + i));
                __m256i m = _mm256_loadu_si256((const __m256i *)(qi + i));
                __m256i pc = _mm256_maddubs_epi16(_mm256_abs_epi8(c), _mm256_sign_epi8(r, c));
                __m256i ps = _mm256_maddubs_epi16(_mm256_abs_epi8(s), _mm256_sign_epi8(m, s));
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(pc, ones));
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(ps, ones));
            }
            out[qn * n + k] = hsum_epi32(acc);
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// SuperSeed v1.1 packed lattice (2026-09-30, TORUS-TEL step 7): 4 bits per phase, lossless. Layout "block-64 split":
// for every block of 64 dimensions, byte j (0..31) holds phase[64b + j] in its LOW nibble and phase[64b + 32 + j] in its
// HIGH nibble. A 32-byte load then yields two ready 32-lane phase vectors (and 0x0F / shift 4), so the pshufb lookup runs
// unchanged and the query stays in natural order. Row stride = d/2 bytes; d must be a multiple of 64. Scores are
// bit-identical to the unpacked kernels (same products, same sums).
static inline __m256i p4_block(__m256i b, __m256i lo_mask, const int8_t *qr, const int8_t *qi, __m256i lutc, __m256i luts,
                               __m256i ones, __m256i acc) {
    __m256i ph[2] = {_mm256_and_si256(b, lo_mask), _mm256_and_si256(_mm256_srli_epi16(b, 4), lo_mask)};
    for (int h = 0; h < 2; h++) {
        __m256i c = _mm256_shuffle_epi8(lutc, ph[h]);
        __m256i s = _mm256_shuffle_epi8(luts, ph[h]);
        __m256i r = _mm256_loadu_si256((const __m256i *)(qr + 32 * h));
        __m256i m = _mm256_loadu_si256((const __m256i *)(qi + 32 * h));
        __m256i pc = _mm256_maddubs_epi16(_mm256_abs_epi8(c), _mm256_sign_epi8(r, c));
        __m256i ps = _mm256_maddubs_epi16(_mm256_abs_epi8(s), _mm256_sign_epi8(m, s));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(pc, ones));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(ps, ones));
    }
    return acc;
}

// Score rows of a packed lattice using only the first d_used dimensions (d_used % 64 == 0, <= d). rows == NULL scores rows
// 0..n-1; otherwise rows[0..n-1] are the row indices to score (the cascade's rescoring pass). out[j] for the j-th scored row.
void omniring_sift_p4(const uint8_t *packed, size_t stride, size_t d_used, const int64_t *rows, size_t n,
                      const uint8_t *query, int32_t *out) {
    const __m256i lutc = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)C8));
    const __m256i luts = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)S8));
    const __m256i ones = _mm256_set1_epi16(1), lo = _mm256_set1_epi8(0x0F);
    int8_t qr[8192], qi[8192];
    for (size_t i = 0; i < d_used; i++) { qr[i] = C8[query[i] & 15]; qi[i] = S8[query[i] & 15]; }
#pragma omp parallel for schedule(static) if (n >= 256)
    for (long k = 0; k < (long)n; k++) {
        const uint8_t *row = packed + (size_t)(rows ? rows[k] : k) * stride;
        __m256i acc = _mm256_setzero_si256();
        for (size_t blk = 0; blk < d_used; blk += 64)
            acc = p4_block(_mm256_loadu_si256((const __m256i *)(row + blk / 2)), lo, qr + blk, qi + blk, lutc, luts, ones, acc);
        out[k] = hsum_epi32(acc);
    }
}

// Batched packed sift: nq queries x n rows, first d_used dims, one parallel region (as omniring_sift_batch).
void omniring_sift_batch_p4(const uint8_t *packed, size_t n, size_t stride, size_t d_used, const uint8_t *queries,
                            size_t qstride, size_t nq, int32_t *out) {
    const __m256i lutc = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)C8));
    const __m256i luts = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)S8));
    const __m256i ones = _mm256_set1_epi16(1), lo = _mm256_set1_epi8(0x0F);
#pragma omp parallel for schedule(dynamic, 1)
    for (long qn = 0; qn < (long)nq; qn++) {
        const uint8_t *qp = queries + (size_t)qn * qstride;
        int8_t qr[8192], qi[8192];
        for (size_t i = 0; i < d_used; i++) { qr[i] = C8[qp[i] & 15]; qi[i] = S8[qp[i] & 15]; }
        for (size_t k = 0; k < n; k++) {
            const uint8_t *row = packed + k * stride;
            __m256i acc = _mm256_setzero_si256();
            for (size_t blk = 0; blk < d_used; blk += 64)
                acc = p4_block(_mm256_loadu_si256((const __m256i *)(row + blk / 2)), lo, qr + blk, qi + blk, lutc, luts, ones, acc);
            out[qn * n + k] = hsum_epi32(acc);
        }
    }
}
