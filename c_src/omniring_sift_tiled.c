// Geometric sift (2026-09-30, J: "apply it on this laptop"): the work of a batch sift is a 3-D block (queries x marks x
// dims). The plain batch kernel walks it as thin lines (each query re-reads every mark from memory). Here it is cut into
// CUBES — TQ queries x TM marks (all dims) — sized to the cache, walked in HILBERT order (a space-filling fractal path, so
// consecutive tiles share data at every cache level), and the innermost cube is 4 queries x 1 mark: each 32-phase chunk of
// a mark is loaded and table-looked-up ONCE and reused by 4 queries. Same products, same sums: scores are bit-identical.
#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

static const int8_t C8[16] = {127, 117, 90, 49, 0, -49, -90, -117, -127, -117, -90, -49, 0, 49, 90, 117};
static const int8_t S8[16] = {0, 49, 90, 117, 127, 117, 90, 49, 0, -49, -90, -117, -127, -117, -90, -49};

static inline int32_t hsum(__m256i v) {
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E)); s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
    return _mm_cvtsi128_si32(s);
}

#define ACC(A, J) { __m256i r = _mm256_loadu_si256((const __m256i *)(r0 + (J) * d + i)), m = _mm256_loadu_si256((const __m256i *)(i0 + (J) * d + i)); \
                    A = _mm256_add_epi32(A, _mm256_madd_epi16(_mm256_maddubs_epi16(ac, _mm256_sign_epi8(r, c)), ones)); \
                    A = _mm256_add_epi32(A, _mm256_madd_epi16(_mm256_maddubs_epi16(as, _mm256_sign_epi8(m, s)), ones)); }

static void hilbert_d2xy(long side, long t, long *x, long *y) {   // classic Hilbert index -> (x, y)
    long rx, ry, s; *x = *y = 0;
    for (s = 1; s < side; s *= 2) {
        rx = 1 & (t / 2); ry = 1 & (t ^ rx);
        if (ry == 0) { if (rx == 1) { *x = s - 1 - *x; *y = s - 1 - *y; } long tt = *x; *x = *y; *y = tt; }
        *x += s * rx; *y += s * ry; t /= 4;
    }
}

// order: 0 = plain lines (reference batch kernel order per query), 1 = cubes row-major, 2 = cubes in Hilbert order
void omniring_sift_batch_tiled(const uint8_t *lattice, size_t n, size_t d, const uint8_t *queries, size_t nq, int32_t *out,
                               int tq, int tm, int order) {
    const __m256i lutc = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)C8));
    const __m256i luts = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)S8));
    const __m256i ones = _mm256_set1_epi16(1);
    int8_t *QR = aligned_alloc(64, nq * d), *QI = aligned_alloc(64, nq * d);
    for (size_t j = 0; j < nq * d; j++) { QR[j] = C8[queries[j] & 15]; QI[j] = S8[queries[j] & 15]; }
    const int qb8_flag = order >= 10; if (order >= 10) order -= 10;
    long nqb = (nq + tq - 1) / tq, nmb = (n + tm - 1) / tm, side = 1;
    while (side < nqb || side < nmb) side *= 2;
    long ntiles = 0, *tl = malloc(sizeof(long) * 2 * (order == 2 ? side * side : nqb * nmb));
    if (order == 2) { for (long t = 0; t < side * side; t++) { long x, y; hilbert_d2xy(side, t, &x, &y);
                                                              if (x < nqb && y < nmb) { tl[2 * ntiles] = x; tl[2 * ntiles + 1] = y; ntiles++; } } }
    else { for (long y = 0; y < nmb; y++) for (long x = 0; x < nqb; x++) { tl[2 * ntiles] = x; tl[2 * ntiles + 1] = y; ntiles++; } }
    const int getenv_qb8 = qb8_flag;
#pragma omp parallel for schedule(dynamic, 1)
    for (long t = 0; t < ntiles; t++) {
        size_t q0 = tl[2 * t] * tq, q1 = q0 + tq < nq ? q0 + tq : nq, m0 = tl[2 * t + 1] * tm, m1 = m0 + tm < n ? m0 + tm : n;
        for (size_t k = m0; k < m1; k++) {
            const uint8_t *row = lattice + k * d;
            size_t q = q0;
            for (; q + 8 <= q1 && getenv_qb8; q += 8) {           // deeper cube: 8 queries share each chunk
                __m256i a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0;
                const int8_t *r0 = QR + q * d, *i0 = QI + q * d;
                for (size_t i = 0; i < d; i += 32) {
                    __m256i p = _mm256_loadu_si256((const __m256i *)(row + i));
                    __m256i c = _mm256_shuffle_epi8(lutc, p), s = _mm256_shuffle_epi8(luts, p);
                    __m256i ac = _mm256_abs_epi8(c), as = _mm256_abs_epi8(s);
                    ACC(a0, 0) ACC(a1, 1) ACC(a2, 2) ACC(a3, 3) ACC(a4, 4) ACC(a5, 5) ACC(a6, 6) ACC(a7, 7)
                }
                out[(q + 0) * n + k] = hsum(a0); out[(q + 1) * n + k] = hsum(a1); out[(q + 2) * n + k] = hsum(a2);
                out[(q + 3) * n + k] = hsum(a3); out[(q + 4) * n + k] = hsum(a4); out[(q + 5) * n + k] = hsum(a5);
                out[(q + 6) * n + k] = hsum(a6); out[(q + 7) * n + k] = hsum(a7);
            }
            for (; q + 4 <= q1; q += 4) {                       // inner cube: 4 queries share each looked-up chunk
                __m256i a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0, a3 = a0;
                const int8_t *r0 = QR + q * d, *i0 = QI + q * d;
                for (size_t i = 0; i < d; i += 32) {
                    __m256i p = _mm256_loadu_si256((const __m256i *)(row + i));
                    __m256i c = _mm256_shuffle_epi8(lutc, p), s = _mm256_shuffle_epi8(luts, p);
                    __m256i ac = _mm256_abs_epi8(c), as = _mm256_abs_epi8(s);
ACC(a0, 0) ACC(a1, 1) ACC(a2, 2) ACC(a3, 3)
                }
                out[(q + 0) * n + k] = hsum(a0); out[(q + 1) * n + k] = hsum(a1);
                out[(q + 2) * n + k] = hsum(a2); out[(q + 3) * n + k] = hsum(a3);
            }
            for (; q < q1; q++) {                              // tail queries, one at a time
                __m256i a = _mm256_setzero_si256(); const int8_t *r0 = QR + q * d, *i0 = QI + q * d;
                for (size_t i = 0; i < d; i += 32) {
                    __m256i p = _mm256_loadu_si256((const __m256i *)(row + i));
                    __m256i c = _mm256_shuffle_epi8(lutc, p), s = _mm256_shuffle_epi8(luts, p);
                    __m256i ac = _mm256_abs_epi8(c), as = _mm256_abs_epi8(s);
                    ACC(a, 0)
                }
                out[q * n + k] = hsum(a);
            }
        }
    }
    free(tl); free(QR); free(QI);
}
