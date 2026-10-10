// AVX-512 HDC kernels (OMNIKIT, 2026-10-08; design note of 2026-10-08, approved by the author).
// Built ONLY by build_kernel_avx512.sh into libomniring_avx512.so (-mavx512f -mavx512bw -mavx512vl -mavx512vpopcntdq -mavx512vnni).
// Python loads it only after /proc/cpuinfo shows every flag (sift.has_avx512) — never call these on an AVX2-only CPU.
//
// 1. omniring_sift_cplx_avx512 — the exhaustive lattice sift of omniring_sift.c (same inputs, same int32 scores, BIT-IDENTICAL):
//    S[n] = sum_d qr[d]*C8[p] + qi[d]*S8[p]. VNNI vpdpbusd multiplies u8 x s8, so the lattice side is looked up in OFFSET tables
//    CU = C8 + 127, SU = S8 + 127 (0..254, unsigned) and the offset is removed exactly once per query:
//    sum (C8[p] + 127) * q = S + 127 * sum q.  All in int32 without saturation (vpdpbusd, not vpdpbusds): |S| <= 2*8192*254*127.
//    (The AVX2 kernel's maddubs never saturates either: pairs of 127*127 = 32,258 < 32,767 — so both are exact, so equal.)
// 2. omniring_hamming_sift_avx512 — binary hypervectors (1 bit/dim, words of 64 bits): out[n] = popcount(row_n XOR q) (VPOPCNTDQ).
// 3. omniring_ternary_sift_avx512 — ternary hypervectors in 2 bit-planes (sign, active; the omniring_polymorphic.c layout):
//    out[n] = popcnt(both) - 2*popcnt(both & (sign_n XOR sign_q)), both = active_n & active_q — equal to omniring_ternary_dot.
// Lattice rows are row-major, read in order; any length (tails are masked loads).
#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>

static const int8_t C8[16] = {127, 117, 90, 49, 0, -49, -90, -117, -127, -117, -90, -49, 0, 49, 90, 117};
static const int8_t S8[16] = {0, 49, 90, 117, 127, 117, 90, 49, 0, -49, -90, -117, -127, -117, -90, -49};

int omniring_avx512_version(void) { return 1; }

static inline __m512i lut16(const int8_t *t, int off) {
    uint8_t u[16];
    for (int i = 0; i < 16; i++) u[i] = (uint8_t)(t[i] + off);
    return _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)u));
}

static inline __mmask64 tail64(size_t left) { return left >= 64 ? ~(__mmask64)0 : (((__mmask64)1 << left) - 1); }
static inline __mmask8 tail8(size_t left) { return left >= 8 ? (__mmask8)0xFF : (__mmask8)((1u << left) - 1); }

void omniring_sift_cplx_avx512(const uint8_t *lattice, size_t n, size_t d, const int8_t *qr, const int8_t *qi, int32_t *out) {
    const __m512i lc = lut16(C8, 127), ls = lut16(S8, 127);
    int64_t sq = 0;
    for (size_t i = 0; i < d; i++) sq += (int64_t)qr[i] + (int64_t)qi[i];
    const int32_t corr = (int32_t)(127 * sq);
    // Rows are read strictly in order (one sequential stream the hardware prefetcher follows; a 4-row interleave measured 0.48x
    // the AVX2 kernel on a 64 MB lattice, 10-08 pod). Four independent accumulators (cos/sin x even/odd 64-blocks) hide vpdpbusd latency.
#pragma omp parallel for schedule(static)
    for (long k = 0; k < (long)n; k++) {
        const uint8_t *row = lattice + (size_t)k * d;
        __m512i a0 = _mm512_setzero_si512(), a1 = a0, a2 = a0, a3 = a0;
        size_t i = 0;
        for (; i + 128 <= d; i += 128) {
            __m512i p0 = _mm512_loadu_si512(row + i), p1 = _mm512_loadu_si512(row + i + 64);
            a0 = _mm512_dpbusd_epi32(a0, _mm512_shuffle_epi8(lc, p0), _mm512_loadu_si512(qr + i));
            a1 = _mm512_dpbusd_epi32(a1, _mm512_shuffle_epi8(ls, p0), _mm512_loadu_si512(qi + i));
            a2 = _mm512_dpbusd_epi32(a2, _mm512_shuffle_epi8(lc, p1), _mm512_loadu_si512(qr + i + 64));
            a3 = _mm512_dpbusd_epi32(a3, _mm512_shuffle_epi8(ls, p1), _mm512_loadu_si512(qi + i + 64));
        }
        for (; i < d; i += 64) {
            __mmask64 m = tail64(d - i);
            __m512i p = _mm512_maskz_loadu_epi8(m, row + i);
            a0 = _mm512_dpbusd_epi32(a0, _mm512_shuffle_epi8(lc, p), _mm512_maskz_loadu_epi8(m, qr + i));
            a1 = _mm512_dpbusd_epi32(a1, _mm512_shuffle_epi8(ls, p), _mm512_maskz_loadu_epi8(m, qi + i));
        }
        out[k] = _mm512_reduce_add_epi32(_mm512_add_epi32(_mm512_add_epi32(a0, a1), _mm512_add_epi32(a2, a3))) - corr;
    }
}

// Batched: nq queries (nq x d int8 real / imag) x n rows -> out nq x n. Same arithmetic as above, one parallel region.
void omniring_sift_batch_cplx_avx512(const uint8_t *lattice, size_t n, size_t d, const int8_t *qr, const int8_t *qi, size_t nq,
                                     int32_t *out) {
#pragma omp parallel for schedule(dynamic, 1)
    for (long q = 0; q < (long)nq; q++) {
        const __m512i lc = lut16(C8, 127), ls = lut16(S8, 127);
        const int8_t *vr0 = qr + (size_t)q * d, *vi0 = qi + (size_t)q * d;
        int64_t sq = 0;
        for (size_t i = 0; i < d; i++) sq += (int64_t)vr0[i] + (int64_t)vi0[i];
        const int32_t corr = (int32_t)(127 * sq);
        for (size_t k = 0; k < n; k++) {
            const uint8_t *row = lattice + k * d;
            __m512i a = _mm512_setzero_si512();
            for (size_t i = 0; i < d; i += 64) {
                __mmask64 m = tail64(d - i);
                __m512i p = _mm512_maskz_loadu_epi8(m, row + i);
                a = _mm512_dpbusd_epi32(a, _mm512_shuffle_epi8(lc, p), _mm512_maskz_loadu_epi8(m, vr0 + i));
                a = _mm512_dpbusd_epi32(a, _mm512_shuffle_epi8(ls, p), _mm512_maskz_loadu_epi8(m, vi0 + i));
            }
            out[(size_t)q * n + k] = _mm512_reduce_add_epi32(a) - corr;
        }
    }
}

void omniring_hamming_sift_avx512(const uint64_t *lat, size_t n, size_t words, const uint64_t *q, int32_t *out) {
#pragma omp parallel for schedule(static)
    for (long k = 0; k < (long)n; k++) {
        const uint64_t *row = lat + (size_t)k * words;
        __m512i acc = _mm512_setzero_si512();
        for (size_t w = 0; w < words; w += 8) {
            __mmask8 m = tail8(words - w);
            __m512i x = _mm512_xor_si512(_mm512_maskz_loadu_epi64(m, row + w), _mm512_maskz_loadu_epi64(m, q + w));
            acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(x));
        }
        out[k] = (int32_t)_mm512_reduce_add_epi64(acc);
    }
}

void omniring_ternary_sift_avx512(const uint64_t *lat_sign, const uint64_t *lat_active, size_t n, size_t words,
                                  const uint64_t *q_sign, const uint64_t *q_active, int32_t *out) {
#pragma omp parallel for schedule(static)
    for (long k = 0; k < (long)n; k++) {
        const uint64_t *rs = lat_sign + (size_t)k * words, *ra = lat_active + (size_t)k * words;
        __m512i both_n = _mm512_setzero_si512(), neg_n = both_n;
        for (size_t w = 0; w < words; w += 8) {
            __mmask8 m = tail8(words - w);
            __m512i both = _mm512_and_si512(_mm512_maskz_loadu_epi64(m, ra + w), _mm512_maskz_loadu_epi64(m, q_active + w));
            __m512i diff = _mm512_xor_si512(_mm512_maskz_loadu_epi64(m, rs + w), _mm512_maskz_loadu_epi64(m, q_sign + w));
            both_n = _mm512_add_epi64(both_n, _mm512_popcnt_epi64(both));
            neg_n = _mm512_add_epi64(neg_n, _mm512_popcnt_epi64(_mm512_and_si512(both, diff)));
        }
        out[k] = (int32_t)(_mm512_reduce_add_epi64(both_n) - 2 * _mm512_reduce_add_epi64(neg_n));
    }
}
