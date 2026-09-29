#include "omniring_avx2.h"
#include <immintrin.h>

// 16-byte cosine table for angles k * 2*pi / 16, scaled to int8 [-127, 127]
static const int8_t COS_TABLE[16] __attribute__((aligned(16))) = {
    127,  117,   90,   49,
      0,  -49,  -90, -117,
   -127, -117,  -90,  -49,
      0,   49,   90,  117
};

int32_t omniring_similarity_avx2(const uint8_t* a, const uint8_t* b) {
    __m128i t128 = _mm_load_si128((const __m128i*)COS_TABLE);
    __m256i lut = _mm256_broadcastsi128_si256(t128);
    __m256i mask_0f = _mm256_set1_epi8(0x0F);
    __m256i ones_8  = _mm256_set1_epi8(1);
    __m256i ones_16 = _mm256_set1_epi16(1);
    __m256i acc0 = _mm256_setzero_si256();
    __m256i acc1 = _mm256_setzero_si256();

    for (size_t i = 0; i < OMNIRING_PACKED_BYTES; i += 64) {
        // First 32 bytes
        __m256i va0 = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i vb0 = _mm256_loadu_si256((const __m256i*)(b + i));
        __m256i diff_low0 = _mm256_and_si256(_mm256_sub_epi8(va0, vb0), mask_0f);
        __m256i cos_low0 = _mm256_shuffle_epi8(lut, diff_low0);
        __m256i va_high0 = _mm256_and_si256(_mm256_srli_epi16(va0, 4), mask_0f);
        __m256i vb_high0 = _mm256_and_si256(_mm256_srli_epi16(vb0, 4), mask_0f);
        __m256i diff_high0 = _mm256_and_si256(_mm256_sub_epi8(va_high0, vb_high0), mask_0f);
        __m256i cos_high0 = _mm256_shuffle_epi8(lut, diff_high0);
        __m256i sum16_0 = _mm256_add_epi16(_mm256_maddubs_epi16(ones_8, cos_low0),
                                           _mm256_maddubs_epi16(ones_8, cos_high0));
        acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(sum16_0, ones_16));

        // Second 32 bytes
        __m256i va1 = _mm256_loadu_si256((const __m256i*)(a + i + 32));
        __m256i vb1 = _mm256_loadu_si256((const __m256i*)(b + i + 32));
        __m256i diff_low1 = _mm256_and_si256(_mm256_sub_epi8(va1, vb1), mask_0f);
        __m256i cos_low1 = _mm256_shuffle_epi8(lut, diff_low1);
        __m256i va_high1 = _mm256_and_si256(_mm256_srli_epi16(va1, 4), mask_0f);
        __m256i vb_high1 = _mm256_and_si256(_mm256_srli_epi16(vb1, 4), mask_0f);
        __m256i diff_high1 = _mm256_and_si256(_mm256_sub_epi8(va_high1, vb_high1), mask_0f);
        __m256i cos_high1 = _mm256_shuffle_epi8(lut, diff_high1);
        __m256i sum16_1 = _mm256_add_epi16(_mm256_maddubs_epi16(ones_8, cos_low1),
                                           _mm256_maddubs_epi16(ones_8, cos_high1));
        acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(sum16_1, ones_16));
    }

    __m256i acc = _mm256_add_epi32(acc0, acc1);
    int32_t buf[8];
    _mm256_storeu_si256((__m256i*)buf, acc);
    return buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7];
}

void omniring_batch_similarity_avx2(
    const uint8_t* query,
    const uint8_t* matrix,
    size_t n_vectors,
    int32_t* out_scores
) {
    __m128i t128 = _mm_load_si128((const __m128i*)COS_TABLE);
    __m256i lut = _mm256_broadcastsi128_si256(t128);
    __m256i mask_0f = _mm256_set1_epi8(0x0F);
    __m256i ones_8  = _mm256_set1_epi8(1);
    __m256i ones_16 = _mm256_set1_epi16(1);

    for (size_t n = 0; n < n_vectors; n++) {
        const uint8_t* b = matrix + n * OMNIRING_PACKED_BYTES;
        __m256i acc0 = _mm256_setzero_si256();
        __m256i acc1 = _mm256_setzero_si256();

        for (size_t i = 0; i < OMNIRING_PACKED_BYTES; i += 64) {
            __m256i va0 = _mm256_loadu_si256((const __m256i*)(query + i));
            __m256i vb0 = _mm256_loadu_si256((const __m256i*)(b + i));
            __m256i diff_low0 = _mm256_and_si256(_mm256_sub_epi8(va0, vb0), mask_0f);
            __m256i cos_low0 = _mm256_shuffle_epi8(lut, diff_low0);
            __m256i va_high0 = _mm256_and_si256(_mm256_srli_epi16(va0, 4), mask_0f);
            __m256i vb_high0 = _mm256_and_si256(_mm256_srli_epi16(vb0, 4), mask_0f);
            __m256i diff_high0 = _mm256_and_si256(_mm256_sub_epi8(va_high0, vb_high0), mask_0f);
            __m256i cos_high0 = _mm256_shuffle_epi8(lut, diff_high0);
            __m256i sum16_0 = _mm256_add_epi16(_mm256_maddubs_epi16(ones_8, cos_low0),
                                               _mm256_maddubs_epi16(ones_8, cos_high0));
            acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(sum16_0, ones_16));

            __m256i va1 = _mm256_loadu_si256((const __m256i*)(query + i + 32));
            __m256i vb1 = _mm256_loadu_si256((const __m256i*)(b + i + 32));
            __m256i diff_low1 = _mm256_and_si256(_mm256_sub_epi8(va1, vb1), mask_0f);
            __m256i cos_low1 = _mm256_shuffle_epi8(lut, diff_low1);
            __m256i va_high1 = _mm256_and_si256(_mm256_srli_epi16(va1, 4), mask_0f);
            __m256i vb_high1 = _mm256_and_si256(_mm256_srli_epi16(vb1, 4), mask_0f);
            __m256i diff_high1 = _mm256_and_si256(_mm256_sub_epi8(va_high1, vb_high1), mask_0f);
            __m256i cos_high1 = _mm256_shuffle_epi8(lut, diff_high1);
            __m256i sum16_1 = _mm256_add_epi16(_mm256_maddubs_epi16(ones_8, cos_low1),
                                               _mm256_maddubs_epi16(ones_8, cos_high1));
            acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(sum16_1, ones_16));
        }

        __m256i acc = _mm256_add_epi32(acc0, acc1);
        int32_t buf[8];
        _mm256_storeu_si256((__m256i*)buf, acc);
        out_scores[n] = buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7];
    }
}

void omniring_bind_avx2(const uint8_t* a, const uint8_t* b, uint8_t* out) {
    __m256i mask_0f = _mm256_set1_epi8(0x0F);
    for (size_t i = 0; i < OMNIRING_PACKED_BYTES; i += 32) {
        __m256i va = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i*)(b + i));

        __m256i la = _mm256_and_si256(va, mask_0f);
        __m256i lb = _mm256_and_si256(vb, mask_0f);
        __m256i lres = _mm256_and_si256(_mm256_add_epi8(la, lb), mask_0f);

        __m256i ha = _mm256_and_si256(_mm256_srli_epi16(va, 4), mask_0f);
        __m256i hb = _mm256_and_si256(_mm256_srli_epi16(vb, 4), mask_0f);
        __m256i hres = _mm256_slli_epi16(_mm256_and_si256(_mm256_add_epi8(ha, hb), mask_0f), 4);

        __m256i res = _mm256_or_si256(lres, hres);
        _mm256_storeu_si256((__m256i*)(out + i), res);
    }
}

void omniring_unbind_avx2(const uint8_t* a, const uint8_t* b, uint8_t* out) {
    __m256i mask_0f = _mm256_set1_epi8(0x0F);
    for (size_t i = 0; i < OMNIRING_PACKED_BYTES; i += 32) {
        __m256i va = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i*)(b + i));

        __m256i la = _mm256_and_si256(va, mask_0f);
        __m256i lb = _mm256_and_si256(vb, mask_0f);
        __m256i lres = _mm256_and_si256(_mm256_sub_epi8(la, lb), mask_0f);

        __m256i ha = _mm256_and_si256(_mm256_srli_epi16(va, 4), mask_0f);
        __m256i hb = _mm256_and_si256(_mm256_srli_epi16(vb, 4), mask_0f);
        __m256i hres = _mm256_slli_epi16(_mm256_and_si256(_mm256_sub_epi8(ha, hb), mask_0f), 4);

        __m256i res = _mm256_or_si256(lres, hres);
        _mm256_storeu_si256((__m256i*)(out + i), res);
    }
}

void omniring_pack_nibbles(const uint8_t* in, uint8_t* out, size_t n_dims) {
    size_t packed_len = n_dims / 2;
    for (size_t i = 0; i < packed_len; i++) {
        uint8_t low = in[2 * i] & 0x0F;
        uint8_t high = in[2 * i + 1] & 0x0F;
        out[i] = low | (high << 4);
    }
}

void omniring_unpack_nibbles(const uint8_t* in, uint8_t* out, size_t n_dims) {
    size_t packed_len = n_dims / 2;
    for (size_t i = 0; i < packed_len; i++) {
        out[2 * i] = in[i] & 0x0F;
        out[2 * i + 1] = (in[i] >> 4) & 0x0F;
    }
}
