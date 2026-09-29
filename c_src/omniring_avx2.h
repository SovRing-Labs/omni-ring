#ifndef OMNIRING_AVX2_H
#define OMNIRING_AVX2_H

#include <stdint.h>
#include <stddef.h>

#define OMNIRING_D 5120
#define OMNIRING_PACKED_BYTES 2560 // 5120 nibbles (4-bit) packed 2 per byte

#ifdef __cplusplus
extern "C" {
#endif

// Similarity score between two packed vectors (5120 dimensions, 2560 bytes)
// Returns scalar integer sum of cosine values in [-127 * 5120, +127 * 5120]
int32_t omniring_similarity_avx2(const uint8_t* a, const uint8_t* b);

// Batch similarity scores between one query vector and a matrix of candidate vectors
void omniring_batch_similarity_avx2(
    const uint8_t* query,
    const uint8_t* matrix,
    size_t n_vectors,
    int32_t* out_scores
);

// In-place or out-of-place vector binding: out[i] = (a[i] + b[i]) mod 16
void omniring_bind_avx2(const uint8_t* a, const uint8_t* b, uint8_t* out);

// Vector unbinding: out[i] = (a[i] - b[i]) mod 16
void omniring_unbind_avx2(const uint8_t* a, const uint8_t* b, uint8_t* out);

// Pack 5120 uint8 phase indices (0..15) into 2560 bytes
void omniring_pack_nibbles(const uint8_t* in, uint8_t* out, size_t n_dims);

// Unpack 2560 bytes into 5120 uint8 phase indices
void omniring_unpack_nibbles(const uint8_t* in, uint8_t* out, size_t n_dims);

#ifdef __cplusplus
}
#endif

#endif // OMNIRING_AVX2_H
