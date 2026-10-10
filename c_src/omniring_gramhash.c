#include "omniring_gramhash.h"

void omniring_gram_phases(const uint32_t* h, size_t n, uint32_t d, uint8_t* out) {
    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        uint32_t hi = h[i];
        for (uint32_t j = 0; j < d; j++) {
            uint32_t kd = j * 2654435761u + 0x9E3779B9u;
            uint32_t x = (hi ^ kd) * 0x85EBCA6Bu;
            x ^= x >> 13u;
            x *= 0xC2B2AE35u;
            x ^= x >> 16u;
            out[i * (size_t)d + j] = (uint8_t)((x >> 28u) & 15u);
        }
    }
}
