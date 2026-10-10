#ifndef OMNIRING_GRAMHASH_H
#define OMNIRING_GRAMHASH_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// store.py's _gram_phases hot loop (OS-RING-0 step 2): n gram hashes x d dims -> n*d 4-bit phase indices.
// out[i*d + j] = ((((h[i] ^ kd[j]) * 0x85EBCA6B) ^ >>13) * 0xC2B2AE35 ^ >>16) >> 28 & 15, kd[j] = j*2654435761 + 0x9E3779B9.
// Pure uint32 wraparound arithmetic throughout — bit-identical to the numpy reference (store.py::_gram_phases).
void omniring_gram_phases(const uint32_t* h, size_t n, uint32_t d, uint8_t* out);

#ifdef __cplusplus
}
#endif

#endif // OMNIRING_GRAMHASH_H
