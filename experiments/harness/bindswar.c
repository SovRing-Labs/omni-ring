// BIND-SWAR-0: nibble-packed phase binding (2 phases per byte), add/sub mod 16 per nibble, no cross-nibble carry.
#include <stddef.h>
#include <stdint.h>
void bind_p(const uint8_t *a, const uint8_t *b, uint8_t *o, size_t n) {
    for (size_t i = 0; i < n; i++) {
        uint8_t lo = (uint8_t)((a[i] + b[i]) & 0x0F), hi = (uint8_t)(((a[i] >> 4) + (b[i] >> 4)) & 0x0F);
        o[i] = (uint8_t)(lo | (hi << 4));
    }
}
void unbind_p(const uint8_t *a, const uint8_t *b, uint8_t *o, size_t n) {
    for (size_t i = 0; i < n; i++) {
        uint8_t lo = (uint8_t)((a[i] - b[i]) & 0x0F), hi = (uint8_t)(((a[i] >> 4) - (b[i] >> 4)) & 0x0F);
        o[i] = (uint8_t)(lo | (hi << 4));
    }
}
