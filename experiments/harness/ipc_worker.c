// IPC-0 shared-memory worker: single-producer / single-consumer request slot with sequence counters.
// Layout of /dev/shm/ipc0 (4096 B): [0] u64 req_seq, [8] u64 resp_seq, [64..] 1024-byte request, [2048..] 1024-byte reply.
// The worker spins on req_seq; when it advances, it "processes" the request (copies it, reversing bytes as a stand-in
// for real work) and publishes resp_seq = req_seq. Exits when the request starts with "QUIT".
#include <fcntl.h>
#include <stdatomic.h>
#include <stdint.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <immintrin.h>
int main(void) {
    int fd = shm_open("/ipc0", O_RDWR, 0600);
    if (fd < 0) return 1;
    uint8_t *m = mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    _Atomic uint64_t *req = (_Atomic uint64_t *)m, *resp = (_Atomic uint64_t *)(m + 8);
    uint64_t seen = atomic_load(req);
    for (;;) {
        uint64_t s;
        while ((s = atomic_load_explicit(req, memory_order_acquire)) == seen) _mm_pause();
        seen = s;
        if (memcmp(m + 64, "QUIT", 4) == 0) break;
        for (int i = 0; i < 1024; i++) m[2048 + i] = m[64 + 1023 - i];
        atomic_store_explicit(resp, s, memory_order_release);
    }
    return 0;
}
