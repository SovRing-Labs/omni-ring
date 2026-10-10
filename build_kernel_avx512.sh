#!/usr/bin/env sh
# Build the AVX-512 kernels (OMNIKIT 2026-10-08: VNNI sift + VPOPCNTDQ Hamming/ternary sift) next to the Python package.
# Compiles on any x86-64 gcc/clang; RUNS only on CPUs with avx512f/bw/vl/vpopcntdq/vnni (sift.has_avx512 checks /proc/cpuinfo).
set -e
cd "$(dirname "$0")/c_src"
${CC:-gcc} -O3 -mavx512f -mavx512bw -mavx512vl -mavx512vpopcntdq -mavx512vnni -fopenmp -fPIC -shared \
  -o ../omni_ring/libomniring_avx512.so.new omniring_avx512.c
# rename, never overwrite in place: running servers keep the old mapping (no torn library under a live process)
mv -f ../omni_ring/libomniring_avx512.so.new ../omni_ring/libomniring_avx512.so
echo "built omni_ring/libomniring_avx512.so"
