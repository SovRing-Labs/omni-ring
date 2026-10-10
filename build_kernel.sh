#!/usr/bin/env sh
# Build the AVX2 kernel next to the Python package (x86-64 with AVX2; gcc or clang with OpenMP).
set -e
cd "$(dirname "$0")/c_src"
${CC:-gcc} -O3 -mavx2 -fopenmp -fPIC -shared -o ../omni_ring/libomniring_avx2.so.new omniring_avx2.c omniring_polymorphic.c omniring_sift.c omniring_sift_tiled.c omniring_gramhash.c
# rename, never overwrite in place: running servers keep the old mapping (no torn library under a live process)
mv -f ../omni_ring/libomniring_avx2.so.new ../omni_ring/libomniring_avx2.so
echo "built omni_ring/libomniring_avx2.so"
