#!/usr/bin/env sh
# Build the AVX2 kernel next to the Python package (x86-64 with AVX2; gcc or clang with OpenMP).
set -e
cd "$(dirname "$0")/c_src"
${CC:-gcc} -O3 -mavx2 -fopenmp -fPIC -shared -o ../omni_ring/libomniring_avx2.so omniring_avx2.c omniring_polymorphic.c omniring_sift.c
echo "built omni_ring/libomniring_avx2.so"
