#pragma once

#include "def.cuh"

constexpr index_t kBr = 64;
constexpr index_t kBc = 64;
constexpr index_t kHeadDim = 128;
constexpr index_t kSharedMemorySize = (kBr + 2 * kBc) * kHeadDim * sizeof(half);

template <typename value_t, int M, int N>
DEVICE_FUNC void
FillZeros2D(value_t (&p)[M][N]) {
    #pragma unroll
    for (int i = 0; i < M; ++i) {
        #pragma unroll
        for (int j = 0; j < N; ++j) {
            p[i][j] = 0;
        }
    }
}