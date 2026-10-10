#pragma once

constexpr int kBr = 64;
constexpr int kBc = 64;
constexpr int kHeadDim = 128;
constexpr size_t kSharedMemorySize = (kBr + 2 * kBc) * kHeadDim * sizeof(half);

template <typename value_t, int M, int N>
__device__ static __forceinline__ void
FillZeros2D(value_t (&p)[M][N]) {
    #pragma unroll
    for (int i = 0; i < M; ++i) {
        #pragma unroll
        for (int j = 0; j < N; ++j) {
            p[i][j] = 0;
        }
    }
}