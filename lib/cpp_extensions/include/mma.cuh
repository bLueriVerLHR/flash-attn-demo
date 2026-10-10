#pragma once

#include "arch.cuh"

template <typename value_t, int kMFragments, int kNFragments, int kKFragments, typename accum_t>
DEVICE_CONSTEXPR void
WrapFragmentMma(
    uint32_t (&regs_A)[kMFragments][kKFragments],
    uint32_t (&regs_B)[kKFragments][kNFragments],
    accum_t  (&regs_C)[kMFragments][kNFragments]
) {
    #pragma unroll
    for (int k = 0; k < kKFragments; k += kMmaK / kLdMatrixSize) {
        #pragma unroll
        for (int m = 0; m < kMFragments; m += kMmaM / kLdMatrixSize) {
            #pragma unroll
            for (int n = 0; n < kNFragments; n += kMmaN / kLdMatrixSize) {
                MmaM16N8K16F32Accm(
                    regs_C[m][n * 2], regs_C[m][n * 2 + 1], regs_C[m + 1][n * 2], regs_C[m + 1][n * 2 + 1],
                    regs_A[m][k], regs_A[m + 1][k], regs_A[m][k + 1], regs_A[m + 1][k + 1],
                    regs_B[n][k], regs_B[n][k + 1],
                    regs_C[m][n * 2], regs_C[m][n * 2 + 1], regs_C[m + 1][n * 2], regs_C[m + 1][n * 2 + 1]
                );
            }
        }
    }
}