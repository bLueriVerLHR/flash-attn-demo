#pragma once

#include <cinttypes>
#include "def.cuh"

constexpr index_t kMmaM = 16;
constexpr index_t kMmaN = 8;
constexpr index_t kMmaK = 16;

constexpr index_t kLdMatrixSize = 8;
constexpr index_t kCpAsyncLen = 16;
constexpr index_t kElemPerCpAsync = kCpAsyncLen / sizeof(half);

constexpr index_t kWarpSize = 32;
constexpr index_t kNumberOfWarps = 4;
constexpr index_t kNumberOfThreads = kWarpSize * kNumberOfWarps;

template <typename value_t>
DEVICE_FUNC uint32_t
SharedMemoryU32Ptr(value_t* p) {
    /**
     * converts a generic 64-bit memory address
     * to a 32-bit SMEM-specific address, which is more performant.
     */
    return (uint32_t)__cvta_generic_to_shared(p);
}


template <int Len, typename value_t>
DEVICE_FUNC void
CpAsync(value_t* sdst, const value_t* gsrc) {
    uint32_t p = SharedMemoryU32Ptr(sdst);

    /**
     * The .cg (cache-global) option bypasses the L1 cache, reducing cache
     * pollution and providing a more direct path from L2 to shared memory.
     */
    asm volatile (
        "cp.async.cg.shared.global [%0], [%1], %2;"
        :: "r"(p), "l"(gsrc), "n"(Len)
    );
}

DEVICE_FUNC void
CpAsyncCommit() {
    asm volatile ("cp.async.commit_group;");
}

template <int ngroups>
DEVICE_FUNC void
CpAsyncWait() {
    asm volatile (
        "cp.async.wait_group %0;"
        :: "n"(ngroups)
    );
}

template <bool IsTranspose = false, typename value_t = void>
DEVICE_FUNC void
LdMatrixX4(
	value_t* ssrc,
	uint32_t &a1,
	uint32_t &a2,
    uint32_t &a3,
    uint32_t &a4
) {
    uint32_t p = SharedMemoryU32Ptr(ssrc);
    if constexpr (IsTranspose) {
        asm volatile (
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16.trans"
            "{%0,%1,%2,%3}, [%4];"
            : "=r"(a1), "=r"(a2), "=r"(a3), "=r"(a4)
            : "r"(p)
        );
    } else {
        asm volatile (
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
            "{%0,%1,%2,%3}, [%4];"
            : "=r"(a1), "=r"(a2), "=r"(a3), "=r"(a4)
            : "r"(p)
        );
    }
}

template <bool IsTranspose = false, typename value_t = void>
DEVICE_FUNC void
LdMatrixX2(
	value_t* ssrc,
	uint32_t &a1,
	uint32_t &a2
) {
    uint32_t p = SharedMemoryU32Ptr(ssrc);
    if constexpr (IsTranspose) {
        asm volatile (
            "ldmatrix.sync.aligned.m8n8.x2.shared.b16.trans"
            "{%0,%1}, [%2];"
            : "=r"(a1), "=r"(a2)
            : "r"(p)
        );
    } else {
        asm volatile (
            "ldmatrix.sync.aligned.m8n8.x2.shared.b16"
            "{%0,%1}, [%2];"
            : "=r"(a1), "=r"(a2)
            : "r"(p)
        );
    }
}

template <typename value_t>
DEVICE_FUNC void
MmaM16N8K16F32Accm(
	float &d1, float &d2, float &d3, float &d4,
 
    uint32_t const &a1, uint32_t const &a2,
    uint32_t const &a3, uint32_t const &a4,
 
    uint32_t const &b1, uint32_t const &b2,
 
	float const &c1, float const &c2,
	float const &c3, float const &c4
) {
    if constexpr (std::is_same_v<value_t, nv_bfloat16>) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                     " { %0, %1, %2, %3 }, "
                     " { %4, %5, %6, %7 }, "
                     " { %8, %9 }, "
                     " { %10, %11, %12, %13 }; "
                     : "=f"(d1), "=f"(d2), "=f"(d3), "=f"(d4)
                     : "r"(a1), "r"(a2), "r"(a3), "r"(a4), "r"(b1), "r"(b2),
                       "f"(c1), "f"(c2), "f"(c3), "f"(c4));
    } else {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                     " { %0, %1, %2, %3 }, "
                     " { %4, %5, %6, %7 }, "
                     " { %8, %9 }, "
                     " { %10, %11, %12, %13 }; "
                     : "=f"(d1), "=f"(d2), "=f"(d3), "=f"(d4)
                     : "r"(a1), "r"(a2), "r"(a3), "r"(a4), "r"(b1), "r"(b2),
                       "f"(c1), "f"(c2), "f"(c3), "f"(c4));
    }
}

DEVICE_FUNC uint32_t
PackHalf2(float f1, float f2) {
    return __half_as_ushort(__float2half_rn(f1)) | (__half_as_ushort(__float2half_rn(f2)) << 16);
}

DEVICE_FUNC float
WarpMax4(float v) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
    return v;
}

DEVICE_FUNC float
WarpSum4(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    v += __shfl_xor_sync(0xffffffffu, v, 2);
    return v;
}

