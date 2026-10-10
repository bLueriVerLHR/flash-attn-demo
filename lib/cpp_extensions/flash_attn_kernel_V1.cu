#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <torch/torch.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <pybind11/pybind11.h>

#include <inttypes.h>
#include <string>
#include <stdexcept>

namespace py = pybind11;

using index_t = int64_t;

#define Br 64
#define Bc 64
#define WARP_SIZE 32
#define NUM_WARPS 4
#define THREADS (NUM_WARPS * WARP_SIZE)   // 128

// Qwen3-8B
#define HEAD_DIM 128
#define N_HEADS_Q 32
#define N_HEADS_KV 8

// 0/1 ring double buffer
#define RING_STAGES 2

#define MEMORY : "memory"

#define SMEM_BYTES ((Br * HEAD_DIM + RING_STAGES * Bc * HEAD_DIM * 2) * sizeof(half))


struct FlashAttnKernelArgs {
    void* q;
    void* k;
    void* v;
    void* o;

    void* lse;

    index_t q_batch_stride;
    index_t q_seq_stride;
    index_t q_head_stride;
    index_t q_seq_len;
    index_t q_n_heads;

    index_t kv_batch_stride;
    index_t kv_seq_stride;
    index_t kv_head_stride;
    index_t kv_seq_len;
    index_t kv_n_heads;

    float softmax_scale;
    int  causal;
};

__device__ static __forceinline__ uint32_t
SharedMemoryU32Ptr(void* p) {
    // convert to 32bit pointer for better performance
    return (uint32_t)__cvta_generic_to_shared(p);
}

template <size_t len>
__device__ static __forceinline__ void
CpAsync(void* sdst, const void* gsrc) {
    uint32_t p = SharedMemoryU32Ptr(sdst);
    asm volatile (
        "cp.async.cg.shared.global [%0], [%1], %2;"
        :: "r"(p), "l"(gsrc), "n"(len)
        MEMORY
    );
}

__device__ static __forceinline__ void
CpAsyncCommit() {
    asm volatile ("cp.async.commit_group;" ::MEMORY);
}

template <size_t ngroups>
__device__ static __forceinline__ void
CpAsyncWait() {
    asm volatile (
        "cp.async.wait_group %0;"
        :: "n"(ngroups)
        MEMORY
    );
}

template <bool IsTranspose = false>
__device__ static __forceinline__ void
LdMatrixX4(
	void* ssrc,
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

template <bool IsTranspose = false>
__device__ static __forceinline__ void
LdMatrixX2(
	void* ssrc,
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
__device__ static __forceinline__ void
MmaM16N8K16F32Accm(
	float &d1, float &d2, float &d3, float &d4,
 
    uint32_t const &a1, uint32_t const &a2,
    uint32_t const &a3, uint32_t const &a4,
 
    uint32_t const &b1, uint32_t const &b2,
 
	float const &c1, float const &c2,
	float const &c3, float const &c4
) {
    if constexpr (std::is_same_v<std::remove_cv_t<value_t>, nv_bfloat16>) {
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

template <typename value_t, index_t M, index_t N>
__device__ static __forceinline__ void
FillZeros(value_t (&p)[M][N]) {
    #pragma unroll
    for (int i = 0; i < M; ++i) {
        #pragma unroll
        for (int j = 0; j < N; ++j) {
            p[i][j] = 0;
        }
    }
}

__device__ static __forceinline__ uint32_t
PackHalf2(float f1, float f2) {
    uint32_t h1 = __half_as_ushort(__float2half_rn(f1));
    uint32_t h2 = __half_as_ushort(__float2half_rn(f2));
    return h1 | (h2 << 16);
}

__device__ static __forceinline__ float
WarpMax4(float v) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
    return v;
}

__device__ static __forceinline__ float
WarpSum4(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    v += __shfl_xor_sync(0xffffffffu, v, 2);
    return v;
}

template <typename value_t>
__global__ static __launch_bounds__(THREADS) void
FlashAttnKernel(FlashAttnKernelArgs args) {
    extern __shared__ __align__(16) char smem[];
    value_t* sq = reinterpret_cast<value_t*>(smem);
    value_t* sk = sq + Br * HEAD_DIM;
    value_t* sv = sk + RING_STAGES * Bc * HEAD_DIM; // TODO: why ring stage here

    const value_t* const q = (value_t*)args.q;
    const value_t* const k = (value_t*)args.k;
    const value_t* const v = (value_t*)args.v;
    value_t* const o = (value_t*)args.o;

    const int tid = threadIdx.x; // thread id
    const int lid = tid & 31;    // lane id
    const int wid = tid >> 5;    // warp id

    const int q_blk     = blockIdx.x;
    const int batch_idx = blockIdx.y;
    const int head_idx  = blockIdx.z;

    const int heads_per_kv = int(args.q_n_heads / args.kv_n_heads);
    const int head_idx_kv = head_idx / heads_per_kv;

    const value_t* const gq = q + batch_idx * args.q_batch_stride + head_idx * args.q_head_stride;
    const value_t* const gk = k + batch_idx * args.kv_batch_stride + head_idx_kv * args.kv_head_stride;
    const value_t* const gv = v + batch_idx * args.kv_batch_stride + head_idx_kv * args.kv_head_stride;
    value_t* const go = o + batch_idx * args.q_batch_stride + head_idx * args.q_head_stride;

    // TODO: need deep dive
    const int q_start = q_blk * Br;
    const int num_kv_blk = int((args.kv_seq_len + Bc - 1) / Bc);
    const int causal_offset = int(args.kv_seq_len - args.q_seq_len);

    const int wrow = lid >> 2;
    const int wcol = (lid & 3) * 2;

    do /* prefetch Q block */ {
        #pragma unroll
        for (int i = 0; i < (Br * HEAD_DIM) / (8 * THREADS); ++i) {
            const int idx = tid + i * THREADS;
            constexpr int thread_per_row = HEAD_DIM / 8;
            const int row = idx / thread_per_row;
            const int col = idx % thread_per_row * 8;
            CpAsync<16>(&sq[col + row * HEAD_DIM], gq + (q_start + row) * args.q_seq_stride + col);
        }
        CpAsyncCommit();
    } while (0);

    do /* prefetch KV blocks */ {
        #pragma unroll
        for (int i = 0; i < (Bc * HEAD_DIM) / (8 * THREADS); ++i) {
            const int idx = tid + i * THREADS;
            constexpr int thread_per_row = HEAD_DIM / 8;
            const int row = idx / thread_per_row;
            const int col = idx % thread_per_row * 8;
            CpAsync<16>(&sk[col + row * HEAD_DIM], gk + row * args.kv_seq_stride + col);
        }
        CpAsyncCommit();
        #pragma unroll
        for (int i = 0; i < (Bc * HEAD_DIM) / (8 * THREADS); ++i) {
            const int idx = tid + i * THREADS;
            constexpr int thread_per_row = HEAD_DIM / 8;
            const int row = idx / thread_per_row;
            const int col = idx % thread_per_row * 8;
            CpAsync<16>(&sv[col + row * HEAD_DIM], gv + row * args.kv_seq_stride + col);
        }
        CpAsyncCommit();
    } while (0);

    float o_acc[HEAD_DIM / 8][4];
    FillZeros(o_acc);

    float m_row[2] = {-INFINITY, -INFINITY};
    float l_row[2] = {0.f, 0.f};

    const float scale = args.softmax_scale;

    for (int ikv = 0; ikv < num_kv_blk; ++ikv) {
        const int kv_start = ikv * Bc;

        if (args.causal) {
            const int max_q_needed = q_start + Br - 1 + causal_offset;
            if (kv_start > max_q_needed) break;
        }

        const int stage = ikv & 1;
        CpAsyncWait<0>();
        __syncthreads();

        if (ikv + 1 < num_kv_blk) /* prefetch overlapped by computation */ {
            const int bufidx = (ikv + 1) % RING_STAGES;
            #pragma unroll
            for (int i = 0; i < (Bc * HEAD_DIM) / (8 * THREADS); ++i) {
                const int idx = tid + i * THREADS;
                constexpr int thread_per_row = HEAD_DIM / 8;
                const int row = idx / thread_per_row;
                const int col = idx % thread_per_row * 8;
                CpAsync<16>(
                    &sk[col + row * HEAD_DIM + bufidx * Bc * HEAD_DIM],
                    gk + ((ikv + 1) * Bc + row)* args.kv_seq_stride + col
                );
            }
            CpAsyncCommit();
            #pragma unroll
            for (int i = 0; i < (Bc * HEAD_DIM) / (8 * THREADS); ++i) {
                const int idx = tid + i * THREADS;
                constexpr int thread_per_row = HEAD_DIM / 8;
                const int row = idx / thread_per_row;
                const int col = idx % thread_per_row * 8;
                CpAsync<16>(
                    &sv[col + row * HEAD_DIM + bufidx * Bc * HEAD_DIM],
                    gv + ((ikv + 1) * Bc + row) * args.kv_seq_stride + col
                );
            }
            CpAsyncCommit();
        }

        // score = Q @ K
        float s_acc[Bc / 8][4];
        FillZeros(s_acc);

        #pragma unroll
        for (int i = 0; i < HEAD_DIM / 16; ++i) {
            uint32_t a[4];
            do /* load matrix Q */ {
                const int row = lid % 16 + wid * 16;
                const int col = i * 16 + ((lid >> 4) * 8);
                LdMatrixX4(&sq[col + row * HEAD_DIM], a[0], a[1], a[2], a[3]);
            } while (0);

            #pragma unroll
            for (int j = 0; j < Bc / 8; ++j) {
                uint32_t b[2];

                do /* load matrix K */ {
                    int h   = lid & 15;
                    int row = j * 8 + (h & 7);
                    int col = i * 16 + ((h < 8) ? 0 : 8);
                    LdMatrixX2(&sk[col + row * HEAD_DIM + stage * Bc * HEAD_DIM], b[0], b[1]);
                } while (0);

                MmaM16N8K16F32Accm<value_t>(
                    s_acc[j][0], s_acc[j][1], s_acc[j][2], s_acc[j][3],
                    a[0], a[1], a[2], a[3],
                    b[0], b[1],
                    s_acc[j][0], s_acc[j][1], s_acc[j][2], s_acc[j][3]
                );
            }
        }

        #pragma unroll
        for (int i = 0; i < Bc / 8; i++) {
            s_acc[i][0] *= scale;
            s_acc[i][1] *= scale;
            s_acc[i][2] *= scale;
            s_acc[i][3] *= scale;
        }

        
        if (args.causal) {
            const int qi0 = q_start + wid * 16 + wrow;
            const int qi1 = qi0 + 8;
            #pragma unroll
            for (int i = 0; i < Bc / 8; i++) {
                const int j = kv_start + i * 8 + wcol;
                if (j     > qi0 + causal_offset) s_acc[i][0] = -INFINITY;
                if (j + 1 > qi0 + causal_offset) s_acc[i][1] = -INFINITY;
                if (j     > qi1 + causal_offset) s_acc[i][2] = -INFINITY;
                if (j + 1 > qi1 + causal_offset) s_acc[i][3] = -INFINITY;
            }
        }


        // update row max
        float row_max0 = -INFINITY, row_max1 = -INFINITY;
        #pragma unroll
        for (int i = 0; i < Bc / 8; i++) {
            row_max0 = fmaxf(row_max0, fmaxf(s_acc[i][0], s_acc[i][1]));
            row_max1 = fmaxf(row_max1, fmaxf(s_acc[i][2], s_acc[i][3]));
        }
        row_max0 = WarpMax4(row_max0);
        row_max1 = WarpMax4(row_max1);

        const float m_new0 = fmaxf(m_row[0], row_max0);
        const float m_new1 = fmaxf(m_row[1], row_max1);

        // alpha factor
        const float alpha0 = __expf(m_row[0] - m_new0);
        const float alpha1 = __expf(m_row[1] - m_new1);

        m_row[0] = m_new0;
        m_row[1] = m_new1;

        // P = exp(S - m_new)
        float p_sum0 = 0.f, p_sum1 = 0.f;
        #pragma unroll
        for (int i = 0; i < Bc / 8; i++) {
            s_acc[i][0] = __expf(s_acc[i][0] - m_new0);
            s_acc[i][1] = __expf(s_acc[i][1] - m_new0);
            s_acc[i][2] = __expf(s_acc[i][2] - m_new1);
            s_acc[i][3] = __expf(s_acc[i][3] - m_new1);
            p_sum0 += s_acc[i][0] + s_acc[i][1];
            p_sum1 += s_acc[i][2] + s_acc[i][3];
        }
        p_sum0 = WarpSum4(p_sum0);
        p_sum1 = WarpSum4(p_sum1);

        // update l by alpha and P
        l_row[0] = l_row[0] * alpha0 + p_sum0;
        l_row[1] = l_row[1] * alpha1 + p_sum1;

        // update o by alpha
        #pragma unroll
        for (int i = 0; i < HEAD_DIM / 8; i++) {
            o_acc[i][0] *= alpha0;
            o_acc[i][1] *= alpha0;
            o_acc[i][2] *= alpha1;
            o_acc[i][3] *= alpha1;
        }

        // P = softmax * V
        #pragma unroll
        for (int i = 0; i < Bc / 16; i++) {
            uint32_t p_acc[4];
            p_acc[0] = PackHalf2(s_acc[2*i+0][0], s_acc[2*i+0][1]);
            p_acc[1] = PackHalf2(s_acc[2*i+0][2], s_acc[2*i+0][3]);
            p_acc[2] = PackHalf2(s_acc[2*i+1][0], s_acc[2*i+1][1]);
            p_acc[3] = PackHalf2(s_acc[2*i+1][2], s_acc[2*i+1][3]);
            
            #pragma unroll
            for (int j = 0; j < HEAD_DIM / 8; j++) {
                uint32_t b[2];
                do /* load matrix V */ {
                    int h   = lid & 15;
                    int row = i * 16 + h;
                    int col = j * 8;
                    LdMatrixX2<true>(&sv[stage * Bc * HEAD_DIM + row * HEAD_DIM + col], b[0], b[1]);
                } while (0);

                MmaM16N8K16F32Accm<value_t>(
                    o_acc[j][0], o_acc[j][1], o_acc[j][2], o_acc[j][3],
                    p_acc[0], p_acc[1], p_acc[2], p_acc[3],
                    b[0], b[1],
                    o_acc[j][0], o_acc[j][1], o_acc[j][2], o_acc[j][3]
                );
            }
        }
        __syncthreads();
    }

    // O = o_acc / l
    const int o_row0 = q_start + wid * 16 + wrow;
    const int o_row1 = o_row0 + 8;
    const float inv_l0 = 1.f / l_row[0];
    const float inv_l1 = 1.f / l_row[1];

    #pragma unroll
    for (int i = 0; i < HEAD_DIM / 8; i++) {
        const int col = i * 8 + wcol;
        go[o_row0 * args.q_seq_stride + col + 0] = __float2half_rn(o_acc[i][0] * inv_l0);
        go[o_row0 * args.q_seq_stride + col + 1] = __float2half_rn(o_acc[i][1] * inv_l0);
        go[o_row1 * args.q_seq_stride + col + 0] = __float2half_rn(o_acc[i][2] * inv_l1);
        go[o_row1 * args.q_seq_stride + col + 1] = __float2half_rn(o_acc[i][3] * inv_l1);
    }

    if (args.lse != nullptr && ((lid & 3) == 0)) {
        float* const plse = reinterpret_cast<float*>(args.lse);
        const int q_pos0 = q_start + wid * 16 + wrow;
        const int q_pos1 = q_pos0 + 8;
        const index_t lse_base = (index_t)batch_idx * args.q_n_heads * args.q_seq_len
                                + (index_t)head_idx * args.q_seq_len;
        plse[lse_base + q_pos0] = m_row[0] + __logf(l_row[0]);
        if (q_pos1 < args.q_seq_len) {
            plse[lse_base + q_pos1] = m_row[1] + __logf(l_row[1]);
        }
    }
}


template <typename value_t>
static void LaunchFlashAttn(const FlashAttnKernelArgs& args, dim3 block, dim3 grid, cudaStream_t stream) {
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(
            (const void*)&FlashAttnKernel<value_t>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            SMEM_BYTES);
        configured = true;
    }
    FlashAttnKernel<value_t><<<grid, block, SMEM_BYTES, stream>>>(args);
    if (cudaError_t err = cudaGetLastError(); err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel launch failed: ")
                                + cudaGetErrorString(err));
    }
}

void FlashAttnDemoLauncher(const FlashAttnKernelArgs& args, int batch_size, int q_sql_len, int q_n_heads, cudaStream_t stream) {
    dim3 grid(q_sql_len / Br, batch_size, q_n_heads);
    dim3 block(THREADS);

    LaunchFlashAttn<half>(args, block, grid, stream);
}

py::object FlashAttnDemoWrapper(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    bool causal, bool return_lse 
) {
    // assuming qkv layout as (B, L, H, D)

    TORCH_CHECK(Q.is_cuda() and K.is_cuda() and V.is_cuda(), "qkv must be located in cuda");
    TORCH_CHECK(Q.scalar_type() == at::kHalf, "Q must be fp16");
    TORCH_CHECK(K.scalar_type() == at::kHalf, "K must be fp16");
    TORCH_CHECK(V.scalar_type() == at::kHalf, "V must be fp16");

    const int64_t seq_q  = Q.size(-3);
    const int64_t seq_kv = K.size(-3);

    TORCH_CHECK(seq_q  % Br == 0, "seq_len_q must be multiple of ", Br);
    TORCH_CHECK(seq_kv % Bc == 0, "seq_len_kv must be multiple of ", Bc);
    TORCH_CHECK(!causal || seq_kv >= seq_q, "causal requires seq_kv >= seq_q for now");

    Q = Q.contiguous();
    K = K.contiguous();
    V = V.contiguous();

    // output tensor
    torch::Tensor O = torch::zeros_like(Q);
    torch::Tensor LSE;
    FlashAttnKernelArgs args{};

    args.q = Q.data_ptr();
    args.k = K.data_ptr();
    args.v = V.data_ptr();
    args.o = O.data_ptr();
    if (return_lse) {
        LSE = torch::empty({Q.size(-4), Q.size(-2), Q.size(-3)}, Q.options().dtype(at::kFloat));
        args.lse = LSE.data_ptr();
    }

    args.q_batch_stride  = Q.stride(-4);
    args.q_seq_stride    = Q.stride(-3);
    args.q_head_stride   = Q.stride(-2);
    args.q_seq_len       = seq_q;
    args.q_n_heads       = Q.size(-2);

    args.kv_batch_stride = K.stride(-4);
    args.kv_seq_stride   = K.stride(-3);
    args.kv_head_stride  = K.stride(-2);
    args.kv_seq_len      = seq_kv;
    args.kv_n_heads      = K.size(-2);

    args.softmax_scale = 1.0 / sqrtf((float)Q.size(-1));
    args.causal = causal ? 1 : 0;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    FlashAttnDemoLauncher(args, Q.size(-4), Q.size(-3), Q.size(-2), stream);
    if (return_lse) return py::make_tuple(O, LSE);
    return py::cast(O);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attn_demo_cu", &FlashAttnDemoWrapper, "flash attn demo written in cuda",
            py::arg("Q"), py::arg("K"), py::arg("V"),
            py::arg("causal") = false,
            py::arg("return_lse") = false);
}