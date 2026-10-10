#pragma once

#include "common.cuh"
#include "arch.cuh"
#include "opt.cuh"

struct FlashAttnKernelArgs {
    void* q;
    void* k;
    void* v;
    void* o;

    int q_batch_stride;
    int q_seq_stride;
    int q_head_stride;
    int q_seq_len;
    int q_n_heads;

    int kv_batch_stride;
    int kv_seq_stride;
    int kv_head_stride;
    int kv_seq_len;
    int kv_n_heads;
};

template <
    typename value_t, typename accum_t = float
>
__global__ void
FlashAttnKernel(FlashAttnKernelArgs args) {
    const int batch_idx = blockIdx.z;
    const int head_idx  = blockIdx.y;
    const int q_seq_blk = blockIdx.x;

    const int tid = threadIdx.x;
    const int lid = tid & 31;
    const int wid = tid >> 5;

    const int qheads_per_kv = args.q_n_heads / args.kv_n_heads;
    const int kv_head_idx = head_idx / qheads_per_kv;

    const int batch_head_q_offset = batch_idx * args.q_batch_stride + head_idx * args.q_head_stride;
    const int batch_head_kv_offset = batch_idx * args.kv_batch_stride + kv_head_idx * args.kv_head_stride;

    const int qo_gmem_block_offset = batch_head_q_offset + q_seq_blk * kBr * args.q_seq_stride;
    const int kv_gmem_block_offset = batch_head_kv_offset;

    const int num_kv_blk = int((args.kv_seq_len + kBc - 1) / kBc);

    const accum_t softmax_scale = rsqrt(static_cast<accum_t>(kHeadDim));

    struct {
        value_t* const q;
        value_t* const k;
        value_t* const v;
        value_t* const o;
    } gmem {
        reinterpret_cast<value_t*>(args.q) + qo_gmem_block_offset,
        reinterpret_cast<value_t*>(args.k) + kv_gmem_block_offset,
        reinterpret_cast<value_t*>(args.v) + kv_gmem_block_offset,
        reinterpret_cast<value_t*>(args.o) + qo_gmem_block_offset
    };

    extern __shared__ __align__(16) char shm[];
    struct {
        value_t* const q;
        value_t* const k;
        value_t* const v;
        value_t* const o;
    } smem {
        reinterpret_cast<value_t*>(shm),
        reinterpret_cast<value_t*>(shm + kBr * kHeadDim * sizeof(value_t)),
        reinterpret_cast<value_t*>(shm + (kBr + kBc) * kHeadDim * sizeof(value_t)),
        reinterpret_cast<value_t*>(shm)
    };

    do /* prefetch Q block */ {
        #pragma unroll
        for (int i = 0; i < kBr * kHeadDim / (kElemPerVec4Access * kNumberOfThreads); ++i) {
            const int idx = tid + i * kNumberOfThreads;
            constexpr int kThreadPerRow = kHeadDim / kElemPerVec4Access;
            const int row = idx / kThreadPerRow;
            const int col = idx % kThreadPerRow * kElemPerVec4Access;
            const int smem_col = SwizzledCol(row, col);
            CpAsync<kBytePerVec4Access>(
                &smem.q[smem_col + row * kHeadDim],
                gmem.q + row * args.q_seq_stride + col
            );
        }
    
        CpAsyncCommit();
    } while (0);

    do /* prefetch first K block */ {
        #pragma unroll
        for (int i = 0; i < kBc * kHeadDim / (kElemPerVec4Access * kNumberOfThreads); ++i) {
            const int idx = tid + i * kNumberOfThreads;
            constexpr int kThreadPerRow = kHeadDim / kElemPerVec4Access;
            const int row = idx / kThreadPerRow;
            const int col = idx % kThreadPerRow * kElemPerVec4Access;
            const int smem_col = SwizzledCol(row, col);
            CpAsync<kBytePerVec4Access>(
                &smem.k[smem_col + row * kHeadDim],
                gmem.k + row * args.kv_seq_stride + col
            );
        }
        CpAsyncCommit();
    } while (0);
    
    accum_t o_acc[kHeadDim / kLdMatrixSize][4];
    FillZeros2D(o_acc);

    accum_t m_row[2] = {-INFINITY, -INFINITY};
    accum_t l_row[2] = {0.f, 0.f};


    for (int ikv = 0; ikv < num_kv_blk; ++ikv) {
        CpAsyncWait<0>();
        __syncthreads();

        do /* fetch current V */ {
            #pragma unroll
            for (int i = 0; i < (kBc * kHeadDim) / (kElemPerVec4Access * kNumberOfThreads); ++i) {
                const int idx = tid + i * kNumberOfThreads;
                constexpr int kThreadPerRow = kHeadDim / kElemPerVec4Access;
                const int row = idx / kThreadPerRow;
                const int col = idx % kThreadPerRow * kElemPerVec4Access;
                const int smem_col = SwizzledCol(row, col);
                CpAsync<16>(
                    &smem.v[smem_col + row * kHeadDim],
                    gmem.v + (ikv * kBc + row) * args.kv_seq_stride + col
                );
            }
            CpAsyncCommit();
        } while (0);

        // score = Q @ K
        float s_acc[kBc / kElemPerVec4Access][4];
        FillZeros2D(s_acc);

        #pragma unroll
        for (int i = 0; i < kHeadDim / kMmaM; ++i) {
            uint32_t a[4];
            do /* load matrix Q */ {
                const int row = lid % 16 + wid * 16;
                const int col = i * 16 + ((lid >> 4) * 8);
                const int smem_col = SwizzledCol(row, col);
                LdMatrixX4(&smem.q[smem_col + row * kHeadDim], a[0], a[1], a[2], a[3]);
            } while (0);

            #pragma unroll
            for (int j = 0; j < kBc / kMmaN; ++j) {
                uint32_t b[2];

                do /* load matrix K */ {
                    const int h   = lid & 15;
                    const int row = j * 8 + (h & 7);
                    const int col = i * 16 + ((h < 8) ? 0 : 8);
                    const int smem_col = SwizzledCol(row, col);
                    LdMatrixX2(&smem.k[smem_col + row * kHeadDim], b[0], b[1]);
                } while (0);

                MmaM16N8K16F32Accm<value_t>(
                    s_acc[j][0], s_acc[j][1], s_acc[j][2], s_acc[j][3],
                    a[0], a[1], a[2], a[3],
                    b[0], b[1],
                    s_acc[j][0], s_acc[j][1], s_acc[j][2], s_acc[j][3]
                );
            }
        }
        
        CpAsyncWait<0>();
        __syncthreads();

        if (ikv + 1 < num_kv_blk) /* prefetch overlapped by computation */ {
            #pragma unroll
            for (int i = 0; i < (kBc * kHeadDim) / (kElemPerVec4Access * kNumberOfThreads); ++i) {
                const int idx = tid + i * kNumberOfThreads;
                constexpr int kThreadPerRow = kHeadDim / kElemPerVec4Access;
                const int row = idx / kThreadPerRow;
                const int col = idx % kThreadPerRow * kElemPerVec4Access;
                const int smem_col = SwizzledCol(row, col);
                CpAsync<16>(
                    &smem.k[smem_col + row * kHeadDim],
                    gmem.k + ((ikv + 1) * kBc + row)* args.kv_seq_stride + col
                );
            }
            CpAsyncCommit();
        }

        #pragma unroll
        for (int i = 0; i < kBc / kElemPerVec4Access; i++) {
            s_acc[i][0] *= softmax_scale;
            s_acc[i][1] *= softmax_scale;
            s_acc[i][2] *= softmax_scale;
            s_acc[i][3] *= softmax_scale;
        }

        do /* softmax */ {
            // update row max
            float row_max0 = -INFINITY, row_max1 = -INFINITY;
            #pragma unroll
            for (int i = 0; i < kBc / kElemPerVec4Access; i++) {
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
            for (int i = 0; i < kBc / kElemPerVec4Access; i++) {
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
            for (int i = 0; i < kHeadDim / kElemPerVec4Access; i++) {
                o_acc[i][0] *= alpha0;
                o_acc[i][1] *= alpha0;
                o_acc[i][2] *= alpha1;
                o_acc[i][3] *= alpha1;
            }
        } while (0);

        #pragma unroll
        for (int i = 0; i < kBc / kMmaM; i++) {
            uint32_t p_acc[4];
            p_acc[0] = PackHalf2(s_acc[2*i+0][0], s_acc[2*i+0][1]);
            p_acc[1] = PackHalf2(s_acc[2*i+0][2], s_acc[2*i+0][3]);
            p_acc[2] = PackHalf2(s_acc[2*i+1][0], s_acc[2*i+1][1]);
            p_acc[3] = PackHalf2(s_acc[2*i+1][2], s_acc[2*i+1][3]);
            
            #pragma unroll
            for (int j = 0; j < kHeadDim / kElemPerVec4Access; ++j) {
                uint32_t b[2];
                do /* load matrix V */ {
                    const int h   = lid & 15;
                    const int row = i * 16 + h;
                    const int col = j * 8;
                    const int smem_col = SwizzledCol(row, col);
                    LdMatrixX2<true>(&smem.v[smem_col + row * kHeadDim], b[0], b[1]);
                } while (0);

                MmaM16N8K16F32Accm<value_t>(
                    o_acc[j][0], o_acc[j][1], o_acc[j][2], o_acc[j][3],
                    p_acc[0], p_acc[1], p_acc[2], p_acc[3],
                    b[0], b[1],
                    o_acc[j][0], o_acc[j][1], o_acc[j][2], o_acc[j][3]
                );
            }
        }
    }
    
    // O = o_acc / l
    const int o_row0 = wid * 16 + (lid >> 2);
    const int o_row1 = o_row0 + 8;

    #pragma unroll
    for (int i = 0; i < kHeadDim / 8; i++) {
        const int col = i * 8 + (lid & 3) * 2;
        gmem.o[o_row0 * args.q_seq_stride + col + 0] = __float2half_rn(o_acc[i][0] / l_row[0]);
        gmem.o[o_row0 * args.q_seq_stride + col + 1] = __float2half_rn(o_acc[i][1] / l_row[0]);
        gmem.o[o_row1 * args.q_seq_stride + col + 0] = __float2half_rn(o_acc[i][2] / l_row[1]);
        gmem.o[o_row1 * args.q_seq_stride + col + 1] = __float2half_rn(o_acc[i][3] / l_row[1]);
    }

    // __syncwarp();
}