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

#include "include/flash_fwd_kernel.cuh"

namespace py = pybind11;

using index_t = int64_t;

void FlashAttnDemoLauncher(const FlashAttnKernelArgs& args, int batch_size, int q_seq_len, int q_n_heads, cudaStream_t stream) {
    dim3 grid(q_seq_len / kBr, q_n_heads, batch_size);
    dim3 block(kNumberOfThreads);

    FlashAttnKernel<half><<<grid, block, kSharedMemorySize, stream>>>(args);
    if (cudaError_t err = cudaGetLastError(); err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

torch::Tensor FlashAttnDemoWrapper(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V
) {
    // assuming qkv layout as (B, L, H, D)

    TORCH_CHECK(Q.is_cuda() and K.is_cuda() and V.is_cuda(), "qkv must be located in cuda");
    TORCH_CHECK(Q.scalar_type() == at::kHalf, "Q must be fp16");
    TORCH_CHECK(K.scalar_type() == at::kHalf, "K must be fp16");
    TORCH_CHECK(V.scalar_type() == at::kHalf, "V must be fp16");

    TORCH_CHECK(Q.size(-3) % kBr == 0, "seq_len_q must be multiple of ", kBr);
    TORCH_CHECK(K.size(-3) % kBc == 0 and V.size(-3) % kBc == 0, "seq_len_kv must be multiple of ", kBc);

   TORCH_CHECK(Q.size(-1) == kHeadDim && K.size(-1) == kHeadDim && V.size(-1) == kHeadDim,
               "Q/K/V head_dim must equal ", kHeadDim);

    Q = Q.contiguous();
    K = K.contiguous();
    V = V.contiguous();

    // output tensor
    torch::Tensor O = torch::zeros_like(Q);
    FlashAttnKernelArgs args{};

    args.q = Q.data_ptr();
    args.k = K.data_ptr();
    args.v = V.data_ptr();
    args.o = O.data_ptr();

    args.q_batch_stride  = Q.stride(-4);
    args.q_seq_stride    = Q.stride(-3);
    args.q_head_stride   = Q.stride(-2);
    args.q_seq_len       = Q.size(-3);
    args.q_n_heads       = Q.size(-2);

    args.kv_batch_stride = K.stride(-4);
    args.kv_seq_stride   = K.stride(-3);
    args.kv_head_stride  = K.stride(-2);
    args.kv_seq_len      = K.size(-3);
    args.kv_n_heads      = K.size(-2);

    // args.softmax_scale = 1.0 / sqrtf((float)Q.size(-1));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    FlashAttnDemoLauncher(args, Q.size(-4), Q.size(-3), Q.size(-2), stream);
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attn_demo_cu", &FlashAttnDemoWrapper, "flash attn demo written in cuda",
            py::arg("Q"), py::arg("K"), py::arg("V"));
}