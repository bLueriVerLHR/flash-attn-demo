import math
import torch
import flash_attn as fa

from lib.utils.timer import cuda_timer
from lib.kernel import flash_attn_demo
from lib.cukernel import flash_attn_demo_cu_V1, flash_attn_demo_cu_V2

@cuda_timer("flash_attn")
def flash_attn(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
    return fa.flash_attn_func(Q, K, V)


@cuda_timer("torch_attn")
def torch_attn(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
    Q = Q.transpose(-2, -3)
    K = K.transpose(-2, -3)
    V = V.transpose(-2, -3)

    KT = K.transpose(-1, -2)
    P = torch.softmax(Q @ KT / math.sqrt(Q.shape[-1]), dim=-1)
    result = P @ V
    return result.transpose(-2, -3)


@cuda_timer("flash_attn_demo")
def flash_attn_demo_wrapper(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
    return flash_attn_demo(Q, K, V)


@cuda_timer("flash_attn_demo_cuda")
def flash_attn_demo_cu_wrapper(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
    return flash_attn_demo_cu_V1.flash_attn_demo_cu(Q, K, V)


@cuda_timer("flash_attn_demo_cuda_v2")
def flash_attn_demo_cu_wrapper_v2(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
    return flash_attn_demo_cu_V2.flash_attn_demo_cu(Q, K, V)


def build_qkv(batch_size: int, seq_len: int, d_model: int):
    config_tuple = (batch_size, seq_len, d_model)
    demoqkv = []
    for _ in range(3):
        x = torch.randn(*config_tuple, dtype=torch.float16, device='cuda')
        demoqkv.append(x)
    return demoqkv


if __name__ == "__main__":
    torch.cuda.set_device(0)
    torch.manual_seed(42)

    config = [
        (4, 1024, 4096, 32),
        (4, 4096, 4096, 32),
        (8, 2048, 4096, 32),
    ]

    for B, L, D, H in config:
        q, k, v = build_qkv(B, L, D)
        q: torch.Tensor = q.reshape(B, L, H, -1)
        k: torch.Tensor = k.reshape(B, L, H, -1)
        v: torch.Tensor = v.reshape(B, L, H, -1)
        print("Q:", q.shape, ", K:", k.shape, ", V:", v.shape)

        # a = torch_attn(q, k, v)
        b = flash_attn(q, k, v)
        # c = flash_attn_demo_wrapper(q, k, v)
        # d = flash_attn_demo_cu_wrapper(q, k, v)
        e = flash_attn_demo_cu_wrapper_v2(q, k, v)

        torch.testing.assert_close(e, b, atol=1e-2, rtol=1e-2)