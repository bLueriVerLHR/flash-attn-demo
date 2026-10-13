import math
import torch
import flash_attn as fa

from lib.utils.timer import cuda_timer
from lib.cute_attn.kernel import flash_attn_demo
from lib.cpp_extensions.cukernel import flash_attn_demo_cu_V1, flash_attn_demo_cu_V2

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
def flash_attn_demo_cute_wrapper(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
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
        (1, 128, 1, 128),
        (4, 4096, 32, 128),
        (8, 2048, 32, 128),
    ]

    for B, L, H, D in config:
        q = torch.randn((B, L, H, D), device="cuda", dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        print("Q:", q.shape, ", K:", k.shape, ", V:", v.shape)
        
        # a = torch_attn(q, k, v)
        b = flash_attn(q, k, v)
        
        with torch.cuda.nvtx.range("attn_v2_B4_L4096_H32_D128"):
            c = flash_attn_demo_cute_wrapper(q, k, v)
            d = flash_attn_demo_cu_wrapper(q, k, v)
            e = flash_attn_demo_cu_wrapper_v2(q, k, v)

            # ref = v.float().mean(dim=1, keepdim=True).expand_as(v)

            # error = (c.float() - ref).abs()
            # print(
            #     f"L={L}, "
            #     f"max_abs={error.max().item():.6f}, "
            #     f"mean_abs={error.mean().item():.6f}"
            # )

            # torch.testing.assert_close(
            #     c.float(), ref,
            #     atol=1e-2, rtol=1e-2,
            # )
        # print(
        #     "[REF o00]",
        #     "kv=0:", v[0, :64, 0, 0].float().sum().item(),
        #     "kv=1:", v[0, :128, 0, 0].float().sum().item(),
        # )

        # print(
        #     "[FINAL 00]",
        #     "actual:", c[0, 0, 0, 0].item(),
        #     "expected:", v[0, :128, 0, 0].float().mean().item(),
        # )
        # torch.testing.assert_close(c, b, atol=1e-2, rtol=1e-2)