import os

from torch.utils.cpp_extension import load

_base_path = os.path.dirname(os.path.abspath(__file__))

flash_attn_demo_cu_V1 = load(
    name="flash_attn_demo_cu",
    sources=[os.path.join(_base_path, "flash_attn_kernel_V1.cu")],
    extra_cuda_cflags=["-ccbin=/usr/bin/gcc", "-lineinfo", "-Xptxas=-v"],
    verbose=True
)

flash_attn_demo_cu_V2 = load(
    name="flash_attn_demo_cu",
    sources=[os.path.join(_base_path, "flash_attn_kernel_V2.cu")],
    extra_cuda_cflags=["-ccbin=/usr/bin/gcc", "-lineinfo", "-Xptxas=-v"],
    verbose=True
)