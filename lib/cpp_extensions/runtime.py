"""Load only C++ Kernel 2, lazily and with a distinct extension identity.

The CUDA sources are unchanged. Importing the benchmark CLI does not build V1
or V2; this function is called only when the cuda2 backend is requested.
"""
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def load_kernel2():
    from torch.utils.cpp_extension import load
    source = Path(__file__).with_name('flash_attn_kernel_V2.cu')
    return load(
        name='flash_attn_demo_kernel2', sources=[str(source)],
        extra_cuda_cflags=['-ccbin=/usr/bin/gcc', '-lineinfo'], verbose=False,
    )
