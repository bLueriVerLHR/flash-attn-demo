import torch
import cutlass
import cutlass.cute as cute

from .utils.timer import cuda_timer

MNK = (16, 8, 16)

M16N8K16OP = cute.nvgpu.warp.MmaF16BF16Op(
    cutlass.Float16, cutlass.Float32, MNK
)


@cute.kernel
def mma_kernel(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor, D: cute.Tensor):
    atom = cute.make_mma_atom(M16N8K16OP)

    reps = (A.shape[-2] // MNK[0], B.shape[-1] // MNK[1], A.shape[-1] // MNK[2])
    tile_mma = cute.make_tiled_mma(atom, atom_layout_mnk=reps)


@cute.kernel
def flash_attn_demo_kernel(Q: cute.Tensor, K: cute.Tensor, V: cute.Tensor):
    wid = cute.arch.warp_idx()
    wid = cute.arch.make_warp_uniform(wid)
    lid = cute.arch.lane_idx()

    smem = cutlass.memory.SmemAllocator()


@cute.jit
def flash_attn_demo_host(Q: cute.Tensor, K: cute.Tensor, V: cute.Tensor):
    print("[META] layout: ", Q.layout)
    block = (32, 32)
    grid = (4, 4)
    flash_attn_demo_kernel(Q, K, V).launch(grid=grid, block=block)


def flash_attn_demo(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
    Q = cute.runtime.from_dlpack(Q.transpose(-2, -3))
    K = cute.runtime.from_dlpack(K.transpose(-2, -3))
    V = cute.runtime.from_dlpack(V.transpose(-2, -3))

    flash_attn_demo_host(Q, K, V)