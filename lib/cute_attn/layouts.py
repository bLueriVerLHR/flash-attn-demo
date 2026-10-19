"""Shared storage and copy/MMA plans; no runtime launch or caching policy."""
import math
import cutlass
import cutlass.cute as cute
from .config import SharedMemoryPlan, CopyMmaPlan


@cute.jit
def make_swizzle(length: cutlass.Constexpr):
    # This project uses a FLAT row-major layout, not an (8,64) tiled atom.
    # XOR row bits into 16-byte vector bits, matching CUDA SwizzledCol.
    vector_elements_bits = math.log2(8)
    region_elements = min(64, length)
    column_bits = int(math.log2(region_elements)) - vector_elements_bits
    row_shift = int(math.log2(length)) - vector_elements_bits
    return cute.make_swizzle(column_bits, vector_elements_bits, row_shift)


@cute.jit
def make_shared_plan(config: cutlass.Constexpr, head_dim: cutlass.Constexpr):
    swizzle = make_swizzle(head_dim)
    q_layout = cute.make_composed_layout(
        swizzle, 0, cute.make_layout((config.query_rows, head_dim), stride=(head_dim, 1)),
    )
    kv_layout = cute.make_composed_layout(
        swizzle, 0, cute.make_layout((config.kv_rows, head_dim), stride=(head_dim, 1)),
    )

    # A struct documents ownership of the SAME three buffers as CUDA Kernel 2.
    # Alignment remains 16 bytes: no added padding stage or double buffering.
    @cute.struct
    class SharedStorage:
        q: cute.struct.Align[cute.struct.MemRange[cutlass.Float16, cute.cosize(q_layout)], 16]
        k: cute.struct.Align[cute.struct.MemRange[cutlass.Float16, cute.cosize(kv_layout)], 16]
        v: cute.struct.Align[cute.struct.MemRange[cutlass.Float16, cute.cosize(kv_layout)], 16]

    return SharedMemoryPlan(q_layout, kv_layout, SharedStorage)


@cute.jit
def make_copy_mma_plan(config: cutlass.Constexpr, head_dim: cutlass.Constexpr):
    dtype: cutlass.Constexpr = cutlass.Float16
    threads_per_row = head_dim // 8
    copy_rows = config.threads // threads_per_row
    gmem_atom = cute.make_copy_atom(
        cute.nvgpu.cpasync.CopyG2SOp(cute.nvgpu.LoadCacheMode.GLOBAL),
        dtype, num_bits_per_copy=config.copy_bits,
    )
    gmem = cute.make_tiled_copy_tv(
        gmem_atom,
        cute.make_layout((copy_rows, threads_per_row), stride=(threads_per_row, 1)),
        cute.make_layout((1, 8)),
    )
    mma = cute.make_tiled_mma(
        cute.nvgpu.warp.MmaF16BF16Op(dtype, cutlass.Float32, config.mma.shape),
        atom_layout_mnk=(config.warps, 1, 1),
    )
    q = cute.make_tiled_copy_A(
        cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), dtype), mma,
    )
    k = cute.make_tiled_copy_B(
        cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 2), dtype), mma,
    )
    v = cute.make_tiled_copy_B(
        cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(True, 2), dtype), mma,
    )
    output = cute.make_tiled_copy_C(
        cute.make_copy_atom(cute.nvgpu.CopyR2GOp(), dtype, num_bits_per_copy=32), mma,
    )
    return CopyMmaPlan(gmem, q, k, v, output, mma)
