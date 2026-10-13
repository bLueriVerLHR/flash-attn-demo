import math
import torch
import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda


def _is_same(*values) -> bool:
    return not values or all([values[0] == val for val in values])


@cute.jit
def make_swizzle(len: cutlass.Constexpr, element_bits: cutlass.Constexpr):
    element_bytes = element_bits // 8
    chunk_bytes = 16
    bytes_per_row = len * element_bytes
    swizzle_bytes = min(128, 2 ** (bytes_per_row.bit_length() - 1))
    if cutlass.const_expr(swizzle_bytes < chunk_bytes):
        return cute.make_swizzle(0, 4, 3)

    m_base = int(math.log2(chunk_bytes // element_bytes))
    s_shift = int(math.log2(swizzle_bytes // chunk_bytes))
    b_bits = s_shift

    return cute.make_swizzle(b_bits, m_base, s_shift)


@cute.jit
def warp_max(val: cute.Float32) -> cute.Float32:
    other1 = cute.arch.shuffle_sync_bfly(val, 1)
    val = cute.arch.fmax(val, other1)
    other2 = cute.arch.shuffle_sync_bfly(val, 2)
    val = cute.arch.fmax(val, other2)
    return val


@cute.jit
def warp_sum(val: cute.Float32) -> cute.Float32:
    val += cute.arch.shuffle_sync_bfly(val, 1)
    val += cute.arch.shuffle_sync_bfly(val, 2)
    return val


@cute.kernel
def flash_attn_demo_kernel(
        Q: cute.Tensor,
        K: cute.Tensor,
        V: cute.Tensor,
        O: cute.Tensor,
        *,
        Br: cutlass.Constexpr,
        Bc: cutlass.Constexpr,

        batch_size: cutlass.Constexpr,
        seq_len: cutlass.Constexpr,
        d_head: cutlass.Constexpr,
        
        q_heads_per_kv: cutlass.Constexpr,
        number_of_kv_blocks: cutlass.Constexpr,
        softmax_scale: cutlass.Constexpr,
        warp_size: cutlass.Constexpr,
        warps_per_cta: cutlass.Constexpr,
        f16_per_bites: cutlass.Constexpr,

        MmaM: cutlass.Constexpr,
        MmaN: cutlass.Constexpr,
        MmaK: cutlass.Constexpr,

        q_batch_stride: cutlass.Constexpr,
        q_seq_stride: cutlass.Constexpr,
        q_head_stride: cutlass.Constexpr,
        q_layout: cute.ComposedLayout,

        kv_batch_stride: cutlass.Constexpr,
        kv_seq_stride: cutlass.Constexpr,
        kv_head_stride: cutlass.Constexpr,
        kv_layout: cute.ComposedLayout,

        dtype: cutlass.Constexpr,
        alignment: cutlass.Constexpr,

        # copies
        g2s: cute.TiledCopy,
        s2r_a_qk: cute.TiledCopy,
        s2r_b_qk: cute.TiledCopy,

        s2r_b_pv: cute.TiledCopy,

        r2g_o: cute.TiledCopy,
        r2s_o: cute.TiledCopy,
        s2g_o: cute.TiledCopy,

        mma_qk: cute.TiledMma,
        mma_pv: cute.TiledMma,
    ):
    tid, _, _ = cute.arch.thread_idx()
    q_seq_block, head_idx, batch_idx = cute.arch.block_idx()

    smalloc = cutlass.memory.SmemAllocator()

    sQ = smalloc.allocate_tensor(dtype, q_layout, alignment)
    sK = smalloc.allocate_tensor(dtype, kv_layout, alignment)
    sV = smalloc.allocate_tensor(dtype, kv_layout, alignment)

    # print(f"""[META] Shared Memory Attributes
    # Q: {sQ}
    # K: {sK}
    # V: {sV}""")

    kv_head_idx = head_idx // q_heads_per_kv
    batch_head_qo_offset = batch_idx * q_batch_stride + head_idx * q_head_stride
    batch_head_kv_offset = batch_idx * kv_batch_stride + kv_head_idx * kv_head_stride

    # Offsets are in FP16 elements: 8 elements = 16 bytes.
    qo_gmem_block_offset = cute.assume(
        q_seq_block * Br * q_seq_stride + batch_head_qo_offset,
        divby=8,
    )
    kv_gmem_block_offset = cute.assume(batch_head_kv_offset, divby=8)

    gQ = cute.make_tensor(Q.iterator + qo_gmem_block_offset, cute.make_layout((Br, d_head), stride=(q_seq_stride, 1)))
    gO = cute.make_tensor(O.iterator + qo_gmem_block_offset, cute.make_layout((Br, d_head), stride=(q_seq_stride, 1)))
    
    # prefetch QK
    thr_g2s: cute.ThrCopy = g2s.get_slice(tid)
    cute.copy(g2s, thr_g2s.partition_S(gQ), thr_g2s.partition_D(sQ))
    cute.arch.cp_async_commit_group()

    gK = cute.make_tensor(K.iterator + kv_gmem_block_offset, cute.make_layout((Bc, d_head), stride=(kv_seq_stride, 1)))
    cute.copy(g2s, thr_g2s.partition_S(gK), thr_g2s.partition_D(sK))
    cute.arch.cp_async_commit_group()

    thr_mma_qk = mma_qk.get_slice(tid)
    thr_mma_pv = mma_pv.get_slice(tid)

    o_acc: cute.Tensor = mma_pv.make_fragment_C(thr_mma_pv.partition_shape_C((Br, d_head)))
    o_acc.fill(0)

    m_row: cute.Tensor = cute.make_rmem_tensor((2, ), cute.Float32)
    m_row.fill(-cute.Float32.inf)

    l_row: cute.Tensor = cute.make_rmem_tensor((2, ), cute.Float32)
    l_row.fill(0)

    thr_s2r_a_qk = s2r_a_qk.get_slice(tid)
    thr_s2r_b_qk = s2r_b_qk.get_slice(tid)

    tCsQ = thr_s2r_a_qk.partition_S(sQ)
    tCsK = thr_s2r_b_qk.partition_S(sK)

    for ikv in cutlass.range_constexpr(number_of_kv_blocks):
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

        gV = cute.make_tensor(
            V.iterator + cute.assume(
                kv_gmem_block_offset + ikv * Bc * kv_seq_stride,
                divby=8,
            ),
            cute.make_layout((Bc, d_head), stride=(kv_seq_stride, 1)))
        cute.copy(g2s, thr_g2s.partition_S(gV), thr_g2s.partition_D(sV))
        cute.arch.cp_async_commit_group()

        tCrQ = mma_qk.make_fragment_A(thr_mma_qk.partition_shape_A((Br, d_head)))
        tCrK = mma_qk.make_fragment_B(thr_mma_qk.partition_shape_B((Bc, d_head)))
        tCrV = mma_pv.make_fragment_B(thr_mma_pv.partition_shape_B((d_head, Bc)))

        tCrQ_copy = thr_s2r_a_qk.retile(tCrQ)
        tCrK_copy = thr_s2r_b_qk.retile(tCrK)

        cute.copy(s2r_a_qk, tCsQ, tCrQ_copy)
        cute.copy(s2r_b_qk, tCsK, tCrK_copy)

        s_acc = mma_qk.make_fragment_C(thr_mma_qk.partition_shape_C((Br, Bc)))
        s_acc.fill(0)

        # print(f"[META] S=QK+S: {s_acc}, {tCrQ}, {tCrK}, {s_acc}")
        cute.gemm(mma_qk, s_acc, tCrQ, tCrK, s_acc)

        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

        if ikv + 1 < number_of_kv_blocks:
            gK = cute.make_tensor(
                K.iterator + cute.assume(
                    kv_gmem_block_offset + (ikv + 1) * Bc * kv_seq_stride,
                    divby=8,
                ),
                cute.make_layout((Bc, d_head), stride=(kv_seq_stride, 1)))
            cute.copy(g2s, thr_g2s.partition_S(gK), thr_g2s.partition_D(sK))
            cute.arch.cp_async_commit_group()

        # softmax
        row_max0 = -cute.Float32.inf
        row_max1 = -cute.Float32.inf

        for i in cutlass.range_constexpr(Bc // MmaN):
            s_acc[(0, 0), 0, i] *= softmax_scale
            s_acc[(1, 0), 0, i] *= softmax_scale
            s_acc[(0, 1), 0, i] *= softmax_scale
            s_acc[(1, 1), 0, i] *= softmax_scale

            row_max0 = cute.arch.fmax(row_max0, cute.arch.fmax(s_acc[(0, 0), 0, i], s_acc[(1, 0), 0, i]))
            row_max1 = cute.arch.fmax(row_max1, cute.arch.fmax(s_acc[(0, 1), 0, i], s_acc[(1, 1), 0, i]))

        row_max0 = warp_max(row_max0)
        row_max1 = warp_max(row_max1)

        # online softmax update
        m_new0 = cute.arch.fmax(m_row[0], row_max0)
        m_new1 = cute.arch.fmax(m_row[1], row_max1)

        alpha0 = cute.math.exp2(m_row[0] - m_new0)
        alpha1 = cute.math.exp2(m_row[1] - m_new1)

        m_row[0] = m_new0
        m_row[1] = m_new1

        p_acc = mma_qk.make_fragment_C(thr_mma_qk.partition_shape_C((Br, Bc)))

        p_sum0 = cute.Float32(0.0)
        p_sum1 = cute.Float32(0.0)

        for i in cutlass.range_constexpr(Bc // MmaN):
            p_acc[(0, 0), 0, i] = cute.math.exp2(s_acc[(0, 0), 0, i] - m_new0)
            p_acc[(1, 0), 0, i] = cute.math.exp2(s_acc[(1, 0), 0, i] - m_new0)
            p_acc[(0, 1), 0, i] = cute.math.exp2(s_acc[(0, 1), 0, i] - m_new1)
            p_acc[(1, 1), 0, i] = cute.math.exp2(s_acc[(1, 1), 0, i] - m_new1)

            p_sum0 += p_acc[(0, 0), 0, i] + p_acc[(1, 0), 0, i]
            p_sum1 += p_acc[(0, 1), 0, i] + p_acc[(1, 1), 0, i]

        p_sum0 = warp_sum(p_sum0)
        p_sum1 = warp_sum(p_sum1)

        # l_new = alpha * l_old + sum(P)
        l_row[0] = l_row[0] * alpha0 + p_sum0
        l_row[1] = l_row[1] * alpha1 + p_sum1

        for i in cutlass.range_constexpr(d_head // MmaN):
            o_acc[(0, 0), 0, i] *= alpha0
            o_acc[(1, 0), 0, i] *= alpha0
            o_acc[(0, 1), 0, i] *= alpha1
            o_acc[(1, 1), 0, i] *= alpha1

        p_f16 = cute.make_rmem_tensor(p_acc.layout, dtype)

        for i in cutlass.range_constexpr(Bc // MmaN):
            p_f16[(0, 0), 0, i] = p_acc[(0, 0), 0, i].to(dtype)
            p_f16[(1, 0), 0, i] = p_acc[(1, 0), 0, i].to(dtype)
            p_f16[(0, 1), 0, i] = p_acc[(0, 1), 0, i].to(dtype)
            p_f16[(1, 1), 0, i] = p_acc[(1, 1), 0, i].to(dtype)


        p_shape = p_f16.layout.shape
        p_stride = p_f16.layout.stride

        p_frag = cute.make_tensor(
            p_f16.iterator,
            cute.make_layout(
                ((p_shape[0][0], p_shape[0][1], 2), p_shape[1], p_shape[2] // 2),
                stride=((p_stride[0][0], p_stride[0][1], p_stride[2]), p_stride[1], p_stride[2] * 2),
            ),
        )

        cute.gemm(mma_pv, o_acc, p_frag, tCrV, o_acc)
        
    inv_l0 = cute.Float32(1.0) / l_row[0]
    inv_l1 = cute.Float32(1.0) / l_row[1]

    # Preserve the MMA C-fragment layout.
    o_f16 = cute.make_rmem_tensor(o_acc.layout, dtype)

    for i in cutlass.range_constexpr(d_head // MmaN):
        o_f16[(0, 0), 0, i] = (o_acc[(0, 0), 0, i] * inv_l0).to(dtype)
        o_f16[(1, 0), 0, i] = (o_acc[(1, 0), 0, i] * inv_l0).to(dtype)
        o_f16[(0, 1), 0, i] = (o_acc[(0, 1), 0, i] * inv_l1).to(dtype)
        o_f16[(1, 1), 0, i] = (o_acc[(1, 1), 0, i] * inv_l1).to(dtype)

    thr_r2g_o = r2g_o.get_slice(tid)

    tCrO = thr_r2g_o.retile(o_f16)
    tCgO = thr_r2g_o.partition_D(gO)

    cute.copy(r2g_o, tCrO, tCgO)


@cute.jit
def flash_attn_demo_host(
        Q: cute.Tensor,
        K: cute.Tensor,
        V: cute.Tensor,
        O: cute.Tensor,
        stream: cuda.CUstream,
        *,
        Br: cutlass.Constexpr = 64,
        Bc: cutlass.Constexpr = 64,
        MmaM: cutlass.Constexpr = 16,
        MmaN: cutlass.Constexpr = 8,
        MmaK: cutlass.Constexpr = 16,
        alignment: cutlass.Constexpr = 16,
    ):
    q_batch_stride, q_seq_stride, q_head_stride, _ = Q.stride
    k_batch_stride, k_seq_stride, k_head_stride, _ = K.stride

    q_batch_size, q_seq_len, q_n_heads, q_d_head = Q.shape
    k_batch_size, k_seq_len, k_n_heads, k_d_head = K.shape
    v_batch_size, v_seq_len, v_n_heads, v_d_head = V.shape

    assert _is_same(q_batch_size, k_batch_size, v_batch_size), "batch size not equal"
    assert _is_same(q_seq_len, k_seq_len, v_seq_len), "seq len not equal"
    assert _is_same(k_n_heads, v_n_heads), "kv has different number of heads"
    assert q_n_heads % k_n_heads == 0, "the number of q heads should be a multiple of the number of k heads"
    assert _is_same(q_d_head, k_d_head, v_d_head), "dimension of heads not equal"

    batch_size, seq_len, d_head = q_batch_size, q_seq_len, q_d_head
    assert seq_len % Br == 0, f"currently only support seq_len({seq_len}) % Br({Br}) == 0"
    assert seq_len % Bc == 0, f"currently only support seq_len({seq_len}) % Bc({Bc}) == 0"

    kv_n_heads = k_n_heads
    kv_batch_stride, kv_seq_stride, kv_head_stride = k_batch_stride, k_seq_stride, k_head_stride

    warp_size: cutlass.Constexpr = 32
    warps_per_cta: cutlass.Constexpr = 4
    block: cutlass.Constexpr = (warp_size * warps_per_cta, )
    grid: cutlass.Constexpr = (seq_len // Br, q_n_heads, batch_size)

    q_heads_per_kv: cutlass.Constexpr = q_n_heads // kv_n_heads
    number_of_kv_blocks: cutlass.Constexpr = seq_len // Bc
    softmax_scale: cutlass.Constexpr = 1 / math.sqrt(d_head) * math.log2(math.exp(1))
    # smem_size = (Br + Bc + Bc) * d_head

    # print(f"""[META] Compile Time Variables
    # Br={Br}, Bc={Bc}, d_head={q_d_head}, seq_len={seq_len}, batch_size={batch_size}
    # grid={grid}, block={block}
    # Q: layout={Q.layout}, n_heads={q_n_heads}
    # K: layout={K.layout}, n_heads={kv_n_heads}
    # V: layout={V.layout}, n_heads={kv_n_heads}""")

    f16_bits: cutlass.Constexpr = 16
    f16_bytes: cutlass.Constexpr = f16_bits // 8
    vec_bits: cutlass.Constexpr = 128
    f16_per_bites: cutlass.Constexpr = vec_bits // f16_bits
    dtype: cutlass.Constexpr = cutlass.Float16

    swizzle = make_swizzle(d_head, f16_bits)
    q_layout = cute.make_composed_layout(swizzle, 0, cute.make_layout((Br, d_head), stride=(d_head, 1)))
    kv_layout = cute.make_composed_layout(swizzle, 0, cute.make_layout((Bc, d_head), stride=(d_head, 1)))

    gm2sm_atom = cute.make_copy_atom(
        cute.nvgpu.cpasync.CopyG2SOp(
            # data in sm are all readonly
            # sm_86 not support sm2gm async ops
            cute.nvgpu.LoadCacheMode.GLOBAL
        ),
        dtype,
        num_bits_per_copy=vec_bits
    )

    threads_per_row: cutlass.Constexpr = d_head // f16_per_bites
    copy_rows: cutlass.Constexpr = warp_size * warps_per_cta // threads_per_row

    thr_layout = cute.make_layout(
        (copy_rows, threads_per_row),
        stride=(threads_per_row, 1),
    )
    val_layout = cute.make_layout((1, f16_per_bites))
    g2s = cute.make_tiled_copy_tv(gm2sm_atom, thr_layout, val_layout)

    rf2gm_atom = cute.make_copy_atom(
        cute.nvgpu.CopyR2GOp(),
        dtype,
        num_bits_per_copy=32
    )

    ldmatrix_x4_atom = cute.make_copy_atom(
        cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), dtype)

    ldmatrix_x2_atom = cute.make_copy_atom(
        cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 2), dtype)

    ldmatrix_x2t_atom = cute.make_copy_atom(
        cute.nvgpu.warp.LdMatrix8x8x16bOp(True, num_matrices=2), dtype)

    op = cute.nvgpu.warp.MmaF16BF16Op(
        dtype, cutlass.Float32, (MmaM, MmaN, MmaK)
    )
    mma_qk = cute.make_tiled_mma(
        op,
        atom_layout_mnk=(warps_per_cta, 1, 1),
    )
    
    mma_pv = cute.make_tiled_mma(
        op,
        atom_layout_mnk=(warps_per_cta, 1, 1),
    )
    
    s2r_a_qk = cute.make_tiled_copy_A(ldmatrix_x4_atom, mma_qk)
    s2r_b_qk = cute.make_tiled_copy_B(ldmatrix_x2_atom, mma_qk)
    s2r_b_pv = cute.make_tiled_copy_B(ldmatrix_x2t_atom, mma_pv)
    r2g_o = cute.make_tiled_copy_C(rf2gm_atom, mma_pv)
    
    flash_attn_demo_kernel(
        Q, K, V, O,
        Br=Br, Bc=Bc,
        batch_size=batch_size,
        seq_len=seq_len,
        d_head=d_head,
        
        q_heads_per_kv=q_heads_per_kv,
        number_of_kv_blocks=number_of_kv_blocks,
        softmax_scale=softmax_scale,
        warp_size=warp_size,
        warps_per_cta=warps_per_cta,
        f16_per_bites=f16_per_bites,

        MmaM=MmaM,
        MmaN=MmaN,
        MmaK=MmaK,
        
        q_batch_stride=q_batch_stride,
        q_seq_stride=q_seq_stride,
        q_head_stride=q_head_stride,
        q_layout=q_layout,

        kv_batch_stride=kv_batch_stride,
        kv_seq_stride=kv_seq_stride,
        kv_head_stride=kv_head_stride,
        kv_layout=kv_layout,

        dtype=dtype,
        alignment=alignment,

        g2s=g2s,
        s2r_a_qk=s2r_a_qk,
        s2r_b_qk=s2r_b_qk,
        s2r_b_pv=s2r_b_pv,

        r2g_o=r2g_o,
        r2s_o=None,
        s2g_o=None,

        mma_qk=mma_qk,
        mma_pv=mma_pv,
    ).launch(
        grid=grid,
        block=block,
        stream=stream,
    )

def flash_attn_demo(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
    outO = torch.empty_like(Q)
    stream = cuda.CUstream(torch.cuda.current_stream(Q.device).cuda_stream)
    Q = cute.runtime.from_dlpack(Q, assumed_align=16)
    K = cute.runtime.from_dlpack(K, assumed_align=16)
    V = cute.runtime.from_dlpack(V, assumed_align=16)
    cuO = cute.runtime.from_dlpack(outO, assumed_align=16)
    
    flash_attn_demo_host(Q, K, V, cuO, stream)
    return outO