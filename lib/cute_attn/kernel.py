"""CuTe attention kernel and its host launch, following CUDA Kernel 2.

Naming: g/s/r = global/shared/register; tS/tO = score/output partition pattern.
The numbered stages expose the CUDA Kernel 2 prefetch schedule.
"""
import math
import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from .config import (
    KernelConfig, DEFAULT_CONFIG, AttentionTensors, AttentionProblem,
    SharedMemoryPlan, CopyMmaPlan, SoftmaxState,
)
from .layouts import make_shared_plan, make_copy_mma_plan
from .softmax import update_softmax, probability_mma_view


@cute.kernel
def attention_kernel(
    tensors: AttentionTensors,
    problem: cutlass.Constexpr[AttentionProblem],
    memory: SharedMemoryPlan,
    ops: CopyMmaPlan,
    config: cutlass.Constexpr,
):
    Br: cutlass.Constexpr = config.query_rows
    Bc: cutlass.Constexpr = config.kv_rows
    D: cutlass.Constexpr = problem.head_dim
    MmaK: cutlass.Constexpr = config.mma.k
    tid, _, _ = cute.arch.thread_idx()
    q_block, q_head, batch = cute.arch.block_idx()
    q_bs, q_ss, q_hs, _ = tensors.q.stride
    o_bs, o_ss, o_hs, _ = tensors.o.stride
    kv_bs, kv_ss, kv_hs, _ = tensors.k.stride

    storage = cutlass.memory.SmemAllocator().allocate(memory.storage_type)
    sQ = storage.q.get_tensor(memory.q_layout)
    sK = storage.k.get_tensor(memory.kv_layout)
    sV = storage.v.get_tensor(memory.kv_layout)
    # PV's B operand is (N=D,K=Bc). Preserve the original swizzle in this VIEW.
    sVt = cute.composition(sV, cute.make_layout((D, Bc), stride=(Bc, 1)))
    kv_head = q_head // problem.q_heads_per_kv
    q_offset = cute.assume(batch * q_bs + q_head * q_hs + q_block * Br * q_ss, divby=8)
    # A sliced Q can have different strides from its freshly allocated output.
    o_offset = cute.assume(batch * o_bs + q_head * o_hs + q_block * Br * o_ss, divby=8)
    kv_offset = cute.assume(batch * kv_bs + kv_head * kv_hs, divby=8)
    gQ = cute.make_tensor(tensors.q.iterator + q_offset, cute.make_layout((Br, D), stride=(q_ss, 1)))
    gO = cute.make_tensor(tensors.o.iterator + o_offset, cute.make_layout((Br, D), stride=(o_ss, 1)))
    gK = cute.make_tensor(tensors.k.iterator + kv_offset, cute.make_layout((Bc, D), stride=(kv_ss, 1)))

    # 1. Prefetch query and first key tile, exactly as the CUDA prologue.
    gmem_thr = ops.gmem.get_slice(tid)
    tQgQ, tQsQ = gmem_thr.partition_S(gQ), gmem_thr.partition_D(sQ)
    tKsK, tVsV = gmem_thr.partition_D(sK), gmem_thr.partition_D(sV)
    cute.copy(ops.gmem, tQgQ, tQsQ)
    cute.arch.cp_async_commit_group()
    cute.copy(ops.gmem, gmem_thr.partition_S(gK), tKsK)
    cute.arch.cp_async_commit_group()

    mma_thr = ops.mma.get_slice(tid)
    rO = ops.mma.make_fragment_C(mma_thr.partition_shape_C((Br, D)))
    rO.fill(0)
    rM = cute.make_rmem_tensor((2,), cute.Float32)
    rM.fill(-cute.Float32.inf)
    rL = cute.make_rmem_tensor((2,), cute.Float32)
    rL.fill(0)
    q_copy, k_copy, v_copy = ops.q.get_slice(tid), ops.k.get_slice(tid), ops.v.get_slice(tid)
    tSsQ, tSsK = q_copy.partition_S(sQ), k_copy.partition_S(sK)
    tOsVt = v_copy.partition_S(sVt)
    # Keep the already-validated single-reduction-slice operand policy.
    tSrQ = ops.mma.make_fragment_A(mma_thr.partition_shape_A((Br, MmaK)))
    tSrK = ops.mma.make_fragment_B(mma_thr.partition_shape_B((Bc, MmaK)))
    tOrVt = ops.mma.make_fragment_B(mma_thr.partition_shape_B((D, MmaK)))
    tSrQ_copy, tSrK_copy = q_copy.retile(tSrQ), k_copy.retile(tSrK)
    tOrVt_copy = v_copy.retile(tOrVt)

    for ikv in range(problem.kv_blocks):
        # 2. Complete K/Q; also protect previous PV reads before overwriting V.
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        gV = cute.make_tensor(
            tensors.v.iterator + cute.assume(kv_offset + ikv * Bc * kv_ss, divby=8),
            cute.make_layout((Bc, D), stride=(kv_ss, 1)),
        )
        cute.copy(ops.gmem, gmem_thr.partition_S(gV), tVsV)
        cute.arch.cp_async_commit_group()

        # 3. Score QK while current V is being transferred to shared memory.
        rS = ops.mma.make_fragment_C(mma_thr.partition_shape_C((Br, Bc)))
        rS.fill(0)
        for k_slice in cutlass.range_constexpr(D // MmaK):
            cute.copy(ops.q, tSsQ[None, None, k_slice], tSrQ_copy[None, None, 0])
            cute.copy(ops.k, tSsK[None, None, k_slice], tSrK_copy[None, None, 0])
            cute.gemm(ops.mma, rS, tSrQ, tSrK, rS)

        # 4. Current V ready; all K readers finished before next-K overwrite.
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        if ikv + 1 < problem.kv_blocks:
            gK_next = cute.make_tensor(
                tensors.k.iterator + cute.assume(kv_offset + (ikv + 1) * Bc * kv_ss, divby=8),
                cute.make_layout((Bc, D), stride=(kv_ss, 1)),
            )
            cute.copy(ops.gmem, gmem_thr.partition_S(gK_next), tKsK)
            cute.arch.cp_async_commit_group()

        # 5. Online softmax and O rescale; next K overlaps softmax/PV.
        rP = update_softmax(SoftmaxState(rS, rO, rM, rL), problem, config)
        tOrP = probability_mma_view(rP)
        for k_slice in cutlass.range_constexpr(Bc // MmaK):
            # 6. CRITICAL PV FIX: allocation != load. V needs a producer here.
            # This transposed ldmatrix writes the registers consumed by PV MMA.
            cute.copy(ops.v, tOsVt[None, None, k_slice], tOrVt_copy[None, None, 0])
            cute.gemm(ops.mma, rO, tOrP[None, None, k_slice], tOrVt[None, None, 0], rO)

    # 7. Final normalization and packed output; no additional shared stage.
    inv_l0, inv_l1 = cute.Float32(1) / rL[0], cute.Float32(1) / rL[1]
    rO_half = cute.make_rmem_tensor(rO.layout, cutlass.Float16)
    for n in cutlass.range_constexpr(D // config.mma.n):
        rO_half[(0, 0), 0, n] = (rO[(0, 0), 0, n] * inv_l0).to(cutlass.Float16)
        rO_half[(1, 0), 0, n] = (rO[(1, 0), 0, n] * inv_l0).to(cutlass.Float16)
        rO_half[(0, 1), 0, n] = (rO[(0, 1), 0, n] * inv_l1).to(cutlass.Float16)
        rO_half[(1, 1), 0, n] = (rO[(1, 1), 0, n] * inv_l1).to(cutlass.Float16)
    output_thr = ops.output.get_slice(tid)
    cute.copy(ops.output, output_thr.retile(rO_half), output_thr.partition_D(gO))


class AmpereAttention:
    """Build the shared/copy plans and launch the kernel above."""
    def __init__(self, config: KernelConfig = DEFAULT_CONFIG):
        self.config = config

    @cute.jit
    def __call__(self, Q: cute.Tensor, K: cute.Tensor, V: cute.Tensor, O: cute.Tensor, stream: cuda.CUstream):
        batch, length, q_heads, head_dim = Q.shape
        assert Q.element_type == K.element_type == V.element_type == O.element_type == cutlass.Float16
        assert head_dim in (64, 128), 'Only D=64/128 is validated'
        assert Q.shape == O.shape and K.shape == V.shape
        assert batch == K.shape[0] and length == K.shape[1] and head_dim == K.shape[3]
        assert q_heads % K.shape[2] == 0 and K.shape[2] > 0
        assert length > 0 and length % self.config.query_rows == length % self.config.kv_rows == 0
        assert Q.stride[3] == K.stride[3] == V.stride[3] == O.stride[3] == 1
        assert K.stride == V.stride
        problem = AttentionProblem(
            batch, length, q_heads, K.shape[2], head_dim,
            length // self.config.kv_rows, math.log2(math.e) / math.sqrt(head_dim),
        )
        memory = make_shared_plan(self.config, head_dim)
        ops = make_copy_mma_plan(self.config, head_dim)
        attention_kernel(AttentionTensors(Q, K, V, O), problem, memory, ops, self.config).launch(
            grid=(length // self.config.query_rows, q_heads, batch),
            block=(self.config.threads,), stream=stream,
        )
