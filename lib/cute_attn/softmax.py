"""Online softmax and register-layout views, unchanged from the recovery version."""
import cutlass
import cutlass.cute as cute
from .config import SoftmaxState, AttentionProblem


@cute.jit
def warp_max(value: cute.Float32) -> cute.Float32:
    # Each m16n8 row is spread over four adjacent lanes, not the whole warp.
    value = cute.arch.fmax(value, cute.arch.shuffle_sync_bfly(value, 1))
    return cute.arch.fmax(value, cute.arch.shuffle_sync_bfly(value, 2))


@cute.jit
def warp_sum(value: cute.Float32) -> cute.Float32:
    value += cute.arch.shuffle_sync_bfly(value, 1)
    value += cute.arch.shuffle_sync_bfly(value, 2)
    return value


@cute.jit
def update_softmax(state: SoftmaxState, problem: cutlass.Constexpr[AttentionProblem], config: cutlass.Constexpr):
    rS, rO, rM, rL = state.scores, state.output, state.row_max, state.row_sum
    row_max0 = -cute.Float32.inf
    row_max1 = -cute.Float32.inf
    for n in cutlass.range_constexpr(config.kv_rows // config.mma.n):
        rS[(0, 0), 0, n] *= problem.scale_log2
        rS[(1, 0), 0, n] *= problem.scale_log2
        rS[(0, 1), 0, n] *= problem.scale_log2
        rS[(1, 1), 0, n] *= problem.scale_log2
        row_max0 = cute.arch.fmax(row_max0, cute.arch.fmax(rS[(0, 0), 0, n], rS[(1, 0), 0, n]))
        row_max1 = cute.arch.fmax(row_max1, cute.arch.fmax(rS[(0, 1), 0, n], rS[(1, 1), 0, n]))
    m_new0 = cute.arch.fmax(rM[0], warp_max(row_max0))
    m_new1 = cute.arch.fmax(rM[1], warp_max(row_max1))
    # exp2 uses the log2-scaled scores above; accumulators remain FP32.
    alpha0 = cute.math.exp2(rM[0] - m_new0, fastmath=True)
    alpha1 = cute.math.exp2(rM[1] - m_new1, fastmath=True)
    rM[0], rM[1] = m_new0, m_new1
    p_sum0, p_sum1 = cute.Float32(0), cute.Float32(0)
    for n in cutlass.range_constexpr(config.kv_rows // config.mma.n):
        rS[(0, 0), 0, n] = cute.math.exp2(rS[(0, 0), 0, n] - m_new0, fastmath=True)
        rS[(1, 0), 0, n] = cute.math.exp2(rS[(1, 0), 0, n] - m_new0, fastmath=True)
        rS[(0, 1), 0, n] = cute.math.exp2(rS[(0, 1), 0, n] - m_new1, fastmath=True)
        rS[(1, 1), 0, n] = cute.math.exp2(rS[(1, 1), 0, n] - m_new1, fastmath=True)
        p_sum0 += rS[(0, 0), 0, n] + rS[(1, 0), 0, n]
        p_sum1 += rS[(0, 1), 0, n] + rS[(1, 1), 0, n]
    rL[0] = rL[0] * alpha0 + warp_sum(p_sum0)
    rL[1] = rL[1] * alpha1 + warp_sum(p_sum1)
    for n in cutlass.range_constexpr(problem.head_dim // config.mma.n):
        rO[(0, 0), 0, n] *= alpha0
        rO[(1, 0), 0, n] *= alpha0
        rO[(0, 1), 0, n] *= alpha1
        rO[(1, 1), 0, n] *= alpha1
    # Reuse S registers for P, then permit packed FP16 conversion.
    rP = cute.make_rmem_tensor(rS.layout, cutlass.Float16)
    rP.store(rS.load().to(cutlass.Float16))
    return rP


@cute.jit
def probability_mma_view(rP: cute.Tensor):
    shape, stride = rP.layout.shape, rP.layout.stride
    # Pair neighboring N=8 C fragments into one K=16 A operand, in registers.
    return cute.make_tensor(
        rP.iterator,
        cute.make_layout(
            ((shape[0][0], shape[0][1], 2), shape[1], shape[2] // 2),
            stride=((stride[0][0], stride[0][1], stride[2]), stride[1], stride[2] * 2),
        ),
    )
