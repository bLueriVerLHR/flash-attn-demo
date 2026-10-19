"""Kernel configuration and grouped CuTe arguments.

Frozen dataclasses group related values without adding GPU allocations. Problem
sizes are passed as Constexpr at device boundaries; tensor pointers stay dynamic.
"""
from dataclasses import dataclass, field
import cutlass.cute as cute


@dataclass(frozen=True)
class MmaShape:
    m: int = 16
    n: int = 8
    k: int = 16

    @property
    def shape(self):
        return (self.m, self.n, self.k)


@dataclass(frozen=True)
class KernelConfig:
    """The validated CUDA Kernel 2 policy, not a tile-tuning interface."""
    query_rows: int = 64
    kv_rows: int = 64
    warps: int = 4
    alignment: int = 16
    copy_bits: int = 128
    mma: MmaShape = field(default_factory=MmaShape)

    def __post_init__(self):
        if (self.query_rows, self.kv_rows, self.warps) != (64, 64, 4):
            raise ValueError('CUDA Kernel 2 requires Br=Bc=64 and four warps')
        if self.mma.shape != (16, 8, 16) or self.alignment != 16 or self.copy_bits != 128:
            raise ValueError('Only m16n8k16, 16-byte alignment and 128-bit copies are validated')

    @property
    def threads(self):
        return 32 * self.warps


DEFAULT_CONFIG = KernelConfig()


@dataclass(frozen=True)
class AttentionTensors:
    q: cute.Tensor
    k: cute.Tensor
    v: cute.Tensor
    o: cute.Tensor


@dataclass(frozen=True)
class AttentionProblem:
    # Plain int fields alone are not static. Use Constexpr[AttentionProblem]
    # where these dimensions determine register-fragment sizes.
    batch_size: int
    seq_len: int
    q_heads: int
    kv_heads: int
    head_dim: int
    kv_blocks: int
    scale_log2: float

    @property
    def q_heads_per_kv(self):
        return self.q_heads // self.kv_heads


@dataclass(frozen=True)
class SharedMemoryPlan:
    q_layout: cute.ComposedLayout
    kv_layout: cute.ComposedLayout
    storage_type: type


@dataclass(frozen=True)
class CopyMmaPlan:
    gmem: cute.TiledCopy
    q: cute.TiledCopy
    k: cute.TiledCopy
    v: cute.TiledCopy
    output: cute.TiledCopy
    mma: cute.TiledMma


@dataclass(frozen=True)
class SoftmaxState:
    scores: cute.Tensor
    output: cute.Tensor
    row_max: cute.Tensor
    row_sum: cute.Tensor
