"""Python policy and compiled launchers; tensor data is never held in the cache.

TVM-FFI accepts Torch tensors directly and gets the CURRENT stream from its
caller environment. DLPack is retained as an explicit, slower reference path.
"""
from contextlib import nullcontext
from dataclasses import dataclass
from functools import lru_cache
import importlib.util
import torch
import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from .config import KernelConfig, DEFAULT_CONFIG
from .kernel import AmpereAttention


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple
    stride: tuple

    @classmethod
    def of(cls, tensor):
        return cls(tuple(tensor.shape), tuple(tensor.stride()))


@dataclass(frozen=True)
class AttentionSignature:
    device: int
    tensors: tuple
    config: KernelConfig
    backend: str


@lru_cache(maxsize=3)
def resolve_backend(backend):
    if backend not in ('auto', 'ffi', 'dlpack'):
        raise ValueError("backend must be 'auto', 'ffi' or 'dlpack'")
    available = importlib.util.find_spec('tvm_ffi') is not None
    if backend == 'ffi' and not available:
        raise RuntimeError('TVM-FFI is unavailable; use backend=dlpack or install apache-tvm-ffi')
    return ('ffi' if available else 'dlpack') if backend == 'auto' else backend


def _device_context(device):
    # Avoid an unnecessary device-switch context on the ordinary same-device path.
    return nullcontext() if torch.cuda.current_device() == device else torch.cuda.device(device)


def _check_tensor(tensor, device):
    if torch.is_grad_enabled() and tensor.requires_grad:
        raise ValueError('CuTe demo is inference-only; run under no_grad/inference_mode')
    if not tensor.is_cuda or tensor.dtype != torch.float16 or tensor.ndim != 4:
        raise ValueError('Q/K/V/O must be rank-4 CUDA FP16 tensors')
    if tensor.device.index != device or tensor.data_ptr() % 16:
        raise ValueError('Q/K/V/O must share a device and be 16-byte aligned')
    if tensor.stride(-1) != 1 or any(s <= 0 or s % 8 for s in tensor.stride()[:-1]):
        raise ValueError('Head dim must be contiguous; row/head/batch strides need 16-byte alignment')


def _check_output_storage(q, k, v, out):
    # Never permit an output to overwrite keys/values still read by other CTAs.
    output_storage = out.untyped_storage().data_ptr()
    if any(x.untyped_storage().data_ptr() == output_storage for x in (q, k, v)):
        raise ValueError('Output storage must not alias Q, K or V')


def _check_output_layout(spec):
    # Positive/aligned strides are NOT enough: as_strided can make different
    # CTAs write the same element. Require disjoint address spans. This admits
    # ordinary permutations and sliced rows with gaps; exotic interleaved views
    # are conservatively rejected. Prepared launch reuses the validated strides.
    span = 1
    dimensions = sorted((stride, size) for size, stride in zip(spec.shape, spec.stride) if size > 1)
    for stride, size in dimensions:
        if stride < span:
            raise ValueError('Output layout must be non-overlapping; overlapping/interleaved strides are unsupported')
        span += (size - 1) * stride


def _signature(q, k, v, out, config, backend):
    device = q.device.index
    for tensor in (q, k, v, out):
        _check_tensor(tensor, device)
    b, length, heads, dim = q.shape
    if b < 1 or length < 1 or heads < 1 or dim not in (64, 128):
        raise ValueError('Positive batch/sequence/heads and D=64/128 are required')
    if length % config.query_rows or length % config.kv_rows:
        raise ValueError('Sequence length must be a multiple of 64; no tail masking is implemented')
    if k.shape != v.shape or tuple(out.shape) != tuple(q.shape):
        raise ValueError('K/V shapes and Q/O shapes must match')
    if (b, length, dim) != (k.shape[0], k.shape[1], k.shape[3]):
        raise ValueError('Q/K/V batch, sequence and head dimension must match')
    if k.shape[2] < 1 or heads % k.shape[2] or k.stride() != v.stride():
        raise ValueError('Invalid GQA heads or mismatching K/V strides')
    _check_output_storage(q, k, v, out)
    specs = tuple(TensorSpec.of(t) for t in (q, k, v, out))
    _check_output_layout(specs[3])
    return AttentionSignature(device, specs, config, backend)


class PreparedAttention:
    """A compiled layout specialization, NOT a binding to one set of tensors.

    Use launch(q,k,v,out) to reuse output storage in repeated inference. Native
    pointers and stream are fresh on every call. The object retains only metadata
    and executable code, not the tensors supplied during prepare_attention.
    """
    def __init__(self, signature):
        self.signature = signature
        fake_tensors = [cute.runtime.make_fake_tensor(
            cutlass.Float16, spec.shape, stride=spec.stride, assumed_align=16,
        ) for spec in signature.tensors]
        ffi = signature.backend == 'ffi'
        fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=ffi)
        self.compiled = cute.compile(
            AmpereAttention(signature.config), *fake_tensors, fake_stream,
            options='--enable-tvm-ffi' if ffi else '',
        )

    @property
    def backend(self):
        return self.signature.backend

    def _launch(self, q, k, v, out):
        with _device_context(self.signature.device):
            if self.backend == 'ffi':
                # No from_dlpack, Python CUstream query, or stale stream argument.
                self.compiled(q, k, v, out)
            else:
                tensors = [cute.runtime.from_dlpack(t, assumed_align=16) for t in (q, k, v, out)]
                stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
                self.compiled(*tensors, stream)
        return out

    def launch(self, q, k, v, out):
        for tensor, spec in zip((q, k, v, out), self.signature.tensors):
            _check_tensor(tensor, self.signature.device)
            if tuple(tensor.shape) != spec.shape or tuple(tensor.stride()) != spec.stride:
                raise ValueError('Prepared layout changed; prepare a new launcher for this shape/stride')
        _check_output_storage(q, k, v, out)
        return self._launch(q, k, v, out)

    def __call__(self, q, k, v, *, out=None):
        if out is None:
            out = torch.empty_like(q)
        return self.launch(q, k, v, out)


@lru_cache(maxsize=32)
def _compile_attention(signature):
    """Cache only metadata/executable; current tensors and stream are not keys."""
    with _device_context(signature.device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('Prepare/warm up attention before CUDA Graph capture')
        return PreparedAttention(signature)


def prepare_attention(q, k, v, *, out=None, config=DEFAULT_CONFIG, backend='auto'):
    """Compile/cache a layout once; reuse the returned launcher with new tensors."""
    if out is None:
        out = torch.empty_like(q)
    signature = _signature(q, k, v, out, config, resolve_backend(backend))
    return _compile_attention(signature)


def flash_attn_demo(q, k, v, *, out=None, config=DEFAULT_CONFIG, backend='auto'):
    """Convenience FP16 non-causal attention in (B,L,H,D) layout.

    No masking, causal attention, dropout or autograd is implemented. First use
    of new metadata compiles; callers should warm up before timing or capture.
    """
    if torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v)):
        raise ValueError('CuTe demo is inference-only; no backward implementation')
    if out is None:
        out = torch.empty_like(q)
    prepared = prepare_attention(q, k, v, out=out, config=config, backend=backend)
    # prepare_attention already validated exactly these tensors; do not repeat it.
    return prepared._launch(q, k, v, out)


def cache_info():
    return _compile_attention.cache_info()


def clear_cache():
    _compile_attention.cache_clear()
