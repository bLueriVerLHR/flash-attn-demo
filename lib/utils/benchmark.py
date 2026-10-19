"""Readable benchmark building blocks, with three distinct timing definitions.

No GPU imports at module scope: `python -m example.main --help` stays lightweight.
"""
from dataclasses import dataclass, asdict
import importlib.util
from pathlib import Path
import statistics
import sys
import time
from .artifacts import PROJECT, configure_runtime_paths, source_fingerprint, write_json


@dataclass(frozen=True)
class BenchmarkShape:
    batch: int
    length: int
    heads: int
    head_dim: int
    kv_heads: int

    @classmethod
    def parse(cls, text):
        values = tuple(map(int, text.replace('x', ',').split(',')))
        if len(values) not in (4, 5) or min(values) < 1:
            raise ValueError('shape must be B,L,H,D[,HK], all positive')
        b,l,h,d = values[:4]
        return cls(b,l,h,d,values[4] if len(values)==5 else h)

    @property
    def label(self):
        return f'{self.batch},{self.length},{self.heads},{self.head_dim},{self.kv_heads}'


@dataclass(frozen=True)
class TimingConfig:
    warmup: int = 5
    repeats: int = 15
    graph_calls: int = 5

    def __post_init__(self):
        if self.warmup < 1 or self.repeats < 2 or self.graph_calls < 1:
            raise ValueError('warmup>=1, repeats>=2, graph_calls>=1 are required')


def _stats(samples):
    return {'median_ms':statistics.median(samples), 'min_ms':min(samples),
            'max_ms':max(samples), 'samples_ms':samples}


def measure(fn, config=TimingConfig(), mode='events'):
    import torch
    import cuda.bindings.driver as cuda
    for _ in range(config.warmup):
        fn()
    torch.cuda.synchronize()
    calls, nodes = 1, None
    if mode == 'graph':
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph):
            for _ in range(config.graph_calls):
                fn()
        status, _, nodes = cuda.cuGraphGetNodes(cuda.CUgraph(graph.raw_cuda_graph()))
        if status != cuda.CUresult.CUDA_SUCCESS or nodes < config.graph_calls:
            raise RuntimeError(f'Invalid/empty CUDA graph: status={status}, nodes={nodes}')
        fn, calls = graph.replay, config.graph_calls
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
    if mode in ('events','graph'):
        starts=[torch.cuda.Event(enable_timing=True) for _ in range(config.repeats)]
        ends=[torch.cuda.Event(enable_timing=True) for _ in range(config.repeats)]
        for start,end in zip(starts,ends):
            start.record(); fn(); end.record()
        torch.cuda.synchronize()
        result=_stats([s.elapsed_time(e)/calls for s,e in zip(starts,ends)])
    elif mode == 'wall':
        samples=[]
        for _ in range(config.repeats):
            torch.cuda.synchronize()
            start=time.perf_counter(); fn(); torch.cuda.synchronize()
            samples.append((time.perf_counter()-start)*1000)
        result=_stats(samples)
    elif mode == 'enqueue':
        samples=[]
        for _ in range(config.repeats):
            start=time.perf_counter(); fn()
            samples.append((time.perf_counter()-start)*1000)
        torch.cuda.synchronize()
        result=_stats(samples)
    else:
        raise ValueError('mode must be events, graph, wall or enqueue')
    if nodes is not None:
        result['graph_nodes']=nodes
    return result


def reference_attention(q,k,v):
    import torch
    import torch.nn.functional as F
    # Small cases use independent FP32 math. Large cases use fused FP16 SDPA
    # to avoid materializing an enormous score matrix on a shared 24GB GPU.
    if q.shape[1] <= 256:
        qf,kf,vf=[x.float().transpose(1,2) for x in (q,k,v)]
        ratio=qf.shape[1]//kf.shape[1]
        if ratio!=1:
            kf,vf=kf.repeat_interleave(ratio,1),vf.repeat_interleave(ratio,1)
        return ((qf@kf.transpose(-1,-2)/q.shape[-1]**0.5).softmax(-1)@vf).transpose(1,2)
    return F.scaled_dot_product_attention(
        q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),
        enable_gqa=q.shape[2]!=k.shape[2],
    ).transpose(1,2).float()


def correctness(out,ref,atol=2e-3,rtol=2e-3):
    import torch
    finite=bool(torch.isfinite(out).all())
    delta=(out.float()-ref.float()).abs()
    result={'finite':finite,'max_abs':delta.max().item() if finite else None,
            'mean_abs':delta.mean().item() if finite else None,'atol':atol,'rtol':rtol}
    try:
        torch.testing.assert_close(out.float(),ref.float(),atol=atol,rtol=rtol)
        result['pass']=True
    except AssertionError as error:
        result.update({'pass':False,'error':str(error)})
    return result


def _legacy_module():
    root=configure_runtime_paths()
    path=root/'history/pre-refactor-v0.2/lib/cute_attn/kernel.py'
    if not path.exists():
        raise FileNotFoundError(f'Optional v0.2 snapshot unavailable: {path}')
    name='flash_attn_demo_v02_snapshot'
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,path)
        module=importlib.util.module_from_spec(spec)
        sys.modules[name]=module
        spec.loader.exec_module(module)
    return sys.modules[name]


def make_backend(name,q,k,v):
    import torch
    from lib.cute_attn import flash_attn_demo,prepare_attention
    if name.startswith('cute-'):
        backend=name.split('-',1)[1]
        return lambda:flash_attn_demo(q,k,v,backend=backend)
    if name.startswith('prepared-'):
        out=torch.empty_like(q)
        prepared=prepare_attention(q,k,v,out=out,backend=name.split('-',1)[1])
        return lambda:prepared.launch(q,k,v,out)
    if name=='cuda2':
        if q.shape[-1]!=128:
            raise ValueError('Unchanged C++ Kernel 2 is fixed to D=128')
        from lib.cpp_extensions.runtime import load_kernel2
        extension=load_kernel2()
        return lambda:extension.flash_attn_demo_cu(q,k,v)
    if name=='flash':
        import flash_attn
        return lambda:flash_attn.flash_attn_func(q,k,v,causal=False)
    if name=='torch':
        import torch.nn.functional as F
        return lambda:F.scaled_dot_product_attention(
            q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),
            enable_gqa=q.shape[2]!=k.shape[2],
        ).transpose(1,2)
    if name=='legacy':
        module=_legacy_module()
        return lambda:module.flash_attn_demo(q,k,v)
    raise ValueError(f'Unknown backend: {name}')


def run_suite(shapes,backends,timing,modes,output,seed=42,device='cuda:0'):
    import torch
    configure_runtime_paths()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; activate the mlc environment on gpunode1')
    torch.cuda.set_device(device)
    torch.manual_seed(seed)
    import importlib.metadata
    packages={}
    for name in ('nvidia-cutlass-dsl','apache-tvm-ffi','flash-attn','transformers'):
        try:
            packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name]=None
    report={'source':source_fingerprint(),'device':torch.cuda.get_device_name(),
            'device_capability':torch.cuda.get_device_capability(),'packages':packages,
            'visible_devices':__import__('os').environ.get('CUDA_VISIBLE_DEVICES'),
            'torch':torch.__version__,'cuda':torch.version.cuda,'seed':seed,
            'timing':asdict(timing),'modes':modes,'semantics':'fp16, non-causal, no mask/dropout',
            'timing_notes':{'events':'May contain host-induced gaps',
              'graph':'Time per captured call; node count verified',
              'wall':'Synchronized end-to-end, includes completion sync',
              'enqueue':'CPU time to enqueue; excludes waiting for GPU completion'},
            'results':[]}
    with torch.inference_mode():
        for shape in shapes:
            print(f'\nShape (B,L,H,D,HK)={shape.label}',flush=True)
            q=torch.randn(shape.batch,shape.length,shape.heads,shape.head_dim,device=device,dtype=torch.float16)
            k=torch.randn(shape.batch,shape.length,shape.kv_heads,shape.head_dim,device=device,dtype=torch.float16)
            v=torch.randn_like(k)
            ref=reference_attention(q,k,v)
            for name in backends:
                # Alternate requested backend order in subsequent CLI rounds.
                start=time.perf_counter()
                fn=make_backend(name,q,k,v)
                out=fn(); torch.cuda.synchronize()
                row={'shape':asdict(shape),'backend':name,
                     'output_mode':'reused buffer' if name.startswith('prepared-') else 'convenience result allocation',
                     'reference':'FP32 math' if shape.length<=256 else 'fused FP16 Torch SDPA, cast output to FP32',
                     'setup_seconds':time.perf_counter()-start,
                     'correctness':correctness(out,ref),'measurements':{}}
                report['results'].append(row)
                if not row['correctness']['pass']:
                    write_json(output,report)
                    raise AssertionError(f'{name} correctness failed; no performance claim made')
                for mode in modes:
                    row['measurements'][mode]=measure(fn,timing,mode)
                metrics=' '.join(f'{mode}={value["median_ms"]:.6f}ms' for mode,value in row['measurements'].items())
                print(f'  {name:16s} {metrics} max_abs={row["correctness"]["max_abs"]:.3g}',flush=True)
                write_json(output,report)
                del fn,out
            del q,k,v,ref
            torch.cuda.empty_cache()
    return report
