import functools
import logging
import sys
from typing import Any, Callable, Optional

import torch


def _report(sink: Any, label: str, stats: dict) -> None:
    if sink is None:
        return
    if isinstance(sink, list):
        sink.append((label, stats))
        return
    if callable(sink):
        sink(label, stats)
        return

    msg = (
        f"[{label}] "
        f"mean={stats['mean']:.4f}ms  "
        f"median={stats['median']:.4f}ms  "
        f"std={stats['std']:.4f}ms  "
        f"min={stats['min']:.4f}ms  "
        f"max={stats['max']:.4f}ms"
    )

    if sink == "stdout":
        print(msg, file=sys.stdout)
    elif sink == "stderr":
        print(msg, file=sys.stderr)
    elif sink == "logger":
        logging.getLogger("cuda_timer").info(msg)
    elif hasattr(sink, "write"):
        sink.write(msg + "\n")
    else:
        raise ValueError(f"Unsupported sink: {sink!r}")


def cuda_timer(
    name: Optional[str] = None,
    warmup: int = 10,
    rep: int = 20,
    *,
    sink: Any = "stdout",
    nvtx: bool = False,
):
    """
    用 CUDA Events 计时，**返回被装饰函数的真实返回值**。

    参数
    ----
    name    : 上报用的名字，默认用 func.__name__。
    warmup  : 预热次数（不计时）。
    rep     : 计时轮数。
    sink    : 计时结果去向。支持
              None | 'stdout' | 'stderr' | 'logger'
              | 可写文件对象 | list | callable(label, stats)。
    nvtx    : 每轮 rep 外推一个同名 NVTX range，
              配合 `ncu --nvtx --nvtx-include "<label>/"` 可做 kernel 级 profiling。

    用法
    ----
    >>> import torch
    >>> records = []
    >>> @cuda_timer(name="my_op", sink=records, nvtx=True)
    ... def my_op(x):
    ...     return x * 2
    >>> y = my_op(torch.randn(1024, device="cuda"))   # y 是真正的返回值
    >>> # ncu 侧: ncu --nvtx --nvtx-include "my_op/" \
    >>> #           --launch-count 5 -o rep python train.py
    """
    if rep < 1:
        raise ValueError("rep must be >= 1")

    def decorator(func: Callable):
        label = name or func.__name__
        last_stats: list = [None]

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # warmup
            result = None
            for _ in range(warmup):
                result = func(*args, **kwargs)
            torch.cuda.synchronize()

            starts = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
            ends   = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]

            # timed
            for i in range(rep):
                if nvtx:
                    torch.cuda.nvtx.range_push(label)
                starts[i].record()
                result = func(*args, **kwargs)
                ends[i].record()
                if nvtx:
                    torch.cuda.nvtx.range_pop()

            torch.cuda.synchronize()

            times = torch.tensor(
                [s.elapsed_time(e) for s, e in zip(starts, ends)]
            )
            stats = {
                "name":   label,
                "mean":   times.mean().item(),
                "median": times.median().item(),
                "std":    times.std().item() if rep > 1 else 0.0,
                "min":    times.min().item(),
                "max":    times.max().item(),
                "times":  times,
            }
            last_stats[0] = stats

            _report(sink, label, stats)

            return result

        wrapper.last_stats = lambda: last_stats[0]
        return wrapper

    return decorator