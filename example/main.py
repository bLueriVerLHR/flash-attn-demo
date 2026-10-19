"""Compare attention backends; CUDA imports happen only after CLI validation."""
from enum import Enum
from pathlib import Path
from typing import Annotated
import typer
from lib.utils.benchmark import BenchmarkShape, TimingConfig
from lib.utils.artifacts import configure_runtime_paths, new_run_directory


class Preset(str, Enum):
    tiny = 'tiny'
    demo = 'demo'
    gqa = 'gqa'


class Backend(str, Enum):
    cute_ffi = 'cute-ffi'
    cute_dlpack = 'cute-dlpack'
    prepared_ffi = 'prepared-ffi'
    prepared_dlpack = 'prepared-dlpack'
    cuda2 = 'cuda2'
    flash = 'flash'
    torch = 'torch'
    legacy = 'legacy'


class Mode(str, Enum):
    events = 'events'
    graph = 'graph'
    wall = 'wall'
    enqueue = 'enqueue'


PRESETS = {
    Preset.tiny: ['1,128,1,128'],
    Preset.demo: ['1,128,1,128', '4,4096,32,128', '8,2048,32,128'],
    Preset.gqa: ['1,192,4,128,2', '1,256,8,64,1'],
}


def main(
    preset: Preset = Preset.demo,
    shape: Annotated[list[str] | None, typer.Option(help='Repeatable B,L,H,D[,HK]; overrides preset.')] = None,
    backend: Annotated[list[Backend] | None, typer.Option(help='Repeat to compare backends.')] = None,
    mode: Annotated[list[Mode] | None, typer.Option(help='Repeat to select timing modes.')] = None,
    warmup: Annotated[int, typer.Option(min=1)] = 5,
    repeat: Annotated[int, typer.Option(min=2)] = 15,
    graph_calls: Annotated[int, typer.Option(min=1)] = 5,
    seed: int = 42,
    device: Annotated[str, typer.Option(help='Logical device after CUDA_VISIBLE_DEVICES.')] = 'cuda:0',
    output: Annotated[Path | None, typer.Option(help='JSON path; defaults to a unique /tmp run.')] = None,
):
    """Correctness-first CuTe / CUDA Kernel 2 attention comparison."""
    try:
        shapes = [BenchmarkShape.parse(text) for text in (shape or PRESETS[preset])]
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint='--shape') from error
    backends = [value.value for value in (backend or [Backend.cute_ffi, Backend.cuda2, Backend.flash])]
    modes = [value.value for value in (mode or [Mode.events, Mode.graph, Mode.wall])]
    if 'cuda2' in backends and any(item.head_dim != 128 for item in shapes):
        raise typer.BadParameter('CUDA Kernel 2 requires D=128; omit --backend cuda2 for D=64.', param_hint='--backend')

    configure_runtime_paths()
    output = output or new_run_directory('benchmark') / 'results.json'
    from lib.utils.benchmark import run_suite
    run_suite(shapes, backends, TimingConfig(warmup, repeat, graph_calls), modes, output, seed, device)
    typer.echo(f'\nEvidence: {output}')


if __name__ == '__main__':
    typer.run(main)
