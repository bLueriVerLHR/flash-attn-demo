"""Evidence locations and provenance; imports only the Python standard library."""
from datetime import datetime, timezone
import getpass
import hashlib
import json
import os
from pathlib import Path
import stat
import uuid

PROJECT = Path(__file__).resolve().parents[2]


def configure_runtime_paths():
    root = Path(os.environ.get('FLASH_ATTN_ARTIFACT_ROOT', f'/tmp/flash-attn-demo-{getpass.getuser()}'))
    if root.exists():
        info = root.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
            raise RuntimeError(f'Refusing unsafe artifact directory: {root}')
    else:
        root.mkdir(parents=True, mode=0o700)
    paths = {'TORCH_EXTENSIONS_DIR': 'cache/torch_extensions', 'CUTE_DSL_CACHE_DIR': 'cache/cute',
             'CUDA_CACHE_PATH': 'cache/cuda', 'CUTE_DSL_DUMP_DIR': 'compiler_artifacts',
             'XDG_CACHE_HOME': 'cache', 'HF_HOME': 'cache/huggingface', 'TMPDIR': 'cache/tmp'}
    for name, relative in paths.items():
        os.environ.setdefault(name, str(root/relative))
        Path(os.environ[name]).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MAX_JOBS', '2')
    os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
    return root


def new_run_directory(label):
    root = configure_runtime_paths()
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    path = root/'runs'/f'{timestamp}-{label}-{uuid.uuid4().hex[:6]}'
    path.mkdir(parents=True, mode=0o700)
    return path


def source_fingerprint():
    files = sorted((PROJECT/'lib/cute_attn').glob('*.py'))
    hashes = {str(p.relative_to(PROJECT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    bundle = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return {'bundle_sha256': bundle, 'files': hashes}


def write_json(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2, allow_nan=False) + '\n')
