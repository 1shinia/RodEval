"""Model Launcher — auto-detect, start, and stop local model inference backends."""

from __future__ import annotations

import os
import socket
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from evalscope.utils.logger import get_logger

logger = get_logger()


class ModelSource(str, Enum):
    OPENAI = 'openai'
    LOCAL = 'local'


class LocalBackend(str, Enum):
    AUTO = 'auto'
    VLLM = 'vllm'
    SGLANG = 'sglang'
    LLAMA_CPP = 'llama_cpp'
    TRANSFORMERS = 'transformers'


class ModelFormat(str, Enum):
    GGUF = 'gguf'
    CHECKPOINT = 'checkpoint'


@dataclass
class ModelInfo:
    path: str
    format: ModelFormat
    is_gguf: bool = False
    has_config: bool = False


@dataclass
class LaunchResult:
    backend: str
    eval_type: str
    api_url: Optional[str] = None
    api_key: str = 'not-needed'
    model_path: Optional[str] = None
    model_args: Dict[str, Any] = field(default_factory=dict)
    _server_process: Optional[subprocess.Popen] = field(default=None, repr=False)


# ── Detection ────────────────────────────────────────────────────


def detect_model(path: str) -> ModelInfo:
    path = os.path.expanduser(path)
    info = ModelInfo(path=path, format=ModelFormat.CHECKPOINT)
    if path.endswith('.gguf'):
        info.format = ModelFormat.GGUF
        info.is_gguf = True
        return info
    config_json = os.path.join(path, 'config.json')
    if os.path.isfile(config_json):
        info.has_config = True
    elif os.path.isdir(path):
        for entry in os.listdir(path):
            if entry.endswith('.gguf'):
                info.is_gguf = True
                info.format = ModelFormat.GGUF
                break
    return info


# ── GPU ───────────────────────────────────────────────────────────

_GPU_AVAILABLE: Optional[bool] = None


def has_gpu() -> bool:
    global _GPU_AVAILABLE
    if _GPU_AVAILABLE is None:
        try:
            import torch
            _GPU_AVAILABLE = torch.cuda.is_available() and torch.cuda.device_count() > 0
        except ImportError:
            _GPU_AVAILABLE = False
    return bool(_GPU_AVAILABLE)


# ── Backend resolution ────────────────────────────────────────────


def resolve_backend(info: ModelInfo, preference: str = 'auto') -> str:
    if preference and preference != 'auto':
        return preference.lower()
    if has_gpu():
        return 'vllm'
    if info.format == ModelFormat.GGUF:
        return 'llama_cpp'
    return 'transformers'


# ── Validation ────────────────────────────────────────────────────


def _validate_backend(resolved: str, info: ModelInfo) -> None:
    gpu = has_gpu()
    if resolved in ('vllm', 'sglang'):
        if info.format == ModelFormat.GGUF:
            raise ValueError(f'{resolved} 无法直接加载 GGUF。请选 auto 或 llama_cpp。\n模型: {info.path}')
        if not gpu:
            raise ValueError(f'{resolved} 需要 GPU，当前服务器无 GPU。请选 auto 或 llama_cpp 或 transformers。')
    if resolved == 'llama_cpp' and info.format == ModelFormat.CHECKPOINT:
        raise ValueError(f'llama.cpp 需要 GGUF，但检测到 HF checkpoint。请选 auto 或 transformers。')
    if resolved == 'transformers' and info.format == ModelFormat.GGUF:
        raise ValueError(f'Transformers 无法加载 GGUF。请选 auto 或 llama_cpp。')


# ── Port ──────────────────────────────────────────────────────────

_PORT_RANGE = range(19000, 19100)


def _find_free_port() -> int:
    for port in _PORT_RANGE:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('127.0.0.1', port))
                return port
            except OSError:
                continue
    raise RuntimeError(f'No free port in {_PORT_RANGE.start}-{_PORT_RANGE.stop - 1}')


def _health_check(url: str, timeout: float = 180.0) -> bool:
    import urllib.request
    deadline = time.time() + timeout
    health_url = url.rstrip('/') + '/models'
    while time.time() < deadline:
        try:
            resp = urllib.request.urlopen(urllib.request.Request(health_url), timeout=5)
            if resp.status == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


# ── Server launchers ──────────────────────────────────────────────


def _launch_vllm(model_path: str, port: int, extra_args=None) -> subprocess.Popen:
    extra_args = extra_args or {}
    cmd = [
        'python', '-m', 'vllm.entrypoints.openai.api_server', '--model', model_path, '--served-model-name',
        os.path.basename(model_path), '--host', '0.0.0.0', '--port',
        str(port), '--disable-log-requests', '--disable-log-stats'
    ]
    if extra_args.get('trust_remote_code'):
        cmd += ['--trust-remote-code']
    if extra_args.get('dtype') and extra_args['dtype'] != 'auto':
        cmd += ['--dtype', str(extra_args['dtype'])]
    if extra_args.get('tensor_parallel_size'):
        cmd += ['--tensor-parallel-size', str(extra_args['tensor_parallel_size'])]
    elif has_gpu():
        import torch
        cmd += ['--tensor-parallel-size', str(torch.cuda.device_count())]
    if extra_args.get('max_model_len'):
        cmd += ['--max-model-len', str(extra_args['max_model_len'])]
    if extra_args.get('gpu_memory_utilization'):
        cmd += ['--gpu-memory-utilization', str(extra_args['gpu_memory_utilization'])]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _launch_sglang(model_path: str, port: int, extra_args=None) -> subprocess.Popen:
    extra_args = extra_args or {}
    cmd = ['python', '-m', 'sglang.launch_server', '--model-path', model_path, '--host', '0.0.0.0', '--port', str(port)]
    if extra_args.get('trust_remote_code'):
        cmd += ['--trust-remote-code']
    if extra_args.get('tp_size'):
        cmd += ['--tp-size', str(extra_args['tp_size'])]
    if extra_args.get('mem_fraction_static'):
        cmd += ['--mem-fraction-static', str(extra_args['mem_fraction_static'])]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _launch_llama_cpp(model_path: str, port: int, extra_args=None) -> subprocess.Popen:
    extra_args = extra_args or {}
    cmd = ['python', '-m', 'llama_cpp.server', '--model', model_path, '--host', '0.0.0.0', '--port', str(port)]
    if extra_args.get('n_ctx'):
        cmd += ['--n_ctx', str(extra_args['n_ctx'])]
    if extra_args.get('n_threads'):
        cmd += ['--n_threads', str(extra_args['n_threads'])]
    if extra_args.get('n_gpu_layers'):
        cmd += ['--n_gpu_layers', str(extra_args['n_gpu_layers'])]
    if extra_args.get('n_batch'):
        cmd += ['--n_batch', str(extra_args['n_batch'])]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


_SERVER_LAUNCHERS = {'vllm': _launch_vllm, 'sglang': _launch_sglang, 'llama_cpp': _launch_llama_cpp}

# Eval types that must NOT run in a subprocess (C extensions crash on spawn)
_DIRECT_EVAL_TYPES = {'llm_ckpt'}

# ── Main API ──────────────────────────────────────────────────────


def launch(model_path: str, backend: str = 'auto', backend_args: Optional[Dict[str, Any]] = None) -> LaunchResult:
    backend_args = backend_args or {}
    info = detect_model(model_path)
    resolved = resolve_backend(info, backend)
    logger.info(
        f'[ModelLauncher] model={model_path} format={info.format.value} '
        f'backend={resolved} gpu={has_gpu()}'
    )
    _validate_backend(resolved, info)

    # Server mode
    if resolved in _SERVER_LAUNCHERS:
        port = _find_free_port()
        api_url = f'http://127.0.0.1:{port}/v1'
        proc = _SERVER_LAUNCHERS[resolved](model_path, port, backend_args)
        if not _health_check(api_url):
            proc.terminate()
            proc.wait(timeout=10)
            raise RuntimeError(f'{resolved} server unhealthy after 180s (model={model_path})')
        return LaunchResult(
            backend=resolved, eval_type='openai_api', api_url=api_url, model_path=model_path, _server_process=proc
        )

    # Direct mode (for backends that load model directly in-process)
    if resolved == 'transformers':
        return LaunchResult(
            backend='transformers',
            eval_type='llm_ckpt',
            model_path=model_path,
            model_args={
                'model_path': model_path,
                'revision': backend_args.get('revision', 'master'),
                'precision': backend_args.get('precision', 'torch.float16'),
                **({
                    'device_map': backend_args['device_map']
                } if 'device_map' in backend_args else {})
            }
        )

    raise ValueError(f'Unknown backend: {resolved}')


def stop(result: LaunchResult) -> None:
    proc = result._server_process
    if proc is None or proc.poll() is not None:
        return
    logger.info(f'[ModelLauncher] Stopping server (pid={proc.pid})')
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


@contextmanager
def launch_context(model_path: str, backend: str = 'auto', backend_args: Optional[Dict[str, Any]] = None):
    result = launch(model_path, backend=backend, backend_args=backend_args)
    try:
        yield result
    finally:
        stop(result)


def is_direct_eval_type(eval_type: str) -> bool:
    return eval_type in _DIRECT_EVAL_TYPES
