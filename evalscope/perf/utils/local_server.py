import os
import signal
import subprocess
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from evalscope.perf.arguments import Arguments
from evalscope.utils.chat_service import ChatCompletionRequest, ChatService, ModelList, TextCompletionRequest
from evalscope.utils.import_utils import check_import
from evalscope.utils.logger import get_logger

logger = get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def create_app(model, attn_implementation=None) -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    chat_service = ChatService(model_path=model, attn_implementation=attn_implementation)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=['*'],
        allow_credentials=True,
        allow_methods=['*'],
        allow_headers=['*'],
    )

    @app.get('/v1/models', response_model=ModelList)
    async def list_models():
        return await chat_service.list_models()

    @app.post('/v1/completions')
    async def create_text_completion(request: TextCompletionRequest):
        return await chat_service._text_completion(request)

    @app.post('/v1/chat/completions')
    async def create_chat_completion(request: ChatCompletionRequest):
        if request.stream:
            return EventSourceResponse(chat_service._stream_chat(request))
        else:
            return await chat_service._chat(request)

    return app


def start_app(args: Arguments):
    logger.info('Starting local server, please wait...')
    if args.api == 'local':
        if args.model.endswith('.gguf'):
            # GGUF → start llama.cpp server
            import sys
            check_import('llama_cpp', 'llama-cpp-python', raise_error=True)
            cmd = [
                sys.executable,
                '-m',
                'llama_cpp.server',
                '--model',
                args.model,
                '--n_ctx',
                '2048',
                '--n_threads',
                '8',
                '--host',
                '0.0.0.0',
                '--port',
                str(args.port),
            ]
            proc = subprocess.Popen(cmd, start_new_session=True)
            # Wait for server to be ready
            import requests
            import time
            health_url = f'http://127.0.0.1:{args.port}/health'
            for _ in range(60):
                try:
                    resp = requests.get(health_url, timeout=2)
                    if resp.status_code == 200:
                        logger.info('llama.cpp server is ready')
                        break
                except Exception:
                    pass
                time.sleep(2)
            else:
                logger.warning('llama.cpp server did not become ready within 120s')
            import atexit

            def on_exit():
                if proc.poll() is None:
                    logger.info('Terminating llama.cpp server...')
                    pgid = None
                    try:
                        pgid = os.getpgid(proc.pid)
                        os.killpg(pgid, signal.SIGTERM)
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        if pgid is not None:
                            try:
                                os.killpg(pgid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                        proc.wait()
                    except ProcessLookupError:
                        pass
                    logger.info('llama.cpp server terminated.')
                else:
                    logger.info('llama.cpp server has already terminated.')

            atexit.register(on_exit)
        else:
            # Transformers / HF checkpoint → start uvicorn as a subprocess
            # (same pattern as GGUF: subprocess + health check + atexit cleanup)
            import atexit as _atexit
            import socket
            import sys
            import time as _time

            check_import('torch', 'torch', raise_error=True)
            attn_arg = repr(args.attn_implementation)  # None → 'None'
            server_code = (
                'import uvicorn\n'
                'from evalscope.perf.utils.local_server import create_app\n'
                'import sys\n'
                "attn = None if sys.argv[2] == 'None' else sys.argv[2]\n"
                'app = create_app(sys.argv[1], attn)\n'
                "uvicorn.run(app, host='0.0.0.0', port=int(sys.argv[3]), workers=1)\n"
            )
            proc = subprocess.Popen([
                sys.executable,
                '-c',
                server_code,
                args.model,
                attn_arg,
                str(args.port),
            ],
                                    start_new_session=True)

            # Wait for the server to start accepting connections
            for _ in range(60):
                try:
                    sock = socket.create_connection(('127.0.0.1', args.port), timeout=2)
                    sock.close()
                    logger.info('Local transformers server is ready')
                    break
                except OSError:
                    pass
                _time.sleep(2)
            else:
                logger.warning('Local transformers server did not become ready within 120s')

            def _on_exit():
                if proc.poll() is None:
                    logger.info('Terminating transformers server...')
                    pgid = None
                    try:
                        pgid = os.getpgid(proc.pid)
                        os.killpg(pgid, signal.SIGTERM)
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        if pgid is not None:
                            try:
                                os.killpg(pgid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                        proc.wait()
                    except ProcessLookupError:
                        pass
                    logger.info('Transformers server terminated.')
                else:
                    logger.info('Transformers server has already terminated.')

            _atexit.register(_on_exit)

    elif args.api == 'local_vllm':
        import torch

        os.environ['VLLM_USE_MODELSCOPE'] = 'True'
        os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'
        os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
        # yapf: disable
        proc = subprocess.Popen([
            'python', '-m', 'vllm.entrypoints.openai.api_server',
            '--model', args.model,
            '--served-model-name', args.model,
            '--tensor-parallel-size', str(torch.cuda.device_count()),
            '--max-model-len', '32768',
            '--gpu-memory-utilization', '0.9',
            '--host', '0.0.0.0',
            '--port', str(args.port),
            '--trust-remote-code',
            '--disable-log-requests',
            '--disable-log-stats',
        ], start_new_session=True)
        # yapf: enable
        import atexit

        def on_exit():
            if proc.poll() is None:
                logger.info('Terminating the child process...')
                pgid = None
                try:
                    pgid = os.getpgid(proc.pid)
                    os.killpg(pgid, signal.SIGTERM)
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    logger.warning('Child process did not terminate within the timeout, killing it forcefully...')
                    if pgid is not None:
                        try:
                            os.killpg(pgid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    proc.wait()
                except ProcessLookupError:
                    pass
                logger.info('Child process terminated.')
            else:
                logger.info('Child process has already terminated.')

        atexit.register(on_exit)
    else:
        raise ValueError(f'Unknown API type: {args.api}')


if __name__ == '__main__':
    from collections import namedtuple

    args = namedtuple('Args', ['model', 'attn_implementation', 'api'])

    start_app(args(model='Qwen/Qwen2.5-0.5B-Instruct', attn_implementation=None, api='local_vllm'))
