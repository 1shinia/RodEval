import time
from typing import Any, Dict, List, Optional

from evalscope.api.messages import ChatMessage, ChatMessageAssistant
from evalscope.api.messages.perf_metrics import PerformanceMetrics
from evalscope.api.model import ChatCompletionChoice, GenerateConfig, ModelAPI, ModelOutput, ModelUsage
from evalscope.api.tool import ToolChoice, ToolInfo
from evalscope.utils.logger import get_logger

logger = get_logger()


class LlamaCppAPI(ModelAPI):
    """Model API for local GGUF models via llama-cpp-python.

    Loads quantized GGUF model files directly from disk and runs CPU/CUDA
    inference without requiring a separate server process.
    """

    def __init__(
        self,
        model_name: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        config: GenerateConfig = GenerateConfig(),
        **model_args: Any,
    ) -> None:
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            config=config,
        )

        from llama_cpp import Llama

        # Resolve model path from model_args (same keys as checkpoint mode)
        model_path = model_args.get('model_path') or model_name
        n_ctx = model_args.get('n_ctx', 2048)
        n_threads = model_args.get('n_threads')
        n_gpu_layers = model_args.get('n_gpu_layers', 0)  # 0 = CPU only

        # Detect device map from model_args
        device_map = model_args.get('device_map', 'auto')
        if device_map == 'cpu':
            n_gpu_layers = 0

        logger.info(f'[LlamaCpp] Loading GGUF model: {model_path} (n_ctx={n_ctx}, n_gpu_layers={n_gpu_layers})')

        self._llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )

        self._model_path = model_path
        self._chat_template = model_args.get('chat_template')

        logger.info(f'[LlamaCpp] Model loaded successfully: {model_path}')

    def _messages_to_llama(self, input: List[ChatMessage]) -> List[Dict[str, str]]:
        """Convert ChatMessage list to llama.cpp dict format."""
        result = []
        for msg in input:
            if msg.role in ('system', 'user', 'assistant'):
                result.append({'role': msg.role, 'content': msg.text})
        return result

    def generate(
        self,
        input: List[ChatMessage],
        tools: Optional[List[ToolInfo]] = None,
        tool_choice: Optional[ToolChoice] = None,
        config: Optional[GenerateConfig] = None,
    ) -> ModelOutput:
        t0 = time.time()
        generate_config = config or self.config

        messages = self._messages_to_llama(input)
        max_tokens = generate_config.max_tokens if generate_config else 512
        temperature = generate_config.temperature if generate_config else 0.0

        try:
            resp = self._llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=generate_config.top_p if generate_config else None,
                top_k=generate_config.top_k if generate_config else None,
            )
            content = resp['choices'][0]['message']['content']
            input_tokens = resp.get('usage', {}).get('prompt_tokens', 0)
            output_tokens = resp.get('usage', {}).get('completion_tokens', 0)
        except Exception as e:
            logger.error(f'[LlamaCpp] Generation failed: {e}')
            return ModelOutput.from_content(
                model=self.model_name,
                content='',
                stop_reason='unknown',
                error=str(e),
            )

        elapsed = time.time() - t0
        usage = ModelUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )

        return ModelOutput(
            model=self.model_name,
            choices=[
                ChatCompletionChoice(
                    message=ChatMessageAssistant(
                        content=content,
                        model=self.model_name,
                        source='generate',
                        perf_metrics=PerformanceMetrics(
                            latency=elapsed,
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                        ),
                    ),
                    stop_reason='stop',
                )
            ],
            usage=usage,
            time=elapsed,
        )
