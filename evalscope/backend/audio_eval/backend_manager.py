"""Audio evaluation backend manager."""
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Union

from evalscope.backend.base import BackendManager
from evalscope.utils.logger import get_logger
from .arguments import AudioToolConfig

logger = get_logger()


class AudioBackendManager(BackendManager):
    """Backend manager for Audio evaluation tasks (ASR/TTS)."""

    def __init__(self, config: Union[str, dict], **kwargs):
        super().__init__(config, **kwargs)
        self._audio_config: AudioToolConfig | None = None

    def run(self, *args, **kwargs) -> Dict[str, Any]:
        """Run Audio evaluation pipeline."""
        config = self._parse_config()
        tool = config.tool

        logger.info(f'Starting Audio evaluation: {tool}')
        logger.info(f'Model: {config.model.model_name_or_path}')
        logger.info(f'Metrics: {config.eval.metrics}')

        if tool == 'asr':
            return self._run_asr(config)
        elif tool == 'tts':
            return self._run_tts(config)
        else:
            raise ValueError(f'Unsupported Audio tool: {tool}')

    def _parse_config(self) -> AudioToolConfig:
        """Parse configuration into AudioToolConfig."""
        if self._audio_config is not None:
            return self._audio_config

        if isinstance(self.config_d, AudioToolConfig):
            self._audio_config = self.config_d
        elif isinstance(self.config_d, dict):
            self._audio_config = AudioToolConfig(**self.config_d)
        else:
            raise ValueError(f'Invalid config type: {type(self.config_d)}')

        return self._audio_config

    def _ensure_output_dir(self, output_dir_str: str) -> Path:
        """Ensure output directory exists."""
        if not output_dir_str:
            raise ValueError(
                'output_dir is not set in eval config. '
                'This is required for saving audio files and results.'
            )
        output_dir = Path(output_dir_str)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _create_asr_model(self, model_config: dict):
        """Factory: create ASR model based on provider."""
        provider = model_config.get('provider', 'openai')
        if provider == 'dashscope':
            from .models.asr_dashscope import AsrModelDashScope
            return AsrModelDashScope(model_config)
        elif provider == 'openai':
            from .models.asr_openai import AsrModel
            return AsrModel(model_config)
        else:
            raise ValueError(f'Unsupported ASR provider: {provider}')

    def _create_tts_model(self, model_config: dict):
        """Factory: create TTS model based on provider."""
        provider = model_config.get('provider', 'openai')
        if provider == 'dashscope':
            from .models.tts_dashscope import TtsModelDashScope
            return TtsModelDashScope(model_config)
        elif provider == 'openai':
            from .models.tts_openai import TtsModel
            return TtsModel(model_config)
        else:
            raise ValueError(f'Unsupported TTS provider: {provider}')

    # ── ASR Pipeline ──────────────────────────────────────────────

    def _run_asr(self, config: AudioToolConfig) -> Dict[str, Any]:
        """Run ASR evaluation pipeline."""
        from .datasets.prompt_loader import load_asr_samples
        from .metrics.wer import compute_cer, compute_wer

        output_dir = self._ensure_output_dir(config.eval.output_dir)
        provider = config.model.provider

        # All providers: decode base64 → save temp file → pass path to model
        audio_base64 = getattr(config.generate, 'audio_base64', None)
        audio_path = getattr(config.generate, 'audio_path', None)

        if not audio_base64 and not audio_path:
            raise ValueError('audio_base64 or audio_path is required for ASR')

        audio_dir = output_dir / 'audio'
        audio_dir.mkdir(exist_ok=True)

        if audio_base64:
            import base64
            b64_data = audio_base64
            if ',' in b64_data:
                b64_data = b64_data.split(',', 1)[1]
            audio_bytes = base64.b64decode(b64_data)
            audio_filename = getattr(config.generate, 'audio_filename', None) or 'input.wav'
            audio_path = str(audio_dir / audio_filename)
            with open(audio_path, 'wb') as f:
                f.write(audio_bytes)
            logger.info(f'Decoded base64 audio → {audio_path} ({len(audio_bytes)} bytes)')

        if not audio_path:
            raise ValueError('audio_path or audio_base64 is required for ASR evaluation')

        logger.info(f'ASR audio input: {audio_path[:80]}...')

        # Load ASR model
        model_config = config.model.dict()
        model = self._create_asr_model(model_config)

        start_time = time.time()
        result = model.generate(
            audio_path, language=config.model.language
        )
        elapsed = time.time() - start_time

        hypothesis = result['text']
        reference = config.generate.reference_text or ''

        # Compute metrics
        metrics_result: Dict[str, float] = {}
        if 'wer' in config.eval.metrics:
            try:
                metrics_result['wer'] = compute_wer(reference, hypothesis)
            except Exception as e:
                logger.warning(f'WER computation failed: {e}')
                metrics_result['wer'] = -1.0

        if 'cer' in config.eval.metrics:
            try:
                metrics_result['cer'] = compute_cer(reference, hypothesis)
            except Exception as e:
                logger.warning(f'CER computation failed: {e}')
                metrics_result['cer'] = -1.0

        # Build results
        per_sample = {
            'audio_path': str(audio_path),
            'reference': reference,
            'hypothesis': hypothesis,
            'elapsed_seconds': round(elapsed, 2),
            'language': config.model.language,
        }
        per_sample.update(metrics_result)

        results = {
            'tool': 'asr',
            'model': config.model.model_name_or_path,
            'metrics': metrics_result,
            'per_sample': per_sample,
            'elapsed_seconds': round(elapsed, 2),
        }

        # Save to results.json
        results_path = output_dir / 'results.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f'ASR results saved to {results_path}')

        return results

    # ── TTS Pipeline ──────────────────────────────────────────────

    def _run_tts(self, config: AudioToolConfig) -> Dict[str, Any]:
        """Run TTS evaluation pipeline (generation only, no metrics yet)."""
        from .datasets.prompt_loader import load_tts_prompts
        from .utils.audio import save_audio

        output_dir = self._ensure_output_dir(config.eval.output_dir)
        audio_dir = output_dir / 'audio'
        audio_dir.mkdir(exist_ok=True)

        # Load prompts
        custom_prompt = getattr(config.eval, 'custom_prompt', None)
        if custom_prompt:
            prompts = [custom_prompt.strip()]
        else:
            prompts = load_tts_prompts(
                config.eval.prompt_dataset,
                limit=config.eval.prompt_limit,
                custom_path=config.eval.custom_dataset_path,
            )

        logger.info(f'TTS prompts loaded: {len(prompts)}')

        # Load TTS model
        model_config = config.model.dict()
        model = self._create_tts_model(model_config)

        per_sample_results = []
        total_elapsed = 0.0

        for i, prompt in enumerate(prompts):
            logger.info(f'TTS sample {i+1}/{len(prompts)}: {prompt[:60]}...')
            start_time = time.time()
            try:
                audio_bytes = model.generate(
                    prompt,
                    voice=config.generate.voice,
                    response_format=config.generate.response_format,
                    speed=config.generate.speed,
                )
                elapsed = time.time() - start_time
                total_elapsed += elapsed

                audio_filename = f'sample_{i:04d}.{config.generate.response_format}'
                audio_path = save_audio(audio_bytes, audio_dir / audio_filename,
                                        fmt=config.generate.response_format)

                per_sample_results.append({
                    'index': i,
                    'prompt': prompt,
                    'audio_path': str(audio_path),
                    'elapsed_seconds': round(elapsed, 2),
                })
            except Exception as e:
                logger.error(f'TTS sample {i} failed: {e}')
                per_sample_results.append({
                    'index': i,
                    'prompt': prompt,
                    'error': str(e),
                })

        results = {
            'tool': 'tts',
            'model': config.model.model_name_or_path,
            'num_samples': len(prompts),
            'total_elapsed_seconds': round(total_elapsed, 2),
            'per_sample': per_sample_results,
        }

        results_path = output_dir / 'results.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f'TTS results saved to {results_path}')

        return results
