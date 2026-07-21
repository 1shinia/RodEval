"""AIGC evaluation backend manager."""
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Union

from evalscope.backend.base import BackendManager
from evalscope.utils.logger import get_logger
from .arguments import AIGCToolConfig

logger = get_logger()


class AIGCBackendManager(BackendManager):
    """Backend manager for AIGC evaluation tasks."""

    def __init__(self, config: Union[str, dict], **kwargs):
        super().__init__(config, **kwargs)
        self._aigc_config: AIGCToolConfig | None = None

    def run(self, *args, **kwargs) -> Dict[str, Any]:
        """Run AIGC evaluation pipeline."""
        config = self._parse_config()
        tool = config.tool

        logger.info(f'Starting AIGC evaluation: {tool}')
        logger.info(f'Model: {config.model.model_name_or_path}')
        logger.info(f'Metrics: {config.eval.metrics}')

        if tool == 'txt2img':
            return self._run_txt2img(config)
        elif tool == 'txt2video':
            return self._run_txt2video(config)
        elif tool == 'img2img':
            return self._run_img2img(config)
        else:
            raise ValueError(f'Unsupported AIGC tool: {tool}')

    def _parse_config(self) -> AIGCToolConfig:
        """Parse configuration into AIGCToolConfig."""
        if self._aigc_config is not None:
            return self._aigc_config

        if isinstance(self.config_d, AIGCToolConfig):
            self._aigc_config = self.config_d
        elif isinstance(self.config_d, dict):
            self._aigc_config = AIGCToolConfig(**self.config_d)
        else:
            raise ValueError(f'Invalid config type: {type(self.config_d)}')

        return self._aigc_config

    def _run_txt2img(self, config: AIGCToolConfig) -> Dict[str, Any]:
        """Run text-to-image evaluation."""
        from .datasets.prompt_loader import load_prompts
        from .metrics.clip_score import compute_clip_score
        from .models.txt2img import Txt2ImgModel
        from .utils.media import create_thumbnails, save_images
        from .utils.progress import ProgressTracker

        # Setup output directory
        output_dir_str = config.eval.output_dir or ''
        if not output_dir_str:
            raise ValueError(
                'output_dir is not set in eval config. '
                'This is required for saving generated images and results.'
            )
        output_dir = Path(output_dir_str)
        output_dir.mkdir(parents=True, exist_ok=True)
        images_dir = output_dir / 'images'
        images_dir.mkdir(exist_ok=True)
        thumbs_dir = output_dir / 'thumbnails'
        thumbs_dir.mkdir(exist_ok=True)

        # Load prompts
        logger.info(f'Loading prompts from: {config.eval.prompt_dataset}')
        prompts = load_prompts(
            config.eval.prompt_dataset,
            limit=config.eval.prompt_limit,
        )
        logger.info(f'Loaded {len(prompts)} prompts')

        # Initialize progress tracker
        progress = ProgressTracker(output_dir / 'progress.json', total=len(prompts))

        # Load model
        model = Txt2ImgModel(config.model.dict())
        model.load()

        try:
            # Generate images
            logger.info('Generating images...')
            start_time = time.time()

            images = []
            per_sample_results = []

            for i, prompt in enumerate(prompts):
                img = model.generate(
                    [prompt],
                    width=config.generate.width,
                    height=config.generate.height,
                    num_inference_steps=config.generate.num_inference_steps,
                    guidance_scale=config.generate.guidance_scale,
                    negative_prompt=config.generate.negative_prompt,
                    seed=config.generate.seed + i,
                )[0]

                images.append(img)

                # Save image
                img_path = images_dir / f'{i:04d}.png'
                img.save(img_path)

                # Create thumbnail
                thumb_path = thumbs_dir / f'{i:04d}_thumb.jpg'
                create_thumbnails([img], [thumb_path])

                # Compute per-sample metrics
                sample_result = {
                    'index': i,
                    'prompt': prompt,
                    'image_path': str(img_path.relative_to(output_dir)),
                    'thumbnail_path': str(thumb_path.relative_to(output_dir)),
                }

                if 'clip_score' in config.eval.metrics:
                    score = compute_clip_score([img], [prompt])
                    sample_result['clip_score'] = score[0]

                per_sample_results.append(sample_result)
                progress.update(i + 1)

            elapsed = time.time() - start_time
            logger.info(f'Generated {len(images)} images in {elapsed:.1f}s')

            # Compute aggregate metrics
            metrics = {}
            if 'clip_score' in config.eval.metrics:
                scores = [r['clip_score'] for r in per_sample_results]
                metrics['clip_score_mean'] = sum(scores) / len(scores)
                metrics['clip_score_min'] = min(scores)
                metrics['clip_score_max'] = max(scores)
                logger.info(f'CLIP Score: mean={metrics["clip_score_mean"]:.4f}')

            # Save results
            results = {
                'model': config.model.model_name_or_path,
                'model_type': 'txt2img',
                'num_samples': len(prompts),
                'metrics': metrics,
                'generation_time': elapsed,
                'per_sample': per_sample_results,
            }

            results_path = output_dir / 'results.json'
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)

            logger.info(f'Results saved to: {results_path}')
            progress.complete()

            return results

        finally:
            model.unload()

    def _run_txt2video(self, config: AIGCToolConfig) -> Dict[str, Any]:
        """Run text-to-video evaluation (placeholder)."""
        raise NotImplementedError('Text-to-video evaluation not yet implemented')

    def _run_img2img(self, config: AIGCToolConfig) -> Dict[str, Any]:
        """Run image-to-image evaluation (placeholder)."""
        raise NotImplementedError('Image-to-image evaluation not yet implemented')
