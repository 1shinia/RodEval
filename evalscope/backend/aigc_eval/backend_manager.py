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

MAX_MODEL_NAME_LENGTH = 200


def _safe_model_name(name: str, tool: str = 'unknown') -> str:
    """Truncate model name if it exceeds the max length (defensive against log corruption)."""
    if isinstance(name, str) and len(name) > MAX_MODEL_NAME_LENGTH:
        logger.warning(
            f'Model name too long ({len(name)} chars) for {tool}, '
            f'truncating to {MAX_MODEL_NAME_LENGTH} chars'
        )
        return name[:MAX_MODEL_NAME_LENGTH]
    return name


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

        try:
            if tool == 'txt2img':
                return self._run_txt2img(config)
            elif tool == 'txt2video':
                return self._run_txt2video(config)
            elif tool == 'img2img':
                return self._run_img2img(config)
            raise ValueError(f'Unsupported AIGC tool: {tool}')
        except Exception as e:
            # 失败原因必须写进任务日志（页面 SSE 只展示任务日志），
            # 否则用户只能看到 progress.json 里的 "Task failed"。
            logger.error(f'AIGC {tool} 任务失败：{e}')
            raise

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

        # Load prompts — use custom_prompt if provided, otherwise load from dataset
        custom_prompt = getattr(config.eval, 'custom_prompt', None)
        if custom_prompt:
            prompts = [custom_prompt.strip()]
            logger.info(f'Using custom prompt for txt2img: {prompts[0][:80]}...')
        else:
            logger.info(f'Loading prompts from: {config.eval.prompt_dataset}')
            prompts = load_prompts(
                config.eval.prompt_dataset,
                limit=config.eval.prompt_limit,
                custom_path=config.eval.custom_dataset_path,
                shuffle=getattr(config.eval, 'random_prompt', False),
                seed=config.generate.seed,
            )
        logger.info(f'Loaded {len(prompts)} prompts')

        # Initialize progress tracker
        progress = ProgressTracker(output_dir / 'progress.json', total=len(prompts))

        # Load model
        model_config = config.model.dict()
        model_config['tool'] = 'txt2img'
        model = Txt2ImgModel(model_config)
        model.load()

        try:
            # Generate images
            logger.info('Generating images...')
            start_time = time.time()

            images = []
            per_sample_results = []

            for i, prompt in enumerate(prompts):
                generation_started = time.monotonic()
                img = model.generate(
                    [prompt],
                    width=config.generate.width,
                    height=config.generate.height,
                    num_inference_steps=config.generate.num_inference_steps,
                    guidance_scale=config.generate.guidance_scale,
                    negative_prompt=config.generate.negative_prompt,
                    seed=config.generate.seed + i if config.generate.seed is not None else None,
                )[0]
                image_generation_seconds = time.monotonic() - generation_started

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
                    'actual_width': img.width,
                    'actual_height': img.height,
                    'image_generation_seconds': image_generation_seconds,
                    'api_request': getattr(model._adapter, 'last_image_request', None),
                }

                if 'clip_score' in config.eval.metrics:
                    try:
                        from .metrics.clip_score import compute_clip_score
                        score = compute_clip_score([img], [prompt])
                        sample_result['clip_score'] = score[0]
                    except Exception as e:
                        logger.warning(f'CLIP score computation failed: {e}')

                if 'lpips' in config.eval.metrics:
                    logger.warning('Skipping LPIPS for txt2img: a reference image is required.')

                per_sample_results.append(sample_result)
                progress.update(i + 1)

            elapsed = time.time() - start_time
            logger.info(f'Generated {len(images)} images in {elapsed:.1f}s')

            # Compute aggregate metrics
            metrics = {}
            if 'clip_score' in config.eval.metrics:
                from .metrics.clip_score import CLIP_SCORE_META
                scores = [r['clip_score'] for r in per_sample_results]
                metrics['clip_score_mean'] = sum(scores) / len(scores)
                metrics['clip_score_min'] = min(scores)
                metrics['clip_score_max'] = max(scores)
                metrics['clip_score_meta'] = CLIP_SCORE_META  # 口径随结果落盘（0~1 余弦）
                logger.info(f'CLIP Score: mean={metrics["clip_score_mean"]:.4f}')

            if 'lpips' in config.eval.metrics:
                scores = [r.get('lpips', 0) for r in per_sample_results if 'lpips' in r]
                if scores:
                    metrics['lpips_mean'] = sum(scores) / len(scores)
                    logger.info(f'LPIPS: mean={metrics["lpips_mean"]:.4f}')

            # Save results
            results = {
                'model': _safe_model_name(config.model.model_name_or_path, 'txt2img'),
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
        """Run text-to-video evaluation."""
        from .datasets.prompt_loader import load_video_prompts
        from .models.txt2video import Txt2VideoModel
        from .utils.media import create_thumbnails
        from .utils.progress import ProgressTracker

        output_dir = self._ensure_output_dir(config.eval.output_dir)
        video_dir = output_dir / 'videos'
        video_dir.mkdir(exist_ok=True)
        thumbs_dir = output_dir / 'thumbnails'
        thumbs_dir.mkdir(exist_ok=True)
        frames_dir = output_dir / 'frames'
        frames_dir.mkdir(exist_ok=True)

        # Load prompts — use custom_prompt if provided, otherwise load from dataset
        custom_prompt = getattr(config.eval, 'custom_prompt', None)
        if custom_prompt:
            prompts = [custom_prompt.strip()]
            logger.info(f'Using custom prompt for txt2video: {prompts[0][:80]}...')
        else:
            logger.info(f'Loading video prompts from: {config.eval.prompt_dataset}')
            prompts = load_video_prompts(
                config.eval.prompt_dataset,
                limit=config.eval.prompt_limit,
                custom_path=config.eval.custom_dataset_path,
                shuffle=getattr(config.eval, 'random_prompt', False),
                seed=config.generate.seed,
            )
        logger.info(f'Loaded {len(prompts)} prompts')

        progress = ProgressTracker(output_dir / 'progress.json', total=len(prompts))

        # Load model
        model_config = config.model.dict()
        model_config['num_frames'] = config.generate.num_frames
        model_config['fps'] = config.generate.fps
        model_config['tool'] = 'txt2video'
        model = Txt2VideoModel(model_config)
        model.load()

        try:
            logger.info('Generating videos...')
            start_time = time.time()

            per_sample_results = []

            for i, prompt in enumerate(prompts):
                result = model.generate(
                    [prompt],
                    width=config.generate.width,
                    height=config.generate.height,
                    num_inference_steps=config.generate.num_inference_steps,
                    guidance_scale=config.generate.guidance_scale,
                    negative_prompt=config.generate.negative_prompt,
                    seed=config.generate.seed + i,
                    num_frames=config.generate.num_frames,
                    fps=config.generate.fps,
                    resolution=getattr(config.generate, 'resolution', '') or '',
                    ratio=getattr(config.generate, 'ratio', '') or '',
                    duration=getattr(config.generate, 'duration', 5),
                )[0]

                sample_result = {
                    'index': i,
                    'prompt': prompt,
                }

                # Save video file if available (API mode)
                if result.get('video_path'):
                    import shutil
                    video_src = Path(result['video_path'])
                    video_dst = video_dir / f'{i:04d}.mp4'
                    shutil.copy(video_src, video_dst)
                    sample_result['video_path'] = str(video_dst.relative_to(output_dir))

                # Extract and save frames
                frames = result.get('frames', [])
                if frames:
                    frame_paths = []
                    for fi, frame in enumerate(frames[:min(len(frames), 4)]):
                        fp = frames_dir / f'{i:04d}_frame{fi:02d}.jpg'
                        frame.save(fp, quality=85)
                        frame_paths.append(str(fp.relative_to(output_dir)))

                    # Create thumbnail from first frame
                    thumb_path = thumbs_dir / f'{i:04d}_thumb.jpg'
                    create_thumbnails(frames[:1], [thumb_path])
                    sample_result['thumbnail_path'] = str(thumb_path.relative_to(output_dir))

                    # Compute frame-level metrics
                    if 'clip_score' in config.eval.metrics:
                        try:
                            from .metrics.clip_score import compute_clip_score

                            # Use first 4 frames for CLIP score
                            eval_frames = frames[:min(len(frames), 4)]
                            scores = compute_clip_score(eval_frames, [prompt] * len(eval_frames))
                            sample_result['clip_score'] = sum(scores) / len(scores)
                            sample_result['clip_scores_per_frame'] = scores
                        except Exception as e:
                            logger.warning(f'CLIP score computation failed: {e}')

                per_sample_results.append(sample_result)
                progress.update(i + 1)

            elapsed = time.time() - start_time
            logger.info(f'Generated {len(prompts)} videos in {elapsed:.1f}s')

            # Compute aggregate metrics
            metrics = {}
            if 'clip_score' in config.eval.metrics:
                from .metrics.clip_score import CLIP_SCORE_META
                scores = [r.get('clip_score', 0) for r in per_sample_results if 'clip_score' in r]
                if scores:
                    metrics['clip_score_mean'] = sum(scores) / len(scores)
                    metrics['clip_score_meta'] = CLIP_SCORE_META  # 口径随结果落盘（0~1 余弦）
                    logger.info(f'CLIP Score: mean={metrics["clip_score_mean"]:.4f}')

            # Save results
            results = {
                'model': _safe_model_name(config.model.model_name_or_path, 'txt2video'),
                'model_type': 'txt2video',
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

    def _run_img2img(self, config: AIGCToolConfig) -> Dict[str, Any]:
        """Run image-to-image evaluation."""
        import base64
        import io
        from PIL import Image

        from .models.img2img import Img2ImgModel
        from .utils.media import create_thumbnails
        from .utils.progress import ProgressTracker

        output_dir = self._ensure_output_dir(config.eval.output_dir)
        images_dir = output_dir / 'images'
        images_dir.mkdir(exist_ok=True)
        thumbs_dir = output_dir / 'thumbnails'
        thumbs_dir.mkdir(exist_ok=True)

        # Fail before model loading or any billable request.
        ref_data = getattr(config.eval, 'reference_image', None)
        if not isinstance(ref_data, str) or not ref_data:
            raise ValueError('图生图需要参考图')
        encoded = ref_data.split(',', 1)[1] if ref_data.startswith('data:image/') else ref_data
        if len(encoded) > 4 * ((10 * 1024 * 1024 + 2) // 3):
            raise ValueError('参考图不得超过 10 MB')
        try:
            img_bytes = base64.b64decode(encoded, validate=True)
            if len(img_bytes) > 10 * 1024 * 1024:
                raise ValueError('参考图不得超过 10 MB')
            with Image.open(io.BytesIO(img_bytes)) as source:
                if source.format not in ('PNG', 'JPEG', 'WEBP', 'BMP'):
                    raise ValueError('参考图仅支持 PNG、JPEG、WEBP、BMP')
                source.load()
                ref_img = source.convert('RGB')
            buf = io.BytesIO()
            ref_img.save(buf, format='PNG')
            if len(buf.getvalue()) > 10 * 1024 * 1024:
                raise ValueError('参考图转换 PNG 后不得超过 10 MB')
        except Exception as e:
            raise ValueError('参考图格式无效或超过 10 MB，请重新上传') from e
        reference_images = [ref_img]
        ref_img.save(images_dir / 'reference.png')
        prompt = getattr(config.eval, 'custom_prompt', None)
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError('图生图需要非空编辑指令')
        prompts = [prompt.strip()]
        logger.info('Loaded img2img reference %s and one editing instruction', ref_img.size)

        progress = ProgressTracker(output_dir / 'progress.json', total=len(prompts))

        # Load model
        model_config = config.model.dict()
        model_config['tool'] = 'img2img'
        model = Img2ImgModel(model_config)
        model.load()

        try:
            logger.info('Generating images (img2img)...')
            start_time = time.time()

            per_sample_results = []

            for i, prompt in enumerate(prompts):
                sample_started = time.monotonic()
                img = model.generate(
                    [prompt],
                    reference_images=reference_images if reference_images else None,
                    width=config.generate.width,
                    height=config.generate.height,
                    num_inference_steps=config.generate.num_inference_steps,
                    guidance_scale=config.generate.guidance_scale,
                    negative_prompt=config.generate.negative_prompt,
                    seed=config.generate.seed + i if config.generate.seed is not None else None,
                    prompt_extend=getattr(config.generate, 'prompt_extend', None),
                    watermark=getattr(config.generate, 'watermark', None),
                )[0]

                # Save image
                img_path = images_dir / f'{i:04d}.png'
                img.save(img_path)

                # Create thumbnail
                thumb_path = thumbs_dir / f'{i:04d}_thumb.jpg'
                create_thumbnails([img], [thumb_path])

                sample_result = {
                    'generation_time': time.monotonic() - sample_started,
                    'output_size': list(img.size),
                    'request_parameters': (getattr(model, 'last_image_requests', []) or [{}])[0].get('parameters', {}),
                    'request_seconds': (getattr(model, 'last_image_requests', []) or [{}])[0].get('request_seconds'),
                    'metric_errors': {},
                    'index': i,
                    'prompt': prompt,
                    'image_path': str(img_path.relative_to(output_dir)),
                    'thumbnail_path': str(thumb_path.relative_to(output_dir)),
                }

                if 'clip_score' in config.eval.metrics:
                    try:
                        from .metrics.clip_score import compute_clip_score
                        sample_result['clip_score'] = compute_clip_score([img], [prompt])[0]
                    except Exception as e:
                        sample_result['metric_errors']['clip_score'] = '文本相关性指标计算失败，请检查服务日志和模型权重'
                        logger.warning(f'CLIP score computation failed: {e}')
                if 'lpips' in config.eval.metrics:
                    sample_result['metric_errors']['lpips'] = '当前感知距离实现不是标准 LPIPS，图生图暂不计算'
                per_sample_results.append(sample_result)
                progress.update(i + 1)

            elapsed = time.time() - start_time
            metrics = {}
            if 'clip_score' in config.eval.metrics:
                scores = [r['clip_score'] for r in per_sample_results if 'clip_score' in r]
                metrics['clip_score_valid_samples'] = len(scores)
                if scores:
                    from .metrics.clip_score import CLIP_SCORE_META
                    metrics['clip_score_mean'] = sum(scores) / len(scores)
                    metrics['clip_score_min'] = min(scores)
                    metrics['clip_score_max'] = max(scores)
                    metrics['clip_score_meta'] = CLIP_SCORE_META

            results = {
                'model': _safe_model_name(config.model.model_name_or_path, 'img2img'),
                'model_type': 'img2img',
                'reference_path': 'images/reference.png',
                'reference_size': list(ref_img.size),
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

    def _ensure_output_dir(self, output_dir_str: str) -> Path:
        """Ensure output directory exists."""
        if not output_dir_str:
            raise ValueError('output_dir is required for saving generated media and results.')
        output_dir = Path(output_dir_str)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
