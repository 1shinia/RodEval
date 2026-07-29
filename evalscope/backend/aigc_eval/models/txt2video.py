"""Text-to-video model adapter using API or local diffusers."""

import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image

from .adapters import create_adapter
from .base import AIGCModelBase

logger = logging.getLogger(__name__.replace('evalscope', 'evalperf'))


class Txt2VideoModel(AIGCModelBase):
    """Text-to-video model using API adapter or local diffusers."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.pipe = None
        self.api_base = config.get('api_base')
        self._adapter = None
        self.num_frames = config.get('num_frames', 16)
        self.fps = config.get('fps', 8)

    def load(self) -> None:
        """Load the video pipeline or init API adapter."""
        if self.api_base:
            self._adapter = create_adapter(self.config)
            logger.info('Using API adapter: %s', type(self._adapter).__name__)
            return

        try:
            from diffusers import StableVideoDiffusionPipeline
        except ImportError:
            raise ImportError(
                'diffusers >= 0.25 is required for local txt2video. '
                'Install with: pip install diffusers'
            )

        logger.info('Loading video model: %s', self.model_name)
        self.pipe = StableVideoDiffusionPipeline.from_pretrained(
            self.model_name,
            torch_dtype=self._get_dtype(),
        )
        self.pipe = self.pipe.to(self.device)
        logger.info('Video model loaded successfully')

    def _get_dtype(self):
        import torch
        dtype_map = {'float16': torch.float16, 'float32': torch.float32, 'bfloat16': torch.bfloat16}
        return dtype_map.get(self.dtype, torch.float16)

    def generate(
        self,
        prompts: List[str],
        width: int = 1024,
        height: int = 576,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: str = '',
        seed: int = 42,
        num_frames: int = 16,
        fps: int = 8,
        resolution: str = '',
        ratio: str = '',
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Generate videos from prompts.

        Returns:
            List of dicts with keys: video_path, frames (List[PIL.Image])
        """
        if self._adapter:
            return self._generate_api(
                prompts, width, height, seed,
                num_frames, fps, resolution, ratio,
            )

        if self.pipe is None:
            raise RuntimeError('Model not loaded. Call load() first.')

        return self._generate_local(
            prompts, width, height, num_inference_steps,
            guidance_scale, negative_prompt, seed,
            num_frames, fps,
        )

    def _generate_api(
        self,
        prompts: List[str],
        width: int,
        height: int,
        seed: int,
        num_frames: int,
        fps: int,
        resolution: str,
        ratio: str,
    ) -> List[Dict[str, Any]]:
        """Generate videos using the API adapter."""
        tool = self.config.get('tool', 'txt2video')
        results: List[Dict[str, Any]] = []

        for i, prompt in enumerate(prompts):
            video_bytes = self._adapter.generate(
                prompt,
                tool=tool,
                width=width,
                height=height,
                seed=seed,
                num_frames=num_frames,
                fps=fps,
                resolution=resolution,
                ratio=ratio,
            )
            if video_bytes:
                video_path = self._save_video_bytes(video_bytes, i)
                frames = self._extract_frames(video_path, max_frames=num_frames)
                results.append({
                    'video_path': str(video_path),
                    'frames': frames,
                })
            else:
                results.append({'video_path': None, 'frames': []})

        return results

    def _generate_local(
        self,
        prompts: List[str],
        width: int,
        height: int,
        num_inference_steps: int,
        guidance_scale: float,
        negative_prompt: str,
        seed: int,
        num_frames: int,
        fps: int,
    ) -> List[Dict[str, Any]]:
        """Generate videos using local diffusers pipeline."""
        import torch

        assert self.pipe is not None, 'pipe must be loaded before _generate_local'

        generator = torch.Generator(device=self.device).manual_seed(seed)
        results = []

        for i, prompt in enumerate(prompts):
            logger.debug('Generating video for: %s...', prompt[:50])
            output = self.pipe(
                prompt,
                width=width,
                height=height,
                num_frames=num_frames if hasattr(self.pipe, 'num_frames') else None,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt if negative_prompt else None,
                generator=generator,
            )

            frames: List[Image.Image] = []
            if hasattr(output, 'frames') and output.frames:
                frames = output.frames[0] if isinstance(output.frames[0], list) else output.frames
            elif isinstance(output, list):
                frames = [f if isinstance(f, Image.Image) else Image.fromarray(f) for f in output]

            results.append({
                'video_path': None,
                'frames': frames,
            })

        return results

    def _save_video_bytes(self, data: bytes, index: int) -> Path:
        """Save video bytes to a temporary file."""
        tmpdir = Path(tempfile.gettempdir()) / 'aigc_videos'
        tmpdir.mkdir(parents=True, exist_ok=True)
        path = tmpdir / f'video_{index:04d}.mp4'
        path.write_bytes(data)
        return path

    def _extract_frames(self, video_path: Path, max_frames: int = 16) -> List[Image.Image]:
        """Extract frames from a video file using cv2."""
        try:
            import cv2
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.warning('Cannot open video: %s', video_path)
                return []

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                cap.release()
                return []

            indices = [int(i * total_frames / max_frames) for i in range(min(max_frames, total_frames))]
            frames = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            cap.release()
            return frames
        except ImportError:
            logger.warning('opencv-python not available, skipping frame extraction')
            return []

    def unload(self) -> None:
        """Unload the model."""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info('Video model unloaded')
