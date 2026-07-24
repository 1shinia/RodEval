"""Text-to-video model adapter."""
import base64
import io
import json
import logging
import os
import requests
import tempfile
from pathlib import Path
from PIL import Image
from typing import Any, Dict, List

from .base import AIGCModelBase, resolve_api_url

logger = logging.getLogger(__name__)


class Txt2VideoModel(AIGCModelBase):
    """Text-to-video model using API or local diffusers."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.pipe = None
        self.api_base = config.get('api_base')
        self.api_key = config.get('api_key')
        self.num_frames = config.get('num_frames', 16)
        self.fps = config.get('fps', 8)

    def load(self) -> None:
        if self.api_base:
            logger.info(f'Using API mode: {self.api_base}')
            return

        try:
            from diffusers import StableVideoDiffusionPipeline
        except ImportError:
            raise ImportError(
                'diffusers >= 0.25 is required for local txt2video. '
                'Install with: pip install diffusers'
            )

        logger.info(f'Loading video model: {self.model_name}')
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
        resolution: str = '720p',
        ratio: str = '16:9',
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Generate videos from prompts.

        Returns:
            List of dicts with keys: video_path, frames (List[PIL.Image])
        """
        if self.api_base:
            return self._generate_api(
                prompts,
                width,
                height,
                num_inference_steps,
                guidance_scale,
                negative_prompt,
                seed,
                num_frames,
                fps,
                resolution,
                ratio,
            )

        if self.pipe is None:
            raise RuntimeError('Model not loaded. Call load() first.')

        return self._generate_local(
            prompts,
            width,
            height,
            num_inference_steps,
            guidance_scale,
            negative_prompt,
            seed,
            num_frames,
            fps,
        )

    def _generate_api(
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
        resolution: str,
        ratio: str,
    ) -> List[Dict[str, Any]]:
        """Generate videos using OpenAI-compatible API."""
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'

        url = resolve_api_url(self.api_base or '', self.config.get('tool', 'txt2video'))
        duration = num_frames // fps if fps > 0 else 5

        results = []
        for i, prompt in enumerate(prompts):
            payload = {
                'model': self.model_name,
                'prompt': prompt,
                'n': 1,
                'size': f'{width}x{height}',
            }
            if duration >= 2:
                payload['duration'] = duration
            if resolution:
                payload['resolution'] = resolution
            if ratio:
                payload['ratio'] = ratio

            logger.info(
                f'API payload keys: {list(payload.keys())}, duration={payload.get("duration")}, '
                f'resolution={payload.get("resolution")}, ratio={payload.get("ratio")}'
            )

            response = requests.post(url, json=payload, headers=headers, timeout=300)
            if not response.ok:
                logger.error(f'API error {response.status_code}: {response.text}')
            response.raise_for_status()

            data = response.json()
            logger.info(f'API response keys: {list(data.keys())}')

            # Handle async video generation (task_id + status pattern)
            if 'task_id' in data and 'status' in data:
                data = self._poll_async_task(data, url, headers, i)

            # Handle different response formats
            item = None
            if 'data' in data and isinstance(data['data'], list) and len(data['data']) > 0:
                item = data['data'][0]
            elif 'url' in data:
                item = data  # Flat response with url field
            elif 'video' in data:
                item = {'url': data['video']}
            elif 'result' in data:
                result_data = data['result']
                if isinstance(result_data, dict):
                    item = result_data
                elif isinstance(result_data, list) and len(result_data) > 0:
                    item = result_data[0]
            else:
                raise ValueError(
                    f'Unexpected API response format: keys={list(data.keys())}. '
                    f'First 200 chars: {str(data)[:200]}'
                )

            video_path = None
            frames: List[Image.Image] = []

            if 'b64_json' in item:
                video_data = base64.b64decode(item['b64_json'])
                video_path = self._save_video_bytes(video_data, i)
            elif 'url' in item:
                video_url = item['url']
                resp = requests.get(video_url, timeout=120)
                resp.raise_for_status()
                video_path = self._save_video_bytes(resp.content, i)
            else:
                raise ValueError(f'Unexpected API response: {list(item.keys())}')

            # Extract frames for metrics
            if video_path:
                frames = self._extract_frames(video_path, max_frames=num_frames)

            results.append({
                'video_path': str(video_path) if video_path else None,
                'frames': frames,
            })

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
        """Generate videos using local diffusers pipeline (placeholder)."""
        import torch

        generator = torch.Generator(device=self.device).manual_seed(seed)
        results = []

        for i, prompt in enumerate(prompts):
            logger.debug(f'Generating video for: {prompt[:50]}...')
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

            # Extract frames from output
            frames: List[Image.Image] = []
            if hasattr(output, 'frames') and output.frames:
                frames = output.frames[0] if isinstance(output.frames[0], list) else output.frames
            elif isinstance(output, list):
                frames = [f if isinstance(f, Image.Image) else Image.fromarray(f) for f in output]

            results.append({
                'video_path': None,  # Local mode: frames only, no video file saved
                'frames': frames,
            })

        return results

    def _poll_async_task(self, data: dict, api_base: str, headers: dict, index: int) -> dict:
        """Poll async video generation task until complete.

        Returns the completed response data.
        """
        import time as _time

        task_id = data.get('task_id', data.get('id', ''))
        status = data.get('status', '')
        logger.info(f'Async video task {task_id}: status={status}')

        if status in ('completed', 'succeeded', 'done'):
            return data

        # Build poll URL: strip /generations suffix, append /{task_id}
        base = api_base.rstrip('/')
        poll_url = f'{base}/{task_id}'

        max_attempts = 60  # 5 minutes max (5s intervals)
        for attempt in range(max_attempts):
            _time.sleep(5)
            resp = requests.get(poll_url, headers=headers, timeout=30)
            if not resp.ok:
                logger.warning(f'Poll attempt {attempt + 1}: HTTP {resp.status_code}')
                continue

            raw = resp.json()
            # Unwrap common API wrappers: {code, message, data}
            data = raw.get('data', raw)
            while isinstance(data, dict) and 'data' in data:
                if isinstance(data['data'], dict) and 'status' in data['data']:
                    data = data['data']
                else:
                    break
            status = data.get('status', '')
            progress = data.get('progress', 0)
            logger.info(f'Video task {task_id}: status={status}, progress={progress}')

            if status in ('completed', 'succeeded', 'done'):
                logger.info(f'Video task {task_id} completed')
                # Completed response: return the data dict directly
                # It should have url/video_url/b64_json/content for the main parser
                if 'url' in data or 'video_url' in data or 'b64_json' in data:
                    return data
                # doubao/seedance format: {content: "video_url_or_base64", ...}
                if 'content' in data:
                    content = data['content']
                    if isinstance(content, str):
                        return {'url': content}
                    if isinstance(content, dict) and 'url' in content:
                        return {'url': content['url']}
                    if isinstance(content, dict) and 'video_url' in content:
                        return {'url': content['video_url']}
                    return {'url': str(content)}
                # Some APIs nest the result under a 'result' or 'video' key
                if 'result' in data:
                    return {'url': data['result']}
                if 'video' in data:
                    return {'url': data['video']} if isinstance(data['video'], str) else data['video']
                return data
            elif status in ('failed', 'error', 'cancelled'):
                raise RuntimeError(f'Video task {task_id} failed: {data}')

        raise TimeoutError(f'Video task {task_id} did not complete within {max_attempts * 5}s')

    def _save_video_bytes(self, data: bytes, index: int) -> Path:
        """Save video bytes to a temporary file."""
        tmpdir = Path(tempfile.gettempdir()) / 'aigc_videos'
        tmpdir.mkdir(parents=True, exist_ok=True)
        path = tmpdir / f'video_{index:04d}.mp4'
        path.write_bytes(data)
        return path

    def _extract_frames(self, video_path: Path, max_frames: int = 16) -> List[Image.Image]:
        """Extract frames from a video file using ffmpeg or cv2."""
        try:
            import cv2
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.warning(f'Cannot open video: {video_path}')
                return []

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                cap.release()
                return []

            # Sample evenly spaced frames
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
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info('Video model unloaded')
