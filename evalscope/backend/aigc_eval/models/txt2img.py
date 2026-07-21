"""Text-to-image model adapter using diffusers."""
import logging
import torch
from pathlib import Path
from PIL import Image
from typing import Any, Dict, List, Optional

from .base import AIGCModelBase

logger = logging.getLogger(__name__)


class Txt2ImgModel(AIGCModelBase):
    """Text-to-image model using HuggingFace diffusers."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.pipe = None
        self.api_base = config.get('api_base')
        self.api_key = config.get('api_key')

    def load(self) -> None:
        """Load the text-to-image pipeline."""
        if self.api_base:
            logger.info(f'Using API mode: {self.api_base}')
            # API mode will be handled in generate()
            return

        try:
            from diffusers import StableDiffusionPipeline
        except ImportError:
            raise ImportError('diffusers is required for local txt2img. Install with: pip install diffusers')

        logger.info(f'Loading model: {self.model_name}')
        dtype = torch.float16 if self.dtype == 'float16' else torch.float32

        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            safety_checker=None,
        )
        self.pipe = self.pipe.to(self.device)
        logger.info('Model loaded successfully')

    def generate(
        self,
        prompts: List[str],
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: str = '',
        seed: int = 42,
        **kwargs,
    ) -> List[Image.Image]:
        """Generate images from prompts."""
        if self.api_base:
            return self._generate_api(prompts, width, height, **kwargs)

        if self.pipe is None:
            raise RuntimeError('Model not loaded. Call load() first.')

        generator = torch.Generator(device=self.device).manual_seed(seed)

        images = []
        for prompt in prompts:
            logger.debug(f'Generating image for: {prompt[:50]}...')
            result = self.pipe(
                prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt if negative_prompt else None,
                generator=generator,
            )
            images.append(result.images[0])

        return images

    def _generate_api(self, prompts: List[str], width: int, height: int, **kwargs) -> List[Image.Image]:
        """Generate images using OpenAI-compatible API."""
        import base64
        import io
        import requests

        images = []
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        } if self.api_key else {
            'Content-Type': 'application/json'
        }

        # OpenAI-compatible images/generations endpoint
        api_base = self.api_base or ''
        url = f'{api_base.rstrip("/")}/images/generations'

        # Map width/height to size string
        size = f'{width}x{height}'

        for prompt in prompts:
            payload = {
                'model': self.model_name,
                'prompt': prompt,
                'n': 1,
                'size': size,
            }
            response = requests.post(url, json=payload, headers=headers, timeout=120)
            if not response.ok:
                logger.error(f'API error {response.status_code}: {response.text}')
            response.raise_for_status()

            data = response.json()
            # OpenAI format: {"data": [{"b64_json": "..."} or {"url": "..."}]}
            img_item = data['data'][0]

            if 'b64_json' in img_item:
                img_data = base64.b64decode(img_item['b64_json'])
                img = Image.open(io.BytesIO(img_data))
            elif 'url' in img_item:
                img_resp = requests.get(img_item['url'], timeout=60)
                img_resp.raise_for_status()
                img = Image.open(io.BytesIO(img_resp.content))
            else:
                raise ValueError(f'Unexpected API response format: {list(img_item.keys())}')

            images.append(img)

        return images

    def unload(self) -> None:
        """Unload the model."""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info('Model unloaded')
