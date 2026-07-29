"""Text-to-image model adapter using diffusers or API."""

import logging
from io import BytesIO
from typing import Any, Dict, List

import torch
from PIL import Image

from .adapters import create_adapter
from .base import AIGCModelBase

logger = logging.getLogger(__name__.replace('evalscope', 'evalperf'))


class Txt2ImgModel(AIGCModelBase):
    """Text-to-image model using HuggingFace diffusers (local) or API adapter."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.pipe = None
        self.api_base = config.get('api_base')
        self._adapter = None

    def load(self) -> None:
        """Load the text-to-image pipeline or init API adapter."""
        if self.api_base:
            self._adapter = create_adapter(self.config)
            logger.info('Using API adapter: %s', type(self._adapter).__name__)
            return

        try:
            from diffusers import StableDiffusionPipeline
        except ImportError:
            raise ImportError(
                'diffusers is required for local txt2img. Install with: pip install diffusers'
            )

        logger.info('Loading model: %s', self.model_name)
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
        if self._adapter:
            tool = self.config.get('tool', 'txt2img')
            images: List[Image.Image] = []
            for prompt in prompts:
                img_bytes = self._adapter.generate(
                    prompt,
                    tool=tool,
                    width=width,
                    height=height,
                    seed=seed,
                )
                images.append(Image.open(BytesIO(img_bytes)).convert('RGB'))
            return images

        if self.pipe is None:
            raise RuntimeError('Model not loaded. Call load() first.')

        generator = torch.Generator(device=self.device).manual_seed(seed)
        images = []
        for prompt in prompts:
            logger.debug('Generating image for: %s...', prompt[:50])
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

    def unload(self) -> None:
        """Unload the model."""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info('Model unloaded')
