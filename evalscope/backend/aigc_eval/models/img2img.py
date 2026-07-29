"""Image-to-image model adapter using diffusers or API."""

import base64
import io
import logging
from typing import Any, Dict, List, Optional

from PIL import Image

from .adapters import create_adapter
from .base import AIGCModelBase

logger = logging.getLogger(__name__.replace('evalscope', 'evalperf'))


class Img2ImgModel(AIGCModelBase):
    """Image-to-image model using API adapter or local diffusers."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.pipe = None
        self.api_base = config.get('api_base')
        self._adapter = None
        self.strength = config.get('strength', 0.8)

    def load(self) -> None:
        """Load the img2img pipeline or init API adapter."""
        if self.api_base:
            self._adapter = create_adapter(self.config)
            logger.info('Using API adapter: %s', type(self._adapter).__name__)
            return

        try:
            from diffusers import StableDiffusionImg2ImgPipeline
        except ImportError:
            raise ImportError(
                'diffusers is required for local img2img. Install with: pip install diffusers'
            )

        logger.info('Loading img2img model: %s', self.model_name)
        self._dtype = self._get_dtype()
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            self.model_name,
            torch_dtype=self._dtype,
            safety_checker=None,
        )
        self.pipe = self.pipe.to(self.device)
        logger.info('Img2Img model loaded successfully')

    def _get_dtype(self):
        import torch
        dtype_map = {'float16': torch.float16, 'float32': torch.float32, 'bfloat16': torch.bfloat16}
        return dtype_map.get(self.dtype, torch.float16)

    def generate(
        self,
        prompts: List[str],
        reference_images: Optional[List[Image.Image]] = None,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        strength: float = 0.8,
        negative_prompt: str = '',
        seed: int = 42,
        **kwargs,
    ) -> List[Image.Image]:
        """Generate images from prompts and reference images.

        Args:
            prompts: Text prompts.
            reference_images: Optional reference images for img2img.
            width: Output image width.
            height: Output image height.
            num_inference_steps: Number of denoising steps.
            guidance_scale: CFG scale.
            strength: How much to transform the reference image (0-1).
            negative_prompt: Negative prompt.
            seed: Random seed.

        Returns:
            List of PIL Images.
        """
        if self._adapter:
            tool = self.config.get('tool', 'img2img')
            images: List[Image.Image] = []
            for i, prompt in enumerate(prompts):
                ref_b64 = None
                if reference_images and i < len(reference_images) and reference_images[i] is not None:
                    ref = self._fit_to_size(reference_images[i], width, height)
                    buf = io.BytesIO()
                    ref.save(buf, format='PNG')
                    ref_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

                img_bytes = self._adapter.generate(
                    prompt,
                    tool=tool,
                    width=width,
                    height=height,
                    seed=seed,
                    strength=strength,
                    reference_image_b64=ref_b64,
                )
                images.append(Image.open(io.BytesIO(img_bytes)).convert('RGB'))
            return images

        if self.pipe is None:
            raise RuntimeError('Model not loaded. Call load() first.')

        return self._generate_local(
            prompts, reference_images,
            width, height, num_inference_steps,
            guidance_scale, strength, negative_prompt, seed,
        )

    def _generate_local(
        self,
        prompts: List[str],
        reference_images: Optional[List[Image.Image]],
        width: int,
        height: int,
        num_inference_steps: int,
        guidance_scale: float,
        strength: float,
        negative_prompt: str,
        seed: int,
    ) -> List[Image.Image]:
        """Generate images using local diffusers Img2Img pipeline."""
        import torch

        assert self.pipe is not None, 'pipe must be loaded before _generate_local'

        generator = torch.Generator(device=self.device).manual_seed(seed)
        images = []

        for i, prompt in enumerate(prompts):
            if reference_images and i < len(reference_images) and reference_images[i] is not None:
                init_image = self._fit_to_size(reference_images[i], width, height)
            else:
                init_image = Image.new('RGB', (width, height), color=(128, 128, 128))

            result = self.pipe(
                prompt=prompt,
                image=init_image,
                strength=strength,
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
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info('Img2Img model unloaded')

    @staticmethod
    def _fit_to_size(img: Image.Image, target_width: int, target_height: int) -> Image.Image:
        """Resize image to fit target size while preserving aspect ratio.

        Pads with black bars (letterbox/pillarbox) to achieve exact target dimensions,
        avoiding distortion from direct resize when aspect ratios differ.
        """
        img_w, img_h = img.size
        if img_w == target_width and img_h == target_height:
            return img

        scale = min(target_width / img_w, target_height / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        canvas = Image.new('RGB', (target_width, target_height), (0, 0, 0))
        offset_x = (target_width - new_w) // 2
        offset_y = (target_height - new_h) // 2
        canvas.paste(resized, (offset_x, offset_y))

        return canvas
