"""CLIP Score metric for text-image similarity."""
import logging
import torch
from PIL import Image
from typing import List

logger = logging.getLogger(__name__)


def compute_clip_score(
    images: List[Image.Image],
    prompts: List[str],
    model_name: str = 'openai/clip-vit-base-patch32',
    device: str = 'cuda',
) -> List[float]:
    """Compute CLIP Score between images and text prompts.

    Args:
        images: List of PIL Images
        prompts: List of text prompts
        model_name: CLIP model name
        device: Device to run on

    Returns:
        List of CLIP scores (one per image-prompt pair)
    """
    try:
        from transformers import CLIPModel, CLIPProcessor
    except ImportError:
        raise ImportError('transformers is required for CLIP Score. Install with: pip install transformers')

    if len(images) != len(prompts):
        raise ValueError(f'Number of images ({len(images)}) != number of prompts ({len(prompts)})')

    logger.info(f'Loading CLIP model: {model_name}')
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)

    device = device if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()

    scores = []
    with torch.no_grad():
        for img, prompt in zip(images, prompts):
            inputs = processor(
                text=[prompt],
                images=img,
                return_tensors='pt',
                padding=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            score = logits_per_image.cpu().item()
            scores.append(score)

    logger.info(f'Computed CLIP scores for {len(scores)} samples')
    return scores
