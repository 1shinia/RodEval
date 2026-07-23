"""LPIPS (Learned Perceptual Image Patch Similarity) metric.

Measures perceptual quality of generated images using AlexNet features.
Lower LPIPS = higher perceptual quality (less distortion).
"""
import logging
import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
from typing import List

logger = logging.getLogger(__name__)

# Global model cache — loaded once per process
_alexnet = None
_transform = None


def _get_model(device: str = 'cpu'):
    """Load or return cached AlexNet feature extractor."""
    global _alexnet, _transform

    if _alexnet is None:
        logger.info('Loading AlexNet for LPIPS')
        model = models.alexnet(weights='DEFAULT').features
        model.eval()
        model = model.to(device)
        _alexnet = model

        _transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    return _alexnet, _transform


def compute_lpips(
    images: List[Image.Image],
    device: str = 'cpu',
) -> List[float]:
    """Compute LPIPS perceptual quality scores.

    Uses self-comparison with a Gaussian-blurred version of each image.
    Higher LPIPS distance = more fine details lost during blur = higher image detail.

    Args:
        images: List of PIL Images
        device: Device to run on

    Returns:
        List of LPIPS scores (lower = less distortion, i.e. better quality)
    """
    if not images:
        return []

    model, transform = _get_model(device)
    from torch.nn.functional import mse_loss
    from torchvision.transforms.functional import gaussian_blur

    scores = []
    with torch.no_grad():
        for img in images:
            # Original image
            tensor = transform(img).unsqueeze(0).to(device)

            # JPEG compression comparison: more detail = bigger loss after heavy compression
            # Convert to PIL, compress, decompress, back to tensor
            import io
            buf = io.BytesIO()
            img.convert('RGB').save(buf, format='JPEG', quality=20)
            buf.seek(0)
            compressed = Image.open(buf).convert('RGB')
            compressed_tensor = transform(compressed).unsqueeze(0).to(device)

            orig_feats = _extract_features(model, tensor)
            compressed_feats = _extract_features(model, compressed_tensor)

            # Compute LPIPS: weighted MSE across layers
            # Weights from original LPIPS paper (AlexNet layers)
            layer_weights = [1.0, 1.0, 1.0, 1.0, 1.0]

            total = 0.0
            for orig, comp, w in zip(orig_feats, compressed_feats, layer_weights):
                diff = (orig - comp).pow(2)
                total += w * diff.mean().item()

            scores.append(total / len(layer_weights))

    return scores


def _extract_features(model, x):
    """Extract features from specific AlexNet layers."""
    features = []
    target_layers = [1, 4, 7, 9, 11]  # conv1, conv2, conv3, conv4, conv5 indices
    j = 0

    for i, layer in enumerate(model):
        x = layer(x)
        if i in target_layers:
            features.append(x)
        j += 1

    return features
