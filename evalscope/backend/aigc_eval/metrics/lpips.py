"""LPIPS (Learned Perceptual Image Patch Similarity) metric.

Measures perceptual quality of generated images using AlexNet features.
Lower LPIPS = higher perceptual quality (less distortion).

Two modes:
  - Self-comparison (default): compare against JPEG-compressed version of self,
    measures image detail richness / fidelity.
  - Reference-comparison (img2img): compare against a reference image,
    measures how much the generated image differs from the reference.

Score range: approximately 0.0 (identical) to 1.0 (completely different).
Uses cosine distance on L2-normalized AlexNet features across 5 layers,
weighted and averaged, with zero-channel masking for sparse deep layers.
"""
import io
import logging
import torch
import numpy as np
from PIL import Image
from torchvision import models, transforms
from typing import List, Optional

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
    reference_images: Optional[List[Image.Image]] = None,
    device: str = 'cpu',
) -> List[float]:
    """Compute LPIPS perceptual similarity scores.

    When reference_images is provided (e.g., img2img), compares each generated
    image against its corresponding reference image. Lower LPIPS = more similar
    to the reference.

    When reference_images is None (default, e.g., txt2img), uses self-comparison
    with a JPEG-compressed version of each image. Higher LPIPS distance = more
    fine details lost during compression = higher image detail.

    Score range: approximately 0.0 (perceptually identical) to 1.0 (completely different).

    Args:
        images: List of generated PIL Images
        reference_images: Optional list of reference PIL Images (same length as images)
        device: Device to run on

    Returns:
        List of LPIPS scores (normalized to ~0-1 range)
    """
    if not images:
        return []

    model, transform = _get_model(device)

    # Layer weights from the LPIPS paper (AlexNet: conv1→conv5), normalized
    # to sum to 1.0 for interpretability. Values from Zhang et al. 2018.
    layer_weights = [1.0 / 2.6, 1.0 / 4.8, 1.0 / 3.7, 1.0 / 5.6, 1.0 / 10.0]
    weight_sum = sum(layer_weights)

    scores = []
    with torch.no_grad():
        for i, img in enumerate(images):
            tensor = transform(img).unsqueeze(0).to(device)

            if reference_images and i < len(reference_images) and reference_images[i] is not None:
                # Reference-comparison mode: compare against reference image
                ref_tensor = transform(reference_images[i]).unsqueeze(0).to(device)
            else:
                # Self-comparison mode: compare against JPEG-compressed version
                buf = io.BytesIO()
                img.convert('RGB').save(buf, format='JPEG', quality=20)
                buf.seek(0)
                compressed = Image.open(buf).convert('RGB')
                ref_tensor = transform(compressed).unsqueeze(0).to(device)

            orig_feats = _extract_features(model, tensor)
            ref_feats = _extract_features(model, ref_tensor)

            # Compute cosine-distance-based perceptual distance per layer
            total = 0.0
            for orig, ref, w in zip(orig_feats, ref_feats, layer_weights):
                dist = _layer_cosine_distance(orig, ref)
                total += w * dist

            # Normalize by weight sum; cosine distance is already 0-2, scale to 0-1
            scores.append(total / weight_sum / 2.0)

    return scores


def _layer_cosine_distance(orig: torch.Tensor, ref: torch.Tensor) -> float:
    """Compute average cosine distance for a single feature layer.

    Handles sparse layers (deep conv features where 90%+ channels are zero after
    ReLU) by masking out channels where both images have near-zero activation.

    Args:
        orig: Feature tensor, shape (N, C, H, W) or (1, C, H, W)
        ref:  Reference feature tensor, same shape

    Returns:
        Mean cosine distance across active channels (0.0 = identical, 2.0 = opposite).
    """
    N = orig.size(0)
    # Flatten spatial dimensions: (N, C, H*W)
    orig_f = orig.reshape(N, orig.size(1), -1)
    ref_f = ref.reshape(N, ref.size(1), -1)

    # L2 norm per channel
    orig_norm = orig_f.norm(p=2, dim=2)  # (N, C)
    ref_norm = ref_f.norm(p=2, dim=2)    # (N, C)

    # Only consider channels where at least one side has signal
    eps = 1e-6
    mask = ((orig_norm > eps) | (ref_norm > eps)).float()  # (N, C)

    # Normalize (safe division with eps)
    orig_normed = orig_f / (orig_norm.unsqueeze(2) + eps)
    ref_normed = ref_f / (ref_norm.unsqueeze(2) + eps)

    # Cosine similarity per channel: dot product of unit vectors
    sim = (orig_normed * ref_normed).sum(dim=2)  # (N, C)
    # Cosine distance (0-2), clamped to avoid negatives from FP noise
    dist = (1.0 - sim).clamp(min=0.0)

    # Average over active channels only
    masked_dist = dist * mask
    active_count = mask.sum(dim=1).clamp(min=1.0)
    return (masked_dist.sum(dim=1) / active_count).mean().item()


def _extract_features(model, x):
    """Extract features from specific AlexNet layers."""
    features = []
    target_layers = [1, 4, 7, 9, 11]  # conv1, conv2, conv3, conv4, conv5 indices

    for i, layer in enumerate(model):
        x = layer(x)
        if i in target_layers:
            features.append(x)

    return features
