"""FVD (Fréchet Video Distance) metric for video quality evaluation.

Uses CLIP ViT-B/32 vision encoder for per-frame feature extraction,
then mean-pools across frames before computing Fréchet Distance.
"""
import logging
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Optional

logger = logging.getLogger(__name__)

# Cache for loaded models to avoid reloading per evaluation
_clip_model = None
_clip_processor = None


def compute_fvd(
    generated_frames: List[List[Image.Image]],
    prompts: List[str],
    reference_video_dir: Optional[str] = None,
    device: Optional[str] = None,
) -> float:
    """Compute FVD between generated and reference video features.

    Args:
        generated_frames: List of frame sequences, one per generated video.
            Each element is a list of PIL Images (frames).
        prompts: Text prompts (used for CLIP fallback mode).
        reference_video_dir: Path to reference video files.
            If None, returns 0 (reference-free mode) or uses generated stats
            as a self-reference for debugging.
        device: Device to run on (auto-detected if None).

    Returns:
        FVD score (lower is better). Returns -1 if computation fails.
    """
    if device is None:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if not generated_frames or not any(len(f) > 0 for f in generated_frames):
        logger.warning('No frames available for FVD computation')
        return -1.0

    gen_features = _extract_features(generated_frames, prompts, device)

    if gen_features is None or len(gen_features) == 0:
        logger.warning('Could not extract features for FVD')
        return -1.0

    # Extract reference features
    if reference_video_dir:
        ref_features = _load_reference_features(reference_video_dir, device)
        if ref_features is None:
            logger.warning('Could not load reference features, using self-reference')
            ref_features = gen_features
    else:
        logger.info('No reference video dir, using generated features as reference (debug mode)')
        ref_features = gen_features

    # Compute Fréchet Distance
    fd = _frechet_distance(gen_features, ref_features)
    logger.info(f'FVD: {fd:.2f}')
    return fd


def _extract_features(
    frame_sequences: List[List[Image.Image]],
    prompts: List[str],
    device: str,
) -> Optional[np.ndarray]:
    """Extract features using CLIP vision encoder (per-frame, then mean-pool)."""
    global _clip_model, _clip_processor

    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor
    except ImportError:
        logger.debug('CLIP not available for feature extraction')
        return None

    if _clip_model is None:
        try:
            _clip_model = CLIPModel.from_pretrained(
                'openai/clip-vit-base-patch32',
                local_files_only=True,
                use_safetensors=True,
            )
            _clip_processor = CLIPProcessor.from_pretrained(
                'openai/clip-vit-base-patch32',
                local_files_only=True,
            )
            _clip_model = _clip_model.to(device)
            _clip_model.eval()
            logger.info('Loaded CLIP ViT-B/32 for FVD feature extraction')
        except Exception as e:
            logger.warning(f'Failed to load CLIP: {e}')
            return None

    features = []
    with torch.no_grad():
        for frames in frame_sequences:
            if len(frames) == 0:
                continue

            # Use up to 8 evenly sampled frames
            indices = np.linspace(0, len(frames) - 1, min(8, len(frames)), dtype=int)
            selected = [frames[i] for i in indices]

            frame_feats = []
            for frame in selected:
                inputs = _clip_processor(images=frame, return_tensors='pt')
                inputs = {k: v.to(device) for k, v in inputs.items()}
                img_feat = _clip_model.get_image_features(**inputs).cpu().numpy()
                frame_feats.append(img_feat[0])

            # Mean pool across frames
            features.append(np.mean(frame_feats, axis=0))

    if not features:
        return None

    return np.stack(features)


def _load_reference_features(
    reference_video_dir: str,
    device: str,
) -> Optional[np.ndarray]:
    """Load or extract features from reference videos.

    If a pre-computed features.npy file exists, load it directly.
    Otherwise, extract features from video files in the directory.
    """
    ref_path = Path(reference_video_dir)
    features_file = ref_path / 'fvd_features.npy'

    if features_file.exists():
        logger.info(f'Loading pre-computed reference features from {features_file}')
        return np.load(features_file)

    # Try to load videos and extract features
    video_files = sorted(ref_path.glob('*.mp4')) + sorted(ref_path.glob('*.avi'))
    if not video_files:
        logger.warning(f'No video files found in {reference_video_dir}')
        return None

    logger.info(f'Extracting features from {len(video_files)} reference videos...')
    all_frames = []
    for vf in video_files:
        frames = _read_video_frames(vf, max_frames=16)
        if frames:
            all_frames.append(frames)

    if not all_frames:
        return None

    feat = _extract_features(all_frames, [], device)

    if feat is not None:
        np.save(features_file, feat)
        logger.info(f'Saved reference features to {features_file}')

    return feat


def _read_video_frames(video_path: Path, max_frames: int = 16) -> List[Image.Image]:
    """Read frames from a video file."""
    try:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return []

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return []

        indices = np.linspace(0, total - 1, min(max_frames, total), dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        cap.release()
        return frames
    except ImportError:
        logger.warning('opencv-python not available for reading reference videos')
        return []


def _frechet_distance(feats1: np.ndarray, feats2: np.ndarray) -> float:
    """Compute Fréchet Distance between two feature distributions.

    FD = ||mu1 - mu2||^2 + Tr(S1 + S2 - 2*sqrt(S1*S2))
    """
    # Need at least 2 samples per distribution for meaningful covariance
    if len(feats1) < 2 or len(feats2) < 2:
        return 0.0

    # Use eigendecomposition-based sqrtm (faster than scipy for high-dim features)
    sqrtm = _approx_sqrtm

    mu1, mu2 = np.mean(feats1, axis=0), np.mean(feats2, axis=0)
    diff = mu1 - mu2
    mean_dist = np.dot(diff, diff)

    # Self-reference: feats1 == feats2 → sqrtm(S²) = S
    if feats1 is feats2:
        return float(max(0, mean_dist))

    sigma1 = np.cov(feats1, rowvar=False)
    sigma2 = np.cov(feats2, rowvar=False)

    # Covariance term
    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fd = mean_dist + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(max(0, fd))


def _approx_sqrtm(matrix: np.ndarray) -> np.ndarray:
    """Approximate matrix square root using eigendecomposition (fallback)."""
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.maximum(eigenvalues, 0)
    return eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T
