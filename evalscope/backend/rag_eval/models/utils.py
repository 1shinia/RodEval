"""Shared utilities for model loading and downloading."""
import os
from typing import Optional

from evalscope.constants import HubType
from evalscope.utils.logger import get_logger

logger = get_logger()

HF_CACHE = os.path.expanduser('~/.cache/huggingface/hub')
MS_CACHE = os.path.expanduser('~/.cache/modelscope/hub')


def _find_in_cache(model_id: str, cache_root: str) -> Optional[str]:
    """Find a model in the HF/ModelScope cache directory.

    Cache layout:
        cache_root/models--org--repo/snapshots/<hash>/

    Returns the snapshot path if found, None otherwise.
    """
    folder = model_id.replace('/', '--')
    cache_dir = os.path.join(cache_root, 'models--' + folder)
    snapshots = os.path.join(cache_dir, 'snapshots')
    if not os.path.isdir(snapshots):
        return None
    for entry in os.listdir(snapshots):
        candidate = os.path.join(snapshots, entry)
        if os.path.isdir(candidate) and os.path.isfile(os.path.join(candidate, 'config.json')):
            return candidate
    return None


def download_model(model_id: str, revision: Optional[str] = 'master', hub: str = HubType.MODELSCOPE) -> str:
    """Download a model from ModelScope or HuggingFace hub."""
    if hub == HubType.MODELSCOPE:
        from modelscope import snapshot_download
        logger.info(f'Downloading model {model_id} from ModelScope (revision={revision})')
        return snapshot_download(model_id=model_id, revision=revision)
    else:
        from huggingface_hub import snapshot_download as hf_snapshot_download
        logger.info(f'Downloading model {model_id} from HuggingFace (revision={revision})')
        return hf_snapshot_download(repo_id=model_id, revision=revision)


def resolve_model_path(
    model_name_or_path: str, hub: str = HubType.MODELSCOPE, revision: Optional[str] = 'master'
) -> str:
    """Resolve model name to local path, auto-detecting from cache or downloading.

    Resolution order:
        1. Local filesystem path → use directly
        2. HuggingFace cache → found, use it
        3. ModelScope cache → found, use it
        4. Download from specified hub (HuggingFace uses 'main' branch by default)
    """
    # 1. Local path
    if os.path.exists(model_name_or_path):
        return model_name_or_path

    # 2. Check HF cache
    hf_path = _find_in_cache(model_name_or_path, HF_CACHE)
    if hf_path:
        logger.info(f'Found model {model_name_or_path} in HF cache: {hf_path}')
        return hf_path

    # 3. Check MS cache
    ms_path = _find_in_cache(model_name_or_path, MS_CACHE)
    if ms_path:
        logger.info(f'Found model {model_name_or_path} in ModelScope cache: {ms_path}')
        return ms_path

    # 4. Download — HF defaults to 'main' branch
    if hub != HubType.MODELSCOPE and revision == 'master':
        revision = 'main'
    return download_model(model_name_or_path, revision=revision, hub=hub)
