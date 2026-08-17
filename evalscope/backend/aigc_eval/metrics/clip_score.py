"""CLIP Score metric for text-image similarity.

Uses the locally-cached Damoso Chinese-CLIP model
(``AI-ModelScope/chinese-clip-vit-base-patch16``), which is better suited for
Chinese prompts than the English ``openai/clip-vit-base-patch32``.

Note on the manual text encoding: transformers' ``ChineseCLIPModel.get_text_features``
assumes ``text_model`` has a BertPooler (``add_pooling_layer=True``). This model's
weights contain no pooler, so calling ``get_text_features`` raises
``TypeError: linear(): argument 'input' must be Tensor, not NoneType``. We therefore
encode text manually: take the ``[CLS]`` token of the last hidden state and pass it
through ``text_projection`` — equivalent to standard CLIP text encoding.
"""
import logging
import os
import torch
import torch.nn.functional as F
from PIL import Image
from typing import List, Optional

logger = logging.getLogger(__name__.replace('evalscope', 'evalperf'))

# 默认使用本地已缓存的中文 CLIP 模型（对中文 prompt 语义匹配更好，文本支持 512 token）
_DEFAULT_MODEL_ID = 'AI-ModelScope/chinese-clip-vit-base-patch16'

# 已知本地缓存路径（modelscope 缓存根被自定义到 /data）
_KNOWN_LOCAL_PATHS = (
    '/data/modelscope/hub/models/AI-ModelScope/chinese-clip-vit-base-patch16',
    os.path.expanduser('~/.cache/modelscope/hub/models/AI-ModelScope/chinese-clip-vit-base-patch16'),
)

# 模块级缓存：一次评估内多个 sample 共享同一模型，避免重复加载 753MB 权重
_model = None
_processor = None
_model_device = None


def _resolve_model_path(model_name: str) -> str:
    """Resolve a model id / local dir to an on-disk directory."""
    if os.path.isdir(model_name):
        return model_name
    for p in _KNOWN_LOCAL_PATHS:
        if os.path.isdir(p):
            return p
    # 兜底：modelscope snapshot_download（命中缓存则本地，否则联网下载）
    try:
        from modelscope import snapshot_download
        return snapshot_download(model_id=model_name, revision='master')
    except Exception as e:
        raise RuntimeError(f'Failed to resolve CLIP model path for {model_name}: {e}') from e


def _load_model(model_name: str, device: str):
    """Load (and cache) the CLIP model + processor for the given device."""
    global _model, _processor, _model_device
    if _model is not None and _processor is not None and _model_device == device:
        return _model, _processor

    from transformers import AutoModel, AutoProcessor

    path = _resolve_model_path(model_name)
    logger.info('Loading CLIP model from: %s', path)
    _processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
    _model = AutoModel.from_pretrained(path, trust_remote_code=True)
    _model.eval()
    _model = _model.to(device)
    _model_device = device
    return _model, _processor


def compute_clip_score(
    images: List[Image.Image],
    prompts: List[str],
    model_name: str = _DEFAULT_MODEL_ID,
    device: Optional[str] = None,
) -> List[float]:
    """Compute CLIP Score (cosine similarity) between images and text prompts.

    Args:
        images: List of PIL Images (one per prompt)
        prompts: List of text prompts
        model_name: CLIP model id or local dir
        device: Device to run on (auto-detected if None)

    Returns:
        List of CLIP scores (cosine similarity, -1..1)
    """
    if len(images) != len(prompts):
        raise ValueError(f'Number of images ({len(images)}) != number of prompts ({len(prompts)})')

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model, processor = _load_model(model_name, device)

    scores: List[float] = []
    with torch.no_grad():
        for img, prompt in zip(images, prompts):
            # 文本编码：BERT tokenizer，max 512 + truncation 处理长中文 prompt
            text_inputs = processor(
                text=[prompt],
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512,
            )
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

            # 手动 [CLS] token + text_projection（等价于标准 CLIP 文本编码）
            text_out = model.text_model(
                input_ids=text_inputs['input_ids'],
                attention_mask=text_inputs['attention_mask'],
                return_dict=True,
            )
            text_feat = model.text_projection(text_out.last_hidden_state[:, 0, :])
            text_feat = F.normalize(text_feat, dim=-1)

            image_inputs = processor(images=img, return_tensors='pt')
            image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
            image_feat = model.get_image_features(pixel_values=image_inputs['pixel_values'])
            image_feat = F.normalize(image_feat, dim=-1)

            sim = (text_feat @ image_feat.T).item()
            scores.append(float(sim))

    logger.info('Computed CLIP scores for %d samples', len(scores))
    return scores
