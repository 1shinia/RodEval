"""CLIP Score 口径契约：0~1 余弦 + 口径随结果落盘 + 前端标注。

背景：2026-08-17（commit 5082c0e9）把取分从 HF CLIP 的 logits_per_image（≈余弦×logit_scale≈100，
数值 20~40）换成本地中文 CLIP 的归一化点积（纯余弦，0.4~0.6）。口径与模型同时变了，
跨该日期的分数不可直接比较 —— 这里把口径钉在数据里、并在界面上写明。
"""
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FORM_PATH = REPO_ROOT / 'evalscope/web/src/components/eval/AIGCEvalForm.tsx'
MANAGER_PATH = REPO_ROOT / 'evalscope/backend/aigc_eval/backend_manager.py'

from evalscope.backend.aigc_eval.metrics.clip_score import CLIP_SCORE_META  # noqa: E402


def test_meta_declares_cosine_scale_model_and_breakpoint():
    assert CLIP_SCORE_META['scale'] == 'cosine (0~1)'
    assert 'chinese-clip' in CLIP_SCORE_META['model']
    assert '2026-08-17' in CLIP_SCORE_META['note'], '必须写明旧口径分界，否则跨期比较会被误读'
    assert '百分制' in CLIP_SCORE_META['note']


def test_every_aggregation_site_records_the_meta():
    """三条链路（txt2img / txt2video / img2img）都要把口径写进 results.json.metrics。"""
    source = MANAGER_PATH.read_text(encoding='utf-8')
    means = source.count("metrics['clip_score_mean'] = sum(scores) / len(scores)")
    metas = source.count("metrics['clip_score_meta'] = CLIP_SCORE_META")
    imports = source.count('from .metrics.clip_score import CLIP_SCORE_META')
    assert means == metas == imports == 3, f'聚合点 {means} / 落盘 {metas} / 导入 {imports}，应为 3:3:3'


def test_form_shows_plain_metric_label_without_scale_copy():
    """界面只留指标名：口径后缀与说明文案按用户要求删除，口径改由 results.json 承载（见上两条）。"""
    form = FORM_PATH.read_text(encoding='utf-8')
    assert form.count("{ value: 'clip_score', label: 'CLIP Score' }") == 3
    assert '余弦' not in form, '指标标签不得再带口径后缀'
    assert '2026-08 之前的报表为旧口径' not in form, '口径说明文案已按要求删除'
