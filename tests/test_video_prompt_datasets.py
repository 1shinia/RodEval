"""内置视频提示词数据集契约：基线冻结 + 安全版等价降级。

msr_vtt 是历史基线（类别顺序、文本、条数冻结在 2026-10），msr_vtt_safe 是同一数据的
安全投影：条数/类别/句式不变，只把版权·危险·未成年语义的条目换成同类别中性场景，
并把零风险场景前置（prompt_limit 默认取首条 → 默认配置不撞供应商风控）。
"""
import pytest

from evalscope.backend.aigc_eval.datasets.prompt_loader import load_video_prompts

BASELINE_FIRST = 'A soccer player scoring a goal with a powerful kick'
BASELINE_LAST = 'A traditional lion dance performance during a festival'
SAFE_FIRST = 'A waterfall cascading down a rocky cliff in a lush forest'
TOTAL = 80
# 28 条等价降级（职业赛事/军事·灾难·明火/未成年人/演出/婚庆）
DEGRADED = 28
# 降级后仍不得出现的语义（按原风险条目的措辞精确匹配；不扫 lava/festival/flaming 这类
# 单词——「lava field」「lion dance during a festival」「chef tossing vegetables in a flaming wok」
# 是刻意保留的安全场景）。第一组即实测被方舟判 copyright restrictions 的赛事语义。
RISKY_TOKENS = (
    'soccer player', 'slam dunk', 'tennis player', 'marathon runner', 'starting block',
    'boxer', 'rock climber', 'kickflip', 'large wave',
    'eruption', 'river of lava', 'lava flowing', 'tornado', 'sandstorm', 'lightning striking',
    'christmas market', 'weaving through city traffic',
    'fighter jet', 'rocket launching', 'circuit track',
    'children', 'wedding', 'with scissors',
    'concert stage', 'at a festival with colorful lasers', 'flaming torches', 'marching band', 'monologue',
)


def test_msr_vtt_baseline_stays_frozen():
    prompts = load_video_prompts('msr_vtt', limit=1000)
    assert len(prompts) == TOTAL
    assert prompts[0] == BASELINE_FIRST, '基线首条变了：历史分数将不可比'
    assert prompts[-1] == BASELINE_LAST


def test_safe_set_keeps_shape_and_only_degrades_risky_entries():
    baseline = load_video_prompts('msr_vtt', limit=1000)
    safe = load_video_prompts('msr_vtt_safe', limit=1000)

    assert len(safe) == TOTAL, '安全版必须与基线同条数（类别覆盖度不变）'
    assert len(set(safe)) == TOTAL, '安全版不得出现重复条目'
    kept = [prompt for prompt in safe if prompt in baseline]
    assert len(kept) == TOTAL - DEGRADED, '除降级条目外必须逐条与基线同名'


def test_safe_set_leads_with_zero_risk_scene():
    assert load_video_prompts('msr_vtt_safe', limit=1) == [SAFE_FIRST]


@pytest.mark.parametrize('token', RISKY_TOKENS)
def test_safe_set_has_no_rights_or_hazard_semantics(token):
    hits = [prompt for prompt in load_video_prompts('msr_vtt_safe', limit=1000) if token in prompt.lower()]
    assert not hits, f'{token!r} 仍是风险语义：{hits}'


def test_limit_and_shuffle_still_apply_to_safe_set():
    assert len(load_video_prompts('msr_vtt_safe', limit=5)) == 5
    shuffled = load_video_prompts('msr_vtt_safe', limit=1000, shuffle=True, seed=42)
    assert sorted(shuffled) == sorted(load_video_prompts('msr_vtt_safe', limit=1000))


def test_unknown_video_dataset_is_rejected():
    with pytest.raises(ValueError, match='Unknown video dataset'):
        load_video_prompts('activitynet')
