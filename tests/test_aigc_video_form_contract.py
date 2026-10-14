"""文生视频表单与 API payload 的回归契约。"""
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FORM_PATH = REPO_ROOT / 'evalscope/web/src/components/eval/AIGCEvalForm.tsx'
ADAPTERS_PATH = REPO_ROOT / 'evalscope/backend/aigc_eval/models/adapters.py'
PROMPT_LOADER_PATH = REPO_ROOT / 'evalscope/backend/aigc_eval/datasets/prompt_loader.py'


def _video_form_branch(source: str) -> str:
    start = source.index("{tool === 'txt2video' ? (<>")
    end = source.index('</>) : (<>', start)
    return source[start:end]


def test_video_form_uses_duration_instead_of_diffusion_or_frame_controls():
    source = FORM_PATH.read_text(encoding='utf-8')
    branch = _video_form_branch(source)

    assert 'videoDuration' in source
    assert "t('aigc.steps')" not in branch
    assert "t('aigc.guidance')" not in branch
    assert "t('aigc.numFrames')" not in source
    assert "t('aigc.fps')" not in source
    assert "generateConfig.duration = videoDurationValue === '' ? 5 : Number(videoDurationValue)" in source
    # 时长与分辨率一样按模型给可选项（下拉）；只有无法枚举的 OpenAI 兼容分支保留自由输入
    assert 'const intRange = (low: number, high: number)' in source
    assert 'durationOptions.length > 0 ? (' in source
    assert 'durations: []' in source
    for span in ('intRange(4, 30)', 'intRange(4, 15)', 'intRange(2, 12)', 'intRange(2, 15)', 'intRange(3, 15)'):
        assert span in source, span
    assert "['5', '10']" in source  # wan2.5 只支持 5 或 10 秒
    # 火山的时长/分辨率按模型分档（严格照官方模型列表），不再是写死的 5~15
    assert 'volcengineCapability(model)' in source
    assert '限支持4～30 秒整数' in source
    assert '限支持4～15 秒整数' in source
    assert '限支持2～12 秒整数' in source
    # 5~15 只允许出现在 OpenAI 兼容分支（透传语义），火山分支里不得再有
    volc_ctx = source[source.index('function volcengineCapability'):source.index('function videoCapability')]
    assert '5～15' not in volc_ctx
    assert 'min={videoDurationMin}' not in source
    assert 'max={videoDurationMax}' not in source
    assert 'generateConfig.resolution = videoResolution' in source
    assert 'generateConfig.ratio = videoAspectRatio' in source
    assert 'generateConfig.num_frames' not in source
    assert 'generateConfig.fps' not in source


def test_video_provider_specific_controls_are_scoped_to_supported_providers():
    source = FORM_PATH.read_text(encoding='utf-8')

    assert "value: 'custom', label: '自定义'" not in source
    assert "useState<'openai' | 'custom'" not in source
    assert 'endpointTemplate' not in source
    assert 'paramAliases' not in source
    assert 'responsePath' not in source
    assert 'asyncPollUrl' not in source
    assert 'asyncContentUrl' not in source
    assert "provider === 'volcengine'" in source
    # 分辨率/种子不再按 provider 硬隐藏，改由能力表决定
    assert "provider !== 'dashscope'" not in source
    assert 'videoCapability(provider, model)' in source
    assert 'videoCap.seed' in source
    # 负向提示词字段已移除（并入「自定义提示词」，后端仍兼容 negative_prompt 透传）
    assert 'negativePrompt' not in source
    assert 'negative_prompt' not in source
    # FVD 已整体下线（值恒为 0、无参考视频集入口、无任何成功记录）
    assert 'fvd' not in source
    assert '参考视频目录' not in source
    # 百炼地址 + OpenAI 兼容提供商 = 必 404（实测 /video/generations），前端必须拦下
    assert 'needsDashscopeProvider(provider, apiBase)' in source
    assert '把「API 提供商」选为 DashScope' in source
    # compatible-mode 是 OpenAI 兼容入口，无视频合成端点（实测 404），必须提示改用 /api/v1
    assert 'compatibleModeVideoBase(apiBase)' in source
    assert 'compatible-mode 是 OpenAI 兼容入口' in source


def test_video_prompt_dataset_defaults_to_safe_set():
    """文生视频默认走安全版内置提示词。

    默认 prompt_limit=1 只取首条 → 首条必须零风险，否则一句话就撞供应商风控、整任务失败
    （实测原版首条 'A soccer player scoring...' 被方舟判 copyright restrictions）。
    """
    source = FORM_PATH.read_text(encoding='utf-8')
    loader = PROMPT_LOADER_PATH.read_text(encoding='utf-8')

    assert "{ value: 'msr_vtt_safe', label: '内置视频提示词·安全版 (80 prompts)' }" in source
    assert "{ value: 'msr_vtt', label: '内置视频提示词·原版 (80 prompts)' }" in source
    assert "setPromptDataset('msr_vtt_safe')" in source
    assert "setPromptDataset('msr_vtt')" not in source, '默认值不得回退到基线集合'
    assert "'msr_vtt_safe': _SAFE_VIDEO_CATEGORY_ORDER" in loader
    # 下拉里每个内置值都必须有后端分支，避免选了就 ValueError
    for value in sorted(set(re.findall(r"value: '(msr_vtt[a-z_]*)'", source))):
        assert f"'{value}':" in loader, f'后端缺少数据集分支 {value}'


def test_video_dataset_dropdown_has_only_builtin_sets():
    """视频数据集下拉只留两个内置集合。

    单条提示词已有「自定义提示词」入口（其优先级高于数据集），所以「自定义数据集 /
    不使用（仅用自定义提示词）」是冗余入口，前端删除；后端保留 custom/空值分支，
    以便复现历史任务配置与 API 直连调用。
    """
    source = FORM_PATH.read_text(encoding='utf-8')
    loader = PROMPT_LOADER_PATH.read_text(encoding='utf-8')
    block = source[source.index('const VIDEO_DATASETS'):source.index('const TEXT_TO_IMAGE_METRICS')]

    assert "'msr_vtt_safe'" in block and "'msr_vtt'" in block
    assert "value: 'custom'" not in block
    assert "value: ''" not in block
    assert "dataset_name == 'custom'" in loader, '后端兼容分支不得一并删除'


def test_volcengine_tiers_match_backend_capability_table():
    """前端火山下拉档位必须来自后端 _VOLCENGINE_VIDEO_FAMILIES（同一份事实来源）。

    后端按官方文档收录完整档位（含 Seedance 2.0 的 4k）；前端按产品决定暂不展示 4k
    （官方 4k 并发仅 1），因此前端档位只允许是后端某族的**顺序前缀**。
    """
    form = FORM_PATH.read_text(encoding='utf-8')
    adapters = ADAPTERS_PATH.read_text(encoding='utf-8')
    volc_ctx = form[form.index('function volcengineCapability'):form.index('function videoCapability')]
    table = adapters[adapters.index('_VOLCENGINE_VIDEO_FAMILIES'):adapters.index('def _volcengine_video_family')]

    backend_tiers = [re.findall(r"'([^']+)'", group) for group in re.findall(r"'tiers': \(([^)]*)\)", table)]
    frontend_tiers = [re.findall(r"'([^']+)'", group) for group in re.findall(r'build\(\[([^\]]*)\]', volc_ctx)]

    assert backend_tiers, '后端火山能力表未找到 tiers'
    assert frontend_tiers, '前端未找到 build([...]) 档位'
    for tiers in frontend_tiers:
        assert tiers, '前端档位不能为空'
        assert any(tiers == backend[:len(tiers)] for backend in backend_tiers), \
            f'前端档位 {tiers} 不是任何后端族的顺序前缀'

    # 时长同样按模型分档：后端区间必须在前端以 intRange(low, high) 出现
    backend_ranges = re.findall(r"'duration': \((\d+), (\d+)\)", table)
    assert backend_ranges, '后端火山能力表未找到 duration'
    for low, high in backend_ranges:
        assert f'intRange({low}, {high})' in volc_ctx, f'前端缺少后端时长区间 {low}~{high}'

    # 画幅比同源：前端 VOLC_RATIOS 必须与后端 _VOLCENGINE_RATIOS 完全一致
    backend_ratios = re.search(r'_VOLCENGINE_RATIOS = \(([^)]*)\)', adapters)
    assert backend_ratios, '后端未定义 _VOLCENGINE_RATIOS'
    values = re.findall(r"'([^']+)'", backend_ratios.group(1))
    literal = 'const VOLC_RATIOS = [' + ', '.join(f"'{v}'" for v in values) + ']'
    assert literal in form, f'前端画幅比与后端不一致，期望 {literal}'

    # 4k 是官方仅 2.0 支持的档位，前端按产品决定暂不展示
    assert any('4k' in tiers for tiers in backend_tiers)
    assert not any('4k' in tiers for tiers in frontend_tiers)


def test_video_capability_table_covers_dashscope_families():
    source = FORM_PATH.read_text(encoding='utf-8')

    # 档位与时长语义必须与后端 _DASHSCOPE_VIDEO_FAMILIES 对齐（前端展示 / 后端校验）
    for tiers in ("['720p', '1080p']", "['480p', '720p', '1080p']",
                  "['480p', '1080p']", "['480p', '720p']", "['720p']"):
        assert tiers in source
    for hint in ('限支持2～15 秒整数', '该模型只支持 5 或 10 秒',
                 '该模型时长固定为 5 秒', '限支持3～15 秒整数'):
        assert hint in source
    assert "name.startsWith('kling')" in source
    assert "name.includes('t2v-turbo')" in source
    assert "name.includes('t2v-plus')" in source
    assert 'known: false' in source
    assert "'3:4'" in source  # 3:4 画幅比入口
