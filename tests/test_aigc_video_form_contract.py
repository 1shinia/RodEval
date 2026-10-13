"""文生视频表单与 API payload 的回归契约。"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FORM_PATH = REPO_ROOT / 'evalscope/web/src/components/eval/AIGCEvalForm.tsx'


def _video_form_branch(source: str) -> str:
    start = source.index("{tool === 'txt2video' ? (<>" )
    end = source.index('</>) : (<>' , start)
    return source[start:end]


def test_video_form_uses_duration_instead_of_diffusion_or_frame_controls():
    source = FORM_PATH.read_text(encoding='utf-8')
    branch = _video_form_branch(source)

    assert 'videoDuration' in source
    assert "t('aigc.steps')" not in branch
    assert "t('aigc.guidance')" not in branch
    assert "t('aigc.numFrames')" not in source
    assert "t('aigc.fps')" not in source
    assert "generateConfig.duration = videoDuration === '' ? 5 : Number(videoDuration)" in source
    assert '限支持5～15 秒整数' in source
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
    assert "provider !== 'dashscope'" in source
    assert "provider === 'volcengine'" in source
