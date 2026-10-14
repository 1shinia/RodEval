"""Pydantic configuration classes for AIGC evaluation."""
from pydantic import BaseModel, Field, model_validator
from typing import List, Literal, Optional


class BaseArgument(BaseModel):
    """Base class for configuration arguments."""

    class Config:
        extra = 'allow'


class AIGCModelConfig(BaseArgument):
    """AIGC model configuration."""

    model_name_or_path: str = Field(..., description='Model path or API model name')
    model_type: Literal['txt2img', 'txt2video', 'img2img'] = Field('txt2img', description='Model type')
    provider: Optional[str] = Field(
        None,
        description='API provider. None for local models, "openai" for OpenAI-compatible, '
        '"custom" for custom endpoint mapping, "dashscope" for Aliyun DashScope '
        'video generation (kling/wanx), or "volcengine" for Volcengine Ark '
        'Seedance video generation',
    )
    api_base: Optional[str] = Field(None, description='API base URL for remote models')
    api_key: Optional[str] = Field(None, description='API key for remote models')
    device: str = Field('cuda', description='Device to run model on')
    dtype: str = Field('float16', description='Model precision (float16/float32/bfloat16)')
    # Advanced API overrides (for custom provider)
    endpoint_template: Optional[str] = Field(
        None,
        description='Custom endpoint path template, e.g. /video/generations. '
        'Use "passthrough" to skip path appending entirely.',
    )
    param_aliases: Optional[dict] = Field(
        None,
        description='JSON dict mapping standard param names to provider-specific names, '
        'e.g. {"duration": "seconds"}',
    )
    response_path: Optional[str] = Field(
        None,
        description='JMESPath-style response extraction path, '
        'e.g. data[0].b64_json. If unset, auto-detect OpenAI format.',
    )
    async_poll_url: Optional[str] = Field(
        None,
        description='Async task polling URL template, e.g. /videos/{id}',
    )
    async_content_url: Optional[str] = Field(
        None,
        description='Async task content download URL template, e.g. /videos/{id}/content',
    )


class AIGCGenerateConfig(BaseArgument):
    """AIGC generation parameters."""

    width: int = Field(512, ge=256, le=2048, description='Image width')
    height: int = Field(512, ge=256, le=2048, description='Image height')
    num_inference_steps: Optional[int] = Field(50, ge=1, le=200, description='Number of inference steps')
    guidance_scale: Optional[float] = Field(7.5, ge=1.0, le=30.0, description='Guidance scale (CFG)')
    negative_prompt: str = Field('', description='Negative prompt')
    seed: Optional[int] = Field(42, description='Random seed')

    # Video-specific parameters
    num_frames: int = Field(16, ge=1, le=128, description='Number of frames (video only)')
    fps: int = Field(8, ge=1, le=60, description='Frames per second (video only)')
    resolution: Optional[str] = Field(None, description='Video resolution: 480p/720p/1080p/4k')
    ratio: Optional[str] = Field(None, description='Video aspect ratio: 16:9/9:16/4:3/1:1')
    duration: int = Field(5, ge=1, le=60, description='Video duration in seconds (video only)')


class AIGCEvalConfig(BaseArgument):
    """AIGC evaluation configuration."""

    metrics: List[str] = Field(
        ['clip_score'],
        description='Evaluation metrics to compute (clip_score, fid, is, lpips)',
    )
    prompt_dataset: str = Field(
        'drawbench', description='Prompt dataset (drawbench, coco_captions, parti, custom, msr_vtt, msr_vtt_safe)'
    )
    prompt_limit: int = Field(1, ge=1, le=10000, description='Number of prompts to evaluate')
    custom_prompt: Optional[str] = Field(None, description='Custom prompt for img2img (takes priority over dataset)')
    random_prompt: bool = Field(False, description='Randomly shuffle prompts before taking limit')
    reference_dir: Optional[str] = Field(None, description='Reference image directory for FID calculation')
    custom_dataset_path: Optional[str] = Field(None, description='Path to custom prompt file (one prompt per line)')
    output_dir: str = Field('', description='Output directory for generated media')


class AIGCToolConfig(BaseArgument):
    """Top-level AIGC tool configuration."""

    @model_validator(mode='before')
    @classmethod
    def image_api_defaults(cls, data):
        if not isinstance(data, dict):
            return data
        tool = data.get('tool', 'txt2img')
        model = data.get('model', {})
        if tool == 'txt2img' and isinstance(model, dict) and model.get('api_base'):
            data = dict(data)
            generate = dict(data.get('generate') or {})
            for name in ('num_inference_steps', 'guidance_scale', 'seed'):
                generate.setdefault(name, None)
            data['generate'] = generate
        elif tool != 'txt2img':
            generate = data.get('generate', {})
            if isinstance(generate, dict) and any(
                name in generate and generate[name] is None
                for name in ('num_inference_steps', 'guidance_scale', 'seed')
            ):
                raise ValueError('非文生图模式的步数、引导系数和种子不能为 null')
        return data

    tool: Literal['txt2img', 'txt2video', 'img2img'] = Field('txt2img', description='AIGC tool type')
    model: AIGCModelConfig = Field(..., description='Model configuration')
    generate: AIGCGenerateConfig = Field(default_factory=AIGCGenerateConfig, description='Generation parameters')
    eval: AIGCEvalConfig = Field(default_factory=AIGCEvalConfig, description='Evaluation config')
