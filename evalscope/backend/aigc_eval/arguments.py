"""Pydantic configuration classes for AIGC evaluation."""
from pydantic import BaseModel, Field
from typing import List, Literal, Optional


class BaseArgument(BaseModel):
    """Base class for configuration arguments."""

    class Config:
        extra = 'allow'


class AIGCModelConfig(BaseArgument):
    """AIGC model configuration."""

    model_name_or_path: str = Field(..., description='Model path or API model name')
    model_type: Literal['txt2img', 'txt2video', 'img2img'] = Field('txt2img', description='Model type')
    api_base: Optional[str] = Field(None, description='API base URL for remote models')
    api_key: Optional[str] = Field(None, description='API key for remote models')
    device: str = Field('cuda', description='Device to run model on')
    dtype: str = Field('float16', description='Model precision (float16/float32/bfloat16)')


class AIGCGenerateConfig(BaseArgument):
    """AIGC generation parameters."""

    width: int = Field(512, ge=256, le=2048, description='Image width')
    height: int = Field(512, ge=256, le=2048, description='Image height')
    num_inference_steps: int = Field(50, ge=1, le=200, description='Number of inference steps')
    guidance_scale: float = Field(7.5, ge=1.0, le=30.0, description='Guidance scale (CFG)')
    negative_prompt: str = Field('', description='Negative prompt')
    seed: int = Field(42, description='Random seed')
    batch_size: int = Field(1, ge=1, le=8, description='Batch size')
    # Video-specific parameters
    num_frames: int = Field(16, ge=1, le=128, description='Number of frames (video only)')
    fps: int = Field(8, ge=1, le=60, description='Frames per second (video only)')


class AIGCEvalConfig(BaseArgument):
    """AIGC evaluation configuration."""

    metrics: List[str] = Field(
        ['clip_score'],
        description='Evaluation metrics to compute (clip_score, fid, is, lpips)',
    )
    prompt_dataset: str = Field('drawbench', description='Prompt dataset (drawbench, coco_captions, parti, custom)')
    prompt_limit: int = Field(100, ge=1, le=10000, description='Number of prompts to evaluate')
    reference_dir: Optional[str] = Field(None, description='Reference image directory for FID calculation')
    reference_video_dir: Optional[str] = Field(None, description='Reference video directory for FVD calculation')
    custom_dataset_path: Optional[str] = Field(None, description='Path to custom prompt file (one prompt per line)')
    output_dir: str = Field('', description='Output directory for generated media')


class AIGCToolConfig(BaseArgument):
    """Top-level AIGC tool configuration."""

    tool: Literal['txt2img', 'txt2video', 'img2img'] = Field('txt2img', description='AIGC tool type')
    model: AIGCModelConfig = Field(..., description='Model configuration')
    generate: AIGCGenerateConfig = Field(default_factory=AIGCGenerateConfig, description='Generation parameters')
    eval: AIGCEvalConfig = Field(default_factory=AIGCEvalConfig, description='Evaluation config')
