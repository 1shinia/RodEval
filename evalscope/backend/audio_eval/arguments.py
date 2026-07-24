"""Pydantic configuration classes for Audio evaluation."""
from pydantic import BaseModel, Field
from typing import List, Literal, Optional


class BaseArgument(BaseModel):
    """Base class for configuration arguments."""

    class Config:
        extra = 'allow'


class AudioModelConfig(BaseArgument):
    """Audio model configuration."""

    model_name_or_path: str = Field(..., description='Model path or API model name')
    model_type: Literal['asr', 'tts'] = Field('asr', description='Model type')
    provider: Literal['openai', 'dashscope', 'volcengine'] = Field(
        'openai', description='API provider (openai/dashscope/volcengine)'
    )
    api_base: Optional[str] = Field(None, description='API base URL for remote models')
    api_key: Optional[str] = Field(None, description='API key for remote models')
    device: str = Field('cpu', description='Device to run model on')
    language: str = Field('zh', description='Language code (zh/en/auto)')


class AudioGenerateConfig(BaseArgument):
    """Audio generation parameters."""

    # ASR-specific
    audio_path: Optional[str] = Field(None, description='Path to audio file for ASR')
    reference_text: Optional[str] = Field(None, description='Reference transcription for ASR')

    # TTS-specific
    voice: str = Field('alloy', description='TTS voice preset')
    response_format: str = Field('mp3', description='Audio output format (mp3/wav/ogg)')
    speed: float = Field(1.0, ge=0.25, le=4.0, description='Speech speed multiplier')


class AudioEvalConfig(BaseArgument):
    """Audio evaluation configuration."""

    metrics: List[str] = Field(
        ['wer'],
        description='Evaluation metrics (wer, cer)',
    )
    prompt_dataset: str = Field('builtin', description='Dataset (builtin/custom)')
    prompt_limit: int = Field(1, ge=1, le=1000, description='Number of prompts')
    custom_prompt: Optional[str] = Field(None, description='Custom prompt for TTS')
    custom_dataset_path: Optional[str] = Field(
        None, description='Path to custom audio dataset (one item per line)'
    )
    output_dir: str = Field('', description='Output directory')


class AudioToolConfig(BaseArgument):
    """Top-level Audio tool configuration."""

    tool: Literal['asr', 'tts'] = Field('asr', description='Audio tool type')
    model: AudioModelConfig = Field(..., description='Model configuration')
    generate: AudioGenerateConfig = Field(
        default_factory=AudioGenerateConfig, description='Generation parameters'
    )
    eval: AudioEvalConfig = Field(
        default_factory=AudioEvalConfig, description='Evaluation config'
    )
