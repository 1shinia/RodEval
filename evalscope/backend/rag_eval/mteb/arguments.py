# Copyright (c) Alibaba, Inc. and its affiliates.
from pydantic import Field, field_validator, model_validator
from typing import Any, Dict, List, Literal, Optional

from evalscope.utils.argument_utils import BaseArgument


class CustomTaskConfig(BaseArgument):
    """Configuration for a custom MTEB task."""
    name: str = 'CustomRetrieval'
    data_path: str
    eval_splits: List[str] = Field(default_factory=lambda: ['test'])


class MTEBModelConfig(BaseArgument):
    """MTEB model configuration."""

    model_name_or_path: str = ''
    is_cross_encoder: bool = False
    pooling_mode: Optional[str] = None
    max_seq_length: int = 512
    prompt: Optional[str] = None
    prompts: Optional[Dict[str, str]] = None
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    encode_kwargs: Dict[str, Any] = Field(default_factory=lambda: {'batch_size': 20})
    hub: str = 'modelscope'
    # API model fields
    model_name: Optional[str] = None
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    dimensions: Optional[int] = None

    @model_validator(mode='before')
    @classmethod
    def validate_model_source(cls, data):
        if not isinstance(data, dict):
            return data

        normalized = dict(data)
        for field in ('model_name', 'model_name_or_path', 'api_base', 'api_key'):
            value = normalized.get(field)
            if isinstance(value, str):
                normalized[field] = value.strip()

        # Keep the established compatibility rule: model_name wins when both
        # fields are present, so older API configs remain API configs.
        if not normalized.get('model_name_or_path') and normalized.get('model_name'):
            normalized['model_name_or_path'] = normalized['model_name']

        if normalized.get('model_name'):
            if not normalized.get('api_base'):
                raise ValueError('api_base is required when model_name is set for an API model.')
        elif not normalized.get('model_name_or_path'):
            raise ValueError('model_name_or_path is required for a local MTEB model.')

        return normalized


class MTEBEvalConfig(BaseArgument):
    """MTEB evaluation configuration."""

    task_names: Optional[List[str]] = None
    task_types: Optional[List[str]] = None
    languages: Optional[List[str]] = None
    custom_tasks: Optional[List[CustomTaskConfig]] = None

    @field_validator('custom_tasks', mode='before')
    @classmethod
    def parse_custom_tasks(cls, v):
        if isinstance(v, list):
            return [CustomTaskConfig(**item) if isinstance(item, dict) else item for item in v]
        return v

    output_folder: str = 'outputs'
    overwrite_results: bool = True
    limits: Optional[int] = None
    random_sample: bool = False
    hub: str = 'modelscope'
    # Server-side policy flags. The service sets these for normal users;
    # they are consumed by the MTEB runner and never trusted from the client.
    offline: bool = False
    allow_download: bool = True
    top_k: int = 10
    splits: Optional[Dict[str, Any]] = None
    encode_kwargs: Optional[Dict[str, Any]] = None


class MTEBToolConfig(BaseArgument):
    """Complete configuration for tool='mteb' in eval_config."""

    tool: Literal['mteb'] = 'mteb'
    models: List[MTEBModelConfig]
    eval: MTEBEvalConfig

    @field_validator('tool', mode='before')
    @classmethod
    def normalize_tool(cls, v):
        return v.lower() if isinstance(v, str) else v

    @field_validator('models', mode='before')
    @classmethod
    def parse_models(cls, v):
        if isinstance(v, list):
            return [MTEBModelConfig(**m) if isinstance(m, dict) else m for m in v]
        return v

    @field_validator('eval', mode='before')
    @classmethod
    def parse_eval(cls, v):
        if isinstance(v, dict):
            return MTEBEvalConfig(**v)
        return v
