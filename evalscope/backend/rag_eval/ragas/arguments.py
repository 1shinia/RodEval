# Copyright (c) Alibaba, Inc. and its affiliates.
import re

from pydantic import Field, field_validator, model_validator
from typing import List, Literal, Optional

from evalscope.utils.argument_utils import BaseArgument


SUPPORTED_METRICS = {
    'answer_relevancy': 'AnswerRelevancy',
    'faithfulness': 'Faithfulness',
    'context_precision': 'ContextPrecision',
    'context_recall': 'ContextRecall',
    'answer_correctness': 'AnswerCorrectness',
}


def normalize_metric_name(name: str) -> str:
    name = re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()
    if name not in SUPPORTED_METRICS:
        raise ValueError(f'Unsupported RAGAS metric: {name}')
    return name


class RAGASLLMConfig(BaseArgument):
    """RAGAS LLM configuration - adapts to ragas 0.4.x llm_factory() pattern."""

    model_name: str
    provider: str = 'openai'
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = 0.0
    max_tokens: Optional[int] = None


class RAGASEmbeddingConfig(BaseArgument):
    """RAGAS Embedding configuration - adapts to ragas 0.4.x embedding_factory()."""

    model_name_or_path: str
    provider: str = 'huggingface'
    api_base: Optional[str] = None
    api_key: Optional[str] = None


class RAGASEvalConfig(BaseArgument):
    """RAGAS evaluation configuration - adapts to 0.4.x aevaluate() API."""

    testset_file: str
    critic_llm: RAGASLLMConfig
    embeddings: Optional[RAGASEmbeddingConfig] = None
    metrics: List[str] = Field(default=['answer_relevancy', 'faithfulness'])
    language: str = 'english'
    batch_size: Optional[int] = Field(default=None, gt=0)
    raise_exceptions: bool = False

    @field_validator('metrics')
    @classmethod
    def validate_metrics(cls, value):
        if not value:
            raise ValueError('At least one RAGAS metric is required.')
        return list(dict.fromkeys(normalize_metric_name(name) for name in value))

    @model_validator(mode='after')
    def validate_evaluation_models(self):
        # Evaluation-specific validation must not alter testset generation.
        if self.critic_llm.provider != 'openai':
            raise ValueError('RAGAS evaluation LLM requires the openai-compatible provider.')
        if not self.critic_llm.model_name.strip():
            raise ValueError('RAGAS critic model name cannot be blank.')
        if self.embeddings is not None:
            if self.embeddings.provider not in ('openai', 'huggingface'):
                raise ValueError('Unsupported RAGAS embedding provider.')
            if not self.embeddings.model_name_or_path.strip():
                raise ValueError('RAGAS embedding model name cannot be blank.')
        if any(name in ('answer_relevancy', 'answer_correctness') for name in self.metrics):
            if self.embeddings is None:
                raise ValueError('Selected RAGAS metrics require an embedding model.')
        return self

    @field_validator('critic_llm', mode='before')
    @classmethod
    def parse_critic_llm(cls, v):
        if isinstance(v, dict):
            return RAGASLLMConfig(**v)
        return v

    @field_validator('embeddings', mode='before')
    @classmethod
    def parse_embeddings(cls, v):
        if isinstance(v, dict):
            return RAGASEmbeddingConfig(**v)
        return v


class RAGASTestsetConfig(BaseArgument):
    """RAGAS testset generation configuration."""

    docs: List[str]
    test_size: int = 10
    output_file: str = 'outputs/testset.json'
    generator_llm: RAGASLLMConfig
    embeddings: RAGASEmbeddingConfig
    language: str = 'english'

    @field_validator('generator_llm', mode='before')
    @classmethod
    def parse_generator_llm(cls, v):
        if isinstance(v, dict):
            return RAGASLLMConfig(**v)
        return v

    @field_validator('embeddings', mode='before')
    @classmethod
    def parse_embeddings(cls, v):
        if isinstance(v, dict):
            return RAGASEmbeddingConfig(**v)
        return v


class RAGASToolConfig(BaseArgument):
    """Complete configuration for tool='ragas' in eval_config."""

    tool: Literal['ragas'] = 'ragas'
    testset_generation: Optional[RAGASTestsetConfig] = None
    eval: Optional[RAGASEvalConfig] = None

    @field_validator('tool', mode='before')
    @classmethod
    def normalize_tool(cls, v):
        return v.lower() if isinstance(v, str) else v

    @field_validator('testset_generation', mode='before')
    @classmethod
    def parse_testset_generation(cls, v):
        if isinstance(v, dict):
            return RAGASTestsetConfig(**v)
        return v

    @field_validator('eval', mode='before')
    @classmethod
    def parse_eval(cls, v):
        if isinstance(v, dict):
            return RAGASEvalConfig(**v)
        return v
