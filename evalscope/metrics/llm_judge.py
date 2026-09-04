import os
import re
import time
from typing import Any, Dict, List, Optional

from evalscope.api.messages import ChatMessage, ChatMessageSystem, ChatMessageUser
from evalscope.constants import EvalType, JudgeScoreType
from evalscope.utils.logger import get_logger

logger = get_logger()

DEFAULT_PROMPT_TEMPLATE = """Your job is to look at a question, a gold target, and a predicted answer, and return a letter "A" or "B" to indicate whether the predicted answer is correct or incorrect.

[Question]
{question}

[Reference Answer]
{gold}

[Predicted Answer]
{pred}

Evaluate the model's answer based on correctness compared to the reference answer.
Grade the predicted answer of this new question as one of:
A: CORRECT
B: INCORRECT

Just return the letters "A" or "B", with no text around it.
"""  # noqa: E501


DEFAULT_NUMERIC_SCORE_TEMPLATE = """Please act as an impartial judge and evaluate the correctness and quality of the response provided by an AI assistant to the user question displayed below. Use the reference answer as the ground truth. Your evaluation should prioritize factual correctness, then consider relevance, completeness, and clarity.
Begin your evaluation by providing a short explanation. Be as objective as possible.
After providing your explanation, you must rate the response on a scale of 0 (worst) to 1 (best) by strictly following this format: \"[[rating]]\", for example: \"Rating: [[0.5]]\"

[Question]
{question}

[Reference Answer]
{gold}

[Response]
{pred}
"""  # noqa: E501

DEFAULT_JUDGE_MODEL = 'Qwen/Qwen3-235B-A22B'
DEFAULT_API_URL = 'https://api-inference.modelscope.cn/v1/'


class LLMJudgeError(RuntimeError):
    """Base error for judge transport/generation failures."""


class LLMJudgeParseError(LLMJudgeError):
    """Raised when a judge response cannot be parsed into a valid score."""


class LLMJudge:
    """
    A metric that uses LLM to judge the quality of model predictions by comparing them with reference answers.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        model_id: Optional[str] = None,
        eval_type: Optional[str] = None,
        model_args: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        prompt_template: Optional[str] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        score_pattern: Optional[str] = None,
        score_mapping: Optional[Dict[str, float]] = None,
        score_type: str = JudgeScoreType.PATTERN,  # 'pattern', 'numeric'
        max_retries: int = 2,
        retry_backoff: float = 1.0,
        **kwargs
    ):
        """
        Initialize LLMJudge metric.

        Args:
            api_key (str, optional): API key for OpenAI or compatible service
            api_base (str, optional): API base URL
            model_id (str, optional): Model ID for LLM
            eval_type (str, optional): Evaluation LLM type for the judge
            model_args (dict, optional): Additional model arguments for the judge
            system_prompt (str, optional): System prompt for the judge
            prompt_template (str, optional): Prompt template for the judge
            generation_config (dict, optional): Generation configuration for the judge
            score_pattern (str, optional): Regex pattern to extract score from LLM response
            score_mapping (dict, optional): Mapping from extracted score to float value
            score_type (str, optional): Type of score extraction strategy ('pattern', 'numeric') defaults to 'pattern'.
                - 'pattern': Use score_pattern and score_mapping to extract categorical scores
                - 'numeric': Treat the extracted value as a direct numerical score
        """
        self.api_key = api_key or os.environ.get('MODELSCOPE_SDK_TOKEN', 'EMPTY')
        self.api_url = api_url or os.environ.get('MODELSCOPE_API_BASE', DEFAULT_API_URL)
        self.model_id = model_id or os.environ.get('MODELSCOPE_JUDGE_LLM', DEFAULT_JUDGE_MODEL)
        self.eval_type = eval_type or EvalType.OPENAI_API
        self.system_prompt = system_prompt or os.environ.get('JUDGE_SYSTEM_PROMPT', None)
        self.generation_config = generation_config or {'temperature': 0.0, 'max_tokens': 8192, 'do_sample': False}
        self.model_args = model_args or {}
        self.max_retries = max(0, int(max_retries))
        self.retry_backoff = max(0.0, float(retry_backoff))

        # Default score mapping for A/B pattern
        self.score_type = score_type
        if self.score_type == JudgeScoreType.NUMERIC:
            self.score_pattern = score_pattern or r'\[\[(\d+(?:\.\d+)?)\]\]'
            self.prompt_template = prompt_template or os.environ.get(
                'JUDGE_PROMPT_TEMPLATE', DEFAULT_NUMERIC_SCORE_TEMPLATE
            )
        elif self.score_type == JudgeScoreType.PATTERN:
            # Anchor to only accept a standalone A or B (avoid false positives)
            self.score_pattern = score_pattern or r'^\s*([AB])\s*$'
            self.prompt_template = prompt_template or os.environ.get('JUDGE_PROMPT_TEMPLATE', DEFAULT_PROMPT_TEMPLATE)
        else:
            raise ValueError(f"Invalid score_type: {self.score_type}. Must be 'pattern' or 'numeric'.")
        self.score_mapping = score_mapping or {'A': 1.0, 'B': 0.0}

        self._init_server_adapter()

    def _init_server_adapter(self):
        from evalscope.api.model import GenerateConfig, get_model

        self.model = get_model(
            model=self.model_id,
            eval_type=self.eval_type,
            base_url=self.api_url,
            api_key=self.api_key,
            config=GenerateConfig(**self.generation_config),
            model_args=self.model_args,
        )

    def judge(
        self,
        prompt: str = '',
        system_prompt: Optional[str] = None,
        messages: Optional[List[ChatMessage]] = None
    ) -> str:
        """
        Generate a response from the LLM based on the provided prompt and context.
        If messages is provided, it will be used as the input context.

        Args:
            prompt (str): The prompt to evaluate
            system_prompt (str, optional): The system prompt to use for the evaluation
            messages (List[ChatMessage], optional): A list of chat messages to include in the evaluation
        Returns:
            str: The response from the LLM
        """
        # parse messages
        if messages is not None:
            input_messages = messages
        else:
            system_content = system_prompt or self.system_prompt
            input_messages = [ChatMessageUser(content=prompt)]
            if system_content:
                input_messages.insert(0, ChatMessageSystem(content=system_content))
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.model.generate(input_messages)
                llm_response = response.completion
                if llm_response is None:
                    raise LLMJudgeError('Judge returned an empty completion.')
                return llm_response
            except Exception as e:
                last_error = e
                error_message = f'Error occurred during {self.model_id}@{self.api_url} LLM judge evaluation: {e}'
                logger.error(error_message)
                if attempt < self.max_retries and self.retry_backoff > 0:
                    time.sleep(self.retry_backoff * (2**attempt))

        raise LLMJudgeError(
            f'LLM judge failed after {self.max_retries + 1} attempt(s): {last_error}'
        ) from last_error

    def build_prompt(self, pred: str, gold: str, question: Optional[str] = None):
        if question is None:
            question = 'Not provided'

        # check variables in prompt_template
        prompt = self.prompt_template
        if '{question}' in self.prompt_template:
            prompt = prompt.replace('{question}', question)
        if '{pred}' in self.prompt_template:
            prompt = prompt.replace('{pred}', pred)
        if '{gold}' in self.prompt_template:
            prompt = prompt.replace('{gold}', gold)
        return prompt

    def get_score(self, response: str) -> float:
        """
        Extract score from LLM response using the configured pattern and mapping.

        Args:
            response (str): The response from the LLM

        Returns:
            float: The numeric score extracted from the response
        """
        if response is None:
            raise LLMJudgeParseError('Judge response is None; cannot extract a score.')

        # choose extraction method based on score_type
        if self.score_type == JudgeScoreType.NUMERIC:
            return self._extract_numeric_score(response)
        elif self.score_type == JudgeScoreType.PATTERN:
            return self._extract_pattern_score(response)

    def _extract_numeric_score(self, response: str) -> float:
        """extract numeric score from the response using the score_pattern"""
        # Find all numeric tokens like [[0.5]] and take the last one (most decisive)
        matches = list(re.finditer(self.score_pattern, response))
        if not matches:
            raise LLMJudgeParseError(
                f"No match found for pattern '{self.score_pattern}' in judge response: {response}"
            )

        # iterate from last to first to pick the final rating
        for match in reversed(matches):
            # prefer captured groups
            for group in match.groups():
                if group is None:
                    continue
                try:
                    val = float(group)
                    # clamp to [0, 1] per instruction
                    return max(0.0, min(1.0, val))
                except (ValueError, TypeError):
                    continue
            # fallback: try entire match if groups fail
            try:
                val = float(match.group(0))
                return max(0.0, min(1.0, val))
            except (ValueError, TypeError):
                continue

        raise LLMJudgeParseError(f'Failed to convert judge score to float: {response}')

    def _extract_pattern_score(self, response: str) -> float:
        """use the score_pattern to extract categorical scores"""
        # strict standalone A/B matching using MULTILINE to handle simple outputs
        match = re.search(self.score_pattern, response, re.MULTILINE)
        if match:
            answer = match.group(1) if match.lastindex else match.group(0).strip()
            if answer not in self.score_mapping:
                raise LLMJudgeParseError(
                    f'Judge answer {answer!r} is not present in score_mapping={self.score_mapping!r}.'
                )
            return self.score_mapping[answer]
        else:
            raise LLMJudgeParseError(
                f"No match found for pattern '{self.score_pattern}' in judge response: {response}"
            )
