# Copyright (c) Alibaba, Inc. and its affiliates.
# Copyright (c) EleutherAI Inc, and its affiliates.

from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.metric import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

logger = get_logger()

PROMPT_TEMPLATE = """
Read the content and answer the following question.

Content: {content}

Question: {question}

Keep your The last line of your response should be of the form "ANSWER: [ANSWER]" (without quotes) where [ANSWER] is the answer to the problem.
""".lstrip()  # noqa: E501


@register_benchmark(
    BenchmarkMeta(
        name='trivia_qa',
        pretty_name='TriviaQA',
        dataset_id='evalscope/trivia_qa',
        tags=[Tags.QA, Tags.READING_COMPREHENSION],
        description="""
## Overview

TriviaQA is a large-scale reading comprehension dataset containing over 650K question-answer-evidence triples. Questions are collected from trivia enthusiast websites and paired with Wikipedia articles as evidence documents.

## Task Description

- **Task Type**: Reading Comprehension / Question Answering
- **Input**: Question with Wikipedia context passage
- **Output**: Answer extracted or generated from context
- **Domain**: General knowledge trivia questions

## Key Features

- 650K+ question-answer-evidence triples
- Questions written by trivia enthusiasts (naturally challenging)
- Multiple valid answer aliases for flexible evaluation
- Wikipedia articles provide evidence passages
- Tests both reading comprehension and knowledge retrieval

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Uses the Wikipedia reading comprehension subset (rc.wikipedia)
- Answers should follow the format: "ANSWER: [ANSWER]"
- Supports inclusion-based matching for answer comparison
- Evaluates on validation split
""",
        subset_list=['rc.wikipedia'],
        few_shot_num=0,
        train_split=None,
        eval_split='validation',
        metric_list=[{
            'acc': {
                'allow_inclusion': True,
                # TriviaQA aliases are reference answers that may occur inside
                # a longer model response.  The legacy default uses the
                # opposite containment direction for multi-choice compatibility.
                'inclusion_direction': 'reference_in_prediction'
            }
        }],
        prompt_template=PROMPT_TEMPLATE,
    )
)
class TriviaQaAdapter(DefaultDataAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        question = record['question']
        answers = record['answer']['aliases'] + record['answer']['normalized_aliases']
        content = record['entity_pages']['wiki_context']
        return Sample(
            input=question, target=answers, metadata={
                'question_id': record['question_id'],
                'content': content
            }
        )

    def format_prompt_template(self, sample):
        return self.prompt_template.format(content=sample.metadata['content'], question=sample.input)

    def extract_answer(self, prediction: str, task_state: TaskState):
        # use regex to extract the answer from the prediction
        import re

        matches = re.findall(r'ANSWER:\s*(.*)', prediction)
        if matches:
            return matches[-1].strip()
        return prediction.strip()

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        """Score against TriviaQA aliases without concatenating them.

        ``Sample.target`` is a list of valid aliases.  ``TaskState.target`` is
        intentionally a legacy concatenated representation because other
        benchmarks use list targets for multi-letter multiple-choice answers.
        TriviaQA therefore opts into alias semantics here and accepts the best
        score across the individual references.
        """
        references = task_state.targets or [reference]
        alias_scores = [
            super(TriviaQaAdapter, self).match_score(
                original_prediction=original_prediction,
                filtered_prediction=filtered_prediction,
                reference=alias,
                task_state=task_state,
            ) for alias in references
        ]

        final = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
            metadata={'reference_count': len(references), 'reference_mode': 'any_of'},
        )
        metric_names = {name for score in alias_scores for name in score.value}
        for metric_name in metric_names:
            candidates = [score.value[metric_name] for score in alias_scores if metric_name in score.value]
            if candidates:
                final.value[metric_name] = max(candidates)

        metric_errors = {}
        for score in alias_scores:
            metric_errors.update((score.metadata or {}).get('metric_errors', {}))
        if metric_errors and not final.value:
            final.metadata['metric_errors'] = metric_errors

        return final
