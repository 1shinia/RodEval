# Copyright (c) Alibaba, Inc. and its affiliates.
"""MTEB evaluation entry point.

Implements the MTEB 2.x evaluation flow with optional two-stage
(Encoder + CrossEncoder) reranking. Data is loaded natively via HuggingFace
(with HF_ENDPOINT mirror for faster CN access).
"""
import mteb
import os
from datasets import DatasetDict
from pathlib import Path
from tabulate import tabulate
from typing import List

from evalscope.backend.rag_eval.models import load_model
from evalscope.backend.rag_eval.mteb.arguments import CustomTaskConfig, MTEBEvalConfig, MTEBModelConfig, MTEBToolConfig
from evalscope.utils.logger import get_logger

logger = get_logger()


def run_mteb_eval(config: MTEBToolConfig):
    """Main entry point for MTEB evaluation.

    Dispatch logic:
        - 1 encoder + 1 cross-encoder, with at least one Retrieval task
          → two-stage evaluation (encoder retrieval → cross-encoder rerank)
        - 1 encoder + 1 cross-encoder without Retrieval tasks
          → run cross-encoder directly (single-stage on Reranking tasks)
        - otherwise → run each model independently in single-stage mode
    """
    eval_args = config.eval
    models = config.models

    encoders = [m for m in models if not m.is_cross_encoder]
    rerankers = [m for m in models if m.is_cross_encoder]

    if len(encoders) == 1 and len(rerankers) == 1:
        tasks = resolve_tasks(eval_args)
        has_retrieval = any(getattr(t.metadata, 'type', None) == 'Retrieval' for t in tasks)
        if has_retrieval:
            return two_stage_eval(encoders[0], rerankers[0], eval_args)
        else:
            return one_stage_eval(rerankers[0], eval_args)

    results = None
    for model_config in models:
        results = one_stage_eval(model_config, eval_args)
    return results


def resolve_tasks(eval_args: MTEBEvalConfig) -> list:
    """Resolve tasks based on eval_args configuration."""
    if eval_args.task_names:
        kwargs = {}
        if eval_args.languages:
            kwargs['languages'] = eval_args.languages
        return mteb.get_tasks(tasks=eval_args.task_names, **kwargs)
    elif eval_args.task_types or eval_args.languages:
        kwargs = {}
        if eval_args.task_types:
            kwargs['task_types'] = eval_args.task_types
        if eval_args.languages:
            kwargs['languages'] = eval_args.languages
        # Also pass categories to filter relevant task types
        if hasattr(eval_args, 'categories') and eval_args.categories:
            kwargs['categories'] = eval_args.categories
        return mteb.get_tasks(**kwargs)
    else:
        raise ValueError("Must specify either 'task_names', 'task_types'/'languages', "
                         'or a combination of both.')


def _build_evaluate_kwargs(eval_args: MTEBEvalConfig, output_folder: str, prediction_folder=None) -> dict:
    """Build kwargs dict for mteb.evaluate() from eval args (MTEB 2.18+)."""
    from mteb.cache import ResultCache
    eval_kwargs = {}
    if eval_args.encode_kwargs:
        eval_kwargs['encode_kwargs'] = eval_args.encode_kwargs
    if prediction_folder is not None:
        eval_kwargs['prediction_folder'] = prediction_folder
    # Use task's work_dir as result cache to isolate per-eval results
    os.makedirs(output_folder, exist_ok=True)
    eval_kwargs['cache'] = ResultCache(cache_path=output_folder)
    eval_kwargs['overwrite_strategy'] = 'always'
    return eval_kwargs


def one_stage_eval(model_args: MTEBModelConfig, eval_args: MTEBEvalConfig):
    """Run single-model MTEB evaluation using native HuggingFace data loading."""
    model = load_model(model_args)
    tasks = resolve_tasks(eval_args)

    # Apply per-task limits: subset the dataset after loading
    if eval_args.limits is not None:
        logger.info(f'Applying limits={eval_args.limits} to {len(tasks)} task(s)')
        for task in tasks:
            try:
                task.data_loaded = False
                task.load_data()
                # Directly subset the loaded dataset
                if hasattr(task, 'dataset') and task.dataset is not None:
                    ds = task.dataset
                    subset = DatasetDict({
                        k: v.select(range(min(eval_args.limits, len(v))))
                        for k, v in ds.items()
                    }) if isinstance(ds, DatasetDict) else ds.select(range(min(eval_args.limits, len(ds))))
                    task.dataset = subset
            except Exception as e:
                logger.warning(f'Failed to apply limits to {task.metadata.name}: {e}')

    task_names = [t.metadata.name for t in tasks]
    logger.info(f'Resolved {len(tasks)} task(s): {task_names}')
    logger.info(f'Starting evaluation (data will be downloaded from HF mirror on first run)...')
    eval_kwargs = _build_evaluate_kwargs(eval_args, eval_args.output_folder)

    results = mteb.evaluate(
        model=model,
        tasks=tasks,
        **eval_kwargs,
    )

    show_results(eval_args.output_folder, model, results.task_results)
    return results


def two_stage_eval(
    encoder_args: MTEBModelConfig,
    reranker_args: MTEBModelConfig,
    eval_args: MTEBEvalConfig,
):
    """Run two-stage evaluation: encoder retrieval, then cross-encoder rerank."""
    encoder = load_model(encoder_args)
    reranker = load_model(reranker_args)

    tasks = resolve_tasks(eval_args)

    stage1_path = os.path.join(eval_args.output_folder, 'stage1')
    stage2_path = os.path.join(eval_args.output_folder, 'stage2')
    stage1_predictions = Path(stage1_path) / 'predictions'

    logger.info('=== Stage 1: Encoder retrieval ===')
    eval_kwargs_s1 = _build_evaluate_kwargs(
        eval_args,
        stage1_path,
        prediction_folder=stage1_predictions,
    )
    mteb.evaluate(
        model=encoder,
        tasks=tasks,
        **eval_kwargs_s1,
    )

    logger.info('=== Stage 2: CrossEncoder reranking ===')
    for task in tasks:
        if getattr(task.metadata, 'type', None) == 'Retrieval':
            task.load_data()
            task.convert_to_reranking(
                top_ranked_path=str(stage1_predictions),
                top_k=eval_args.top_k,
            )

    show_results(stage1_path, encoder, None)
    show_results(stage2_path, reranker, None)
    return None


def show_results(output_folder: str, model, task_results) -> None:
    """Display evaluation results."""
    if task_results is None:
        return
    headers = ['Model', 'Task Type', 'Task', 'Split', 'Subset', 'Main Score']
    rows = []
    for result in task_results:
        if hasattr(result, 'scores'):
            for split in result.scores:
                for score in result.scores[split]:
                    model_name = getattr(model, 'model_name_or_path', 'unknown').split('/')[-1]
                    rows.append([
                        model_name,
                        getattr(result, 'task_type', 'N/A'),
                        getattr(result, 'task_name', 'N/A'),
                        split,
                        score.get('hf_subset', 'N/A'),
                        score.get('main_score', 'N/A'),
                    ])
    if rows:
        from evalscope.utils.logger import get_logger
        _logger = get_logger()
        _logger.info('Evaluation results:\n' + tabulate(rows, headers=headers, tablefmt='grid'))
        _logger.info(f'Results saved in: {output_folder}')
