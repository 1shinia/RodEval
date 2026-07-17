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


def _is_retrieval_task(task) -> bool:
    """Check if a task is a Retrieval or Reranking type."""
    task_type = getattr(task.metadata, 'type', None)
    return task_type in ('Retrieval', 'Reranking')


def _apply_retrieval_limits(task, limits: int) -> None:
    """Apply query limits to a Retrieval task's queries/relevant_docs.

    MTEB 2.x stores retrieval data in task.dataset as a nested dict:
        task.dataset[subset][split] = RetrievalSplitData({
            "corpus": Dataset,
            "queries": Dataset,
            "relevant_docs": dict,
        })
    """
    if not hasattr(task, 'dataset') or not task.dataset:
        return

    for subset_key, subset_data in task.dataset.items():
        if not isinstance(subset_data, dict):
            continue
        for split_key, split_data in subset_data.items():
            if not isinstance(split_data, dict):
                continue
            queries = split_data.get('queries')
            if queries is None or len(queries) <= limits:
                continue

            # queries is a HuggingFace Dataset with 'id' column
            limited_queries = queries.select(range(limits))
            keep_qids = set(str(q) for q in limited_queries['id'])

            split_data['queries'] = limited_queries

            # Filter relevant_docs to keep only limited query IDs
            relevant_docs = split_data.get('relevant_docs')
            if relevant_docs is not None and isinstance(relevant_docs, dict):
                split_data['relevant_docs'] = {k: v for k, v in relevant_docs.items() if k in keep_qids}

            logger.info(
                f'Applied limits={limits} to {task.metadata.name}/{subset_key}/{split_key}: '
                f'{len(limited_queries)} queries kept (was {len(queries)})'
            )


def one_stage_eval(model_args: MTEBModelConfig, eval_args: MTEBEvalConfig):
    """Run single-model MTEB evaluation using native HuggingFace data loading."""
    import json as _json
    from datetime import datetime as _datetime

    model = load_model(model_args)
    tasks = resolve_tasks(eval_args)

    work_dir = eval_args.output_folder
    task_names = [t.metadata.name for t in tasks]
    total_tasks = len(tasks)

    # ── Write initial progress ──────────────────────────────────────
    _write_progress(
        work_dir, {
            'status': 'running',
            'phase': 'loading',
            'pipeline': 'eval',
            'total_count': total_tasks,
            'processed_count': 0,
            'percent': 0.0,
            'tasks': task_names,
            'updated_at': _datetime.now().isoformat(),
        }
    )

    # Apply per-task limits: subset the dataset after loading
    if eval_args.limits is not None:
        logger.info(f'Applying limits={eval_args.limits} to {total_tasks} task(s)')
        for task in tasks:
            try:
                task.data_loaded = False
                task.load_data()
                if _is_retrieval_task(task):
                    _apply_retrieval_limits(task, eval_args.limits)
                elif hasattr(task, 'dataset') and task.dataset is not None:
                    ds = task.dataset
                    subset = DatasetDict({
                        k: v.select(range(min(eval_args.limits, len(v))))
                        for k, v in ds.items()
                    }) if isinstance(ds, DatasetDict) else ds.select(range(min(eval_args.limits, len(ds))))
                    task.dataset = subset
            except Exception as e:
                logger.warning(f'Failed to apply limits to {task.metadata.name}: {e}')

    # ── Extract actual sample counts when limits is null ────────────
    actual_limits = eval_args.limits
    if actual_limits is None:
        for task in tasks:
            try:
                task.data_loaded = False
                task.load_data()
                if _is_retrieval_task(task):
                    queries = _get_queries_count(task)
                    if queries:
                        actual_limits = queries if actual_limits is None else max(actual_limits, queries)
                elif hasattr(task, 'dataset') and task.dataset is not None:
                    ds = task.dataset
                    eval_splits = _get_eval_splits(task)
                    if isinstance(ds, DatasetDict):
                        for split_key in eval_splits:
                            if split_key in ds:
                                actual_limits = len(
                                    ds[split_key]
                                ) if actual_limits is None else max(actual_limits, len(ds[split_key]))
                    else:
                        actual_limits = len(ds) if actual_limits is None else max(actual_limits, len(ds))
            except Exception as e:
                logger.debug(f'Could not count samples for {task.metadata.name}: {e}')
        # Write actual count back to config so reports show real number
        if actual_limits is not None:
            _update_config_limits(work_dir, actual_limits)

    logger.info(f'Resolved {total_tasks} task(s): {task_names}')
    logger.info(f'Starting evaluation (data will be downloaded from HF mirror on first run)...')
    eval_kwargs = _build_evaluate_kwargs(eval_args, eval_args.output_folder)

    # ── Update progress: evaluating ─────────────────────────────────
    _write_progress(
        work_dir, {
            'status': 'running',
            'phase': 'evaluating',
            'pipeline': 'eval',
            'total_count': total_tasks,
            'processed_count': 0,
            'percent': 0.0,
            'tasks': task_names,
            'updated_at': _datetime.now().isoformat(),
        }
    )

    results = mteb.evaluate(
        model=model,
        tasks=tasks,
        **eval_kwargs,
    )

    # ── Write completion progress ───────────────────────────────────
    _write_progress(
        work_dir, {
            'status': 'completed',
            'phase': 'completed',
            'pipeline': 'eval',
            'total_count': total_tasks,
            'processed_count': total_tasks,
            'percent': 100.0,
            'tasks': task_names,
            'updated_at': _datetime.now().isoformat(),
        }
    )

    show_results(eval_args.output_folder, model, results.task_results)
    _collect_perf_metrics(work_dir)
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


def _collect_perf_metrics(work_dir: str) -> None:
    """Read MTEB result JSONs and write perf_metrics.json."""
    import json
    import os
    results_dir = os.path.join(work_dir, 'results')
    if not os.path.isdir(results_dir):
        return

    tasks_perf = []
    total_samples = 0
    total_encoding_time = 0.0

    for root, _, files in os.walk(results_dir):
        for fname in files:
            if not fname.endswith('.json') or fname == 'model_meta.json':
                continue
            fpath = os.path.join(root, fname)
            try:
                with open(fpath) as f:
                    data = json.load(f)
            except Exception:
                continue

            task_name = data.get('task_name', fname.replace('.json', ''))
            evaluation_time = data.get('evaluation_time', 0)
            phases = data.get('evaluation_phases', [])

            encoding_time = sum(
                p['end'] - p['start']
                for p in phases
                if 'encode' in p.get('name', '').lower() or 'Encoding' in p.get('name', '')
            ) if phases else evaluation_time

            # Count samples from scores_per_experiment, or fall back to config limits
            samples = 0
            for split_data in data.get('scores', {}).values():
                if isinstance(split_data, list) and split_data:
                    sc = split_data[0].get('scores_per_experiment')
                    if sc:
                        samples = max(samples, len(sc))
            if samples == 0:
                samples = _read_config_limits(work_dir)

            total_samples += samples
            total_encoding_time += encoding_time

            tasks_perf.append({
                'task_name': task_name,
                'total_time': round(evaluation_time, 2),
                'encoding_time': round(encoding_time, 2),
                'samples': samples,
            })

    throughput = round(total_samples / total_encoding_time, 1) if total_encoding_time > 0 else 0
    perf = {
        'total_time': round(sum(t['total_time'] for t in tasks_perf), 2),
        'encoding_time': round(total_encoding_time, 2),
        'total_samples': total_samples,
        'throughput': throughput,
        'tasks': tasks_perf,
    }
    try:
        with open(os.path.join(work_dir, 'perf_metrics.json'), 'w') as f:
            json.dump(perf, f)
        logger.info(f'Collected perf metrics: {total_samples} samples, {throughput} samples/sec')
    except Exception as e:
        logger.debug(f'Could not write perf_metrics: {e}')


def _read_config_limits(work_dir: str) -> int:
    """Read the limits value from task_config.yaml."""
    import yaml
    config_path = os.path.join(work_dir, 'configs', 'task_config.yaml')
    if os.path.exists(config_path):
        try:
            with open(config_path) as f:
                cfg = yaml.safe_load(f) or {}
            limits = cfg.get('eval_config', {}).get('eval', {}).get('limits')
            return int(limits) if limits is not None else 0
        except Exception:
            pass
    return 0


def _write_progress(work_dir: str, data: dict) -> None:
    """Write progress.json for SSE streaming."""
    import json
    import os
    try:
        os.makedirs(work_dir, exist_ok=True)
        with open(os.path.join(work_dir, 'progress.json'), 'w') as f:
            json.dump(data, f)
    except Exception:
        pass


def _get_eval_splits(task) -> list:
    """Get the splits that will be evaluated (default: ['test'])."""
    return getattr(task, 'eval_splits', None) or ['test']


def _get_queries_count(task) -> int | None:
    """Extract query count from a Retrieval task's evaluated splits."""
    if not hasattr(task, 'dataset') or not task.dataset:
        return None
    eval_splits = _get_eval_splits(task)
    for subset_data in task.dataset.values():
        if not isinstance(subset_data, dict):
            continue
        for split_key, split_data in subset_data.items():
            if split_key not in eval_splits:
                continue
            if not isinstance(split_data, dict):
                continue
            queries = split_data.get('queries')
            if queries is not None and hasattr(queries, '__len__'):
                return len(queries)
    return None


def _update_config_limits(work_dir: str, count: int) -> None:
    """Update task_config.yaml with actual sample count."""
    import yaml
    config_path = os.path.join(work_dir, 'configs', 'task_config.yaml')
    if not os.path.exists(config_path):
        return
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
        eval_cfg = cfg.setdefault('eval_config', {}).setdefault('eval', {})
        eval_cfg['limits'] = count
        with open(config_path, 'w') as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
        logger.info(f'Updated config limits to actual sample count: {count}')
    except Exception as e:
        logger.debug(f'Could not update config limits: {e}')
