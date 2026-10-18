import asyncio
import re
import json
import logging
from pathlib import Path
from typing import Any, Dict

from .arguments import SUPPORTED_METRICS, normalize_metric_name

logger = logging.getLogger(__name__.replace('evalscope', 'evalperf'))


def import_metric(metric_name: str):
    """Construct a fresh supported metric, never a user-supplied import path."""
    name = normalize_metric_name(metric_name)
    from ragas import metrics
    return getattr(metrics, SUPPORTED_METRICS[name])()


def _build_llm(llm_config) -> Any:
    """Build LLM from config using ragas 0.4.x llm_factory pattern.

    Args:
        llm_config: RAGASLLMConfig instance.

    Returns:
        A BaseRagasLLM instance compatible with ragas 0.4.x.
    """
    try:
        from openai import OpenAI
        from ragas.llms import llm_factory

        client_kwargs: Dict[str, Any] = {}
        if llm_config.api_base:
            client_kwargs['base_url'] = llm_config.api_base
        if llm_config.api_key:
            client_kwargs['api_key'] = llm_config.api_key

        client = OpenAI(**client_kwargs)
        options = {'temperature': llm_config.temperature}
        if llm_config.max_tokens is not None:
            options['max_tokens'] = llm_config.max_tokens
        return llm_factory(llm_config.model_name, client=client, **options)
    except (ImportError, AttributeError, TypeError):
        logger.warning('Failed to use llm_factory, falling back to LangchainLLMWrapper', exc_info=True)
        from langchain_openai import ChatOpenAI
        from ragas.llms.base import LangchainLLMWrapper

        langchain_llm = ChatOpenAI(
            model=llm_config.model_name,
            base_url=llm_config.api_base,
            api_key=llm_config.api_key or 'EMPTY',
            temperature=llm_config.temperature,
            max_tokens=llm_config.max_tokens,
        )
        return LangchainLLMWrapper(langchain_llm)


def _build_embeddings(embedding_config) -> Any:
    """Build embeddings from config using ragas 0.4.x embedding_factory pattern.

    Args:
        embedding_config: RAGASEmbeddingConfig instance.

    Returns:
        A BaseRagasEmbeddings or BaseRagasEmbedding instance.
    """
    from ragas.embeddings.base import LangchainEmbeddingsWrapper

    if embedding_config.provider == 'openai':
        from langchain_openai import OpenAIEmbeddings
        return LangchainEmbeddingsWrapper(OpenAIEmbeddings(
            model=embedding_config.model_name_or_path,
            base_url=embedding_config.api_base,
            api_key=embedding_config.api_key or 'EMPTY',
            check_embedding_ctx_length=False,
        ))
    if embedding_config.provider != 'huggingface':
        raise ValueError(f'Unsupported embedding provider: {embedding_config.provider}')
    from evalscope.backend.rag_eval.models.utils import resolve_model_path
    from langchain_huggingface import HuggingFaceEmbeddings
    local_path = resolve_model_path(embedding_config.model_name_or_path)
    return LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=local_path))


def _load_dataset(testset_file: str, metrics=None):
    """Load and validate JSON/JSONL rows into a RAGAS evaluation dataset."""
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample

    file_path = Path(testset_file)
    if not file_path.is_file():
        raise ValueError(f'RAGAS testset file does not exist: {testset_file}')
    if file_path.suffix.lower() == '.jsonl':
        rows = []
        with file_path.open('r', encoding='utf-8') as file:
            for line_number, line in enumerate(file, 1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f'Invalid JSONL at line {line_number}: {exc.msg}') from exc
    else:
        try:
            with file_path.open('r', encoding='utf-8') as file:
                data = json.load(file)
        except json.JSONDecodeError as exc:
            raise ValueError(f'Invalid JSON testset: {exc.msg}') from exc
        rows = data if isinstance(data, list) else [data]

    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError('RAGAS testset must contain at least one JSON object row.')

    required = set()
    for metric in metrics or []:
        for columns in getattr(metric, 'required_columns', {}).values():
            required.update(columns)
    samples = []
    for row_number, item in enumerate(rows, 1):
        raw_contexts = item.get('retrieved_contexts', item.get('contexts', []))
        contexts = [] if raw_contexts is None else (
            raw_contexts if isinstance(raw_contexts, list) else [str(raw_contexts)])
        values = {
            'user_input': item.get('user_input', item.get('question')),
            'response': item.get('response', item.get('answer')),
            'retrieved_contexts': contexts,
            'reference': item.get('reference', item.get('ground_truth')),
        }
        missing = []
        for field in required:
            value = values.get(field)
            if field == 'retrieved_contexts':
                valid = (isinstance(value, list) and bool(value)
                         and all(isinstance(text, str) and text.strip() for text in value))
            else:
                valid = isinstance(value, str) and bool(value.strip())
            if not valid:
                missing.append(field)
        if missing:
            raise ValueError(f'RAGAS testset row {row_number} missing required field(s): {", ".join(sorted(missing))}')
        samples.append(SingleTurnSample(**values))
    return EvaluationDataset(samples=samples)


async def _run_ragas_eval(args) -> Any:
    """Async RAGAS evaluation using 0.4.x aevaluate() API.

    Args:
        args: RAGASEvalConfig instance.

    Returns:
        EvaluationResult from ragas.
    """
    from ragas import aevaluate
    from ragas.run_config import RunConfig

    from ragas.metrics.base import MetricWithEmbeddings

    if not args.metrics:
        raise ValueError('At least one RAGAS metric is required.')
    metrics = [import_metric(name) for name in args.metrics]
    dataset = _load_dataset(args.testset_file, metrics=metrics)
    needs_embeddings = any(isinstance(metric, MetricWithEmbeddings) for metric in metrics)
    if needs_embeddings and args.embeddings is None:
        raise ValueError('Selected RAGAS metrics require an embedding model.')
    if needs_embeddings and args.embeddings.provider not in ('openai', 'huggingface'):
        raise ValueError(f'Unsupported embedding provider: {args.embeddings.provider}')
    llm = _build_llm(args.critic_llm)
    embeddings = _build_embeddings(args.embeddings) if needs_embeddings else None

    # 5. Build run config
    run_config = RunConfig(timeout=600, max_retries=3, max_wait=60)

    # 6. Run evaluation
    eval_kwargs: Dict[str, Any] = {
        'dataset': dataset,
        'metrics': metrics,
        'llm': llm,
        'embeddings': embeddings,
        'run_config': run_config,
        'raise_exceptions': args.raise_exceptions,
    }
    if args.batch_size is not None:
        eval_kwargs['batch_size'] = args.batch_size

    result = await aevaluate(**eval_kwargs)
    return result


def rag_eval(args) -> dict:
    """Main entry point for RAGAS evaluation.

    Uses ragas 0.4.x aevaluate() API with asyncio.run() for synchronous invocation.

    Args:
        args: RAGASEvalConfig instance or dict.

    Returns:
        Dictionary of metric scores.
    """
    from evalscope.backend.rag_eval.ragas.arguments import RAGASEvalConfig

    if isinstance(args, dict):
        args = RAGASEvalConfig(**args)

    logger.info(f'Starting RAGAS evaluation with metrics: {args.metrics}')

    result = asyncio.run(_run_ragas_eval(args))

    # Save results
    output_path = Path(args.testset_file).with_name(Path(args.testset_file).stem + '_score.json')
    result_df = result.to_pandas()
    import numpy as np
    import pandas as pd

    scores = {}
    counts = {}
    for metric_name in args.metrics:
        name = re.sub(r'(?<!^)(?=[A-Z])', '_', metric_name).lower()
        if name not in result_df:
            raise ValueError(f'No valid RAGAS result column: {name}')
        values = pd.to_numeric(result_df[name], errors='coerce')
        valid = values.notna() & np.isfinite(values) & values.between(0, 1)
        valid_count = int(valid.sum())
        failed_count = len(values) - valid_count
        counts[name] = {'valid': valid_count, 'failed': failed_count, 'total': len(values)}
        scores[name] = float(values[valid].mean()) if valid_count else None
        result_df.loc[~valid, name] = None
        logger.info('RAGAS %s: valid=%d failed=%d total=%d score=%s',
                    name, valid_count, failed_count, len(values), scores[name])

    result_df.to_json(str(output_path), orient='records', force_ascii=False, indent=2)
    summary_path = output_path.with_name(output_path.stem + '_summary.json')
    summary_path.write_text(json.dumps({'scores': scores, 'counts': counts},
                                      ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')
    if not any(count['valid'] for count in counts.values()):
        raise ValueError('No valid RAGAS scores; see per-metric failure counts.')
    if any(count['failed'] for count in counts.values()):
        logger.warning('RAGAS evaluation has partial failures; scores use valid samples only. Counts: %s', counts)
    logger.info('RAGAS evaluation complete. Results saved to %s', output_path)
    logger.info('Scores: %s', scores)
    return scores
