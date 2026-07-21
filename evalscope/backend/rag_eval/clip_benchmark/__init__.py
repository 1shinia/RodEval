from evalscope.backend.rag_eval.clip_benchmark.arguments import ClipBenchmarkEvalConfig, ClipBenchmarkToolConfig

# evaluate() imported lazily to avoid pulling in langchain_core/CLIP deps for MTEB/RAGAS users


def evaluate(*args, **kwargs):
    from evalscope.backend.rag_eval.clip_benchmark.task_template import evaluate as _eval
    return _eval(*args, **kwargs)
