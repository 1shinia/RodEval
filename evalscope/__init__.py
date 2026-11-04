# Copyright (c) Alibaba, Inc. and its affiliates.
# Mirror HuggingFace downloads to the CN mirror *before* any submodule is
# imported. huggingface_hub reads HF_ENDPOINT into a constant at import time,
# so setting it later (e.g. inside the Flask app factory) is too late. Using
# setdefault keeps an explicitly configured endpoint when the operator set one.
import os as _os

_os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

from evalscope import agent  # noqa: E402,F401  # registered agent strategies
from evalscope.benchmarks import *  # noqa: E402
from evalscope.config import TaskConfig  # noqa: E402
from evalscope.evaluator import *  # noqa: E402
from evalscope.filters import extraction, selection  # noqa: E402
from evalscope.metrics import metric  # noqa: E402
from evalscope.models import model_apis  # noqa: E402
from evalscope.run import run_task  # noqa: E402
from .version import __release_datetime__, __version__  # noqa: E402
