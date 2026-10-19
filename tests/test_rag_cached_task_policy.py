"""RAG cache reuse is allowed while dataset libraries remain offline."""
import socket
from unittest.mock import Mock

from evalscope.backend.rag_eval.mteb.task_template import _preflight_offline_tasks


def test_cached_task_is_usable_with_offline_flags(tmp_path, monkeypatch):
    from datasets import config as datasets_config
    from huggingface_hub import constants as hf_constants

    cached = tmp_path / 'prepared.txt'
    cached.write_text('prepared')
    connect = Mock(side_effect=AssertionError('unexpected network access'))
    monkeypatch.setattr(socket.socket, 'connect', connect)

    class CachedTask:
        data_loaded = False

        def load_data(self):
            assert datasets_config.HF_DATASETS_OFFLINE
            assert hf_constants.HF_HUB_OFFLINE
            self.dataset = cached.read_text()
            self.data_loaded = True

    task = CachedTask()
    _preflight_offline_tasks([task], offline=True, allow_download=False)
    assert task.data_loaded
    assert task.dataset == 'prepared'
    connect.assert_not_called()
