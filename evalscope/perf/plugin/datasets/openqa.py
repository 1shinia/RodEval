import json
import os
from typing import Any, Dict, Iterator, List

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.datasets.base import DatasetPluginBase
from evalscope.perf.plugin.registry import register_dataset


@register_dataset('openqa')
class OpenqaDatasetPlugin(DatasetPluginBase):
    """Read dataset and return prompt.
    Datasets: https://www.modelscope.cn/datasets/AI-ModelScope/HC3-Chinese/resolve/master/open_qa.jsonl
    """

    def __init__(self, query_parameters: Arguments):
        super().__init__(query_parameters)

    def build_messages(self) -> Iterator[List[Dict]]:
        if not self.query_parameters.dataset_path:
            file_name = 'open_qa.jsonl'
            # Try local cache first to avoid ModelScope API call (GFW-blocked)
            import contextlib
            import os as _os
            cache_dir = os.path.expanduser('~/.cache/modelscope/hub/datasets/AI-ModelScope/HC3-Chinese')
            cached_file = os.path.join(cache_dir, file_name)
            if os.path.isfile(cached_file):
                local_path = cache_dir
            else:
                from modelscope import dataset_snapshot_download
                with contextlib.redirect_stdout(open(_os.devnull, 'w')):
                    local_path = dataset_snapshot_download('AI-ModelScope/HC3-Chinese', allow_patterns=[file_name])
            self.query_parameters.dataset_path = os.path.join(local_path, file_name)

        for item in self.dataset_line_by_line(self.query_parameters.dataset_path):
            item = json.loads(item)
            prompt = item['question'].strip()
            is_valid, _ = self.check_prompt_length(prompt)
            if is_valid:
                if self.query_parameters.apply_chat_template:
                    message = self.create_message(prompt)
                    yield [message]
                else:
                    yield prompt
