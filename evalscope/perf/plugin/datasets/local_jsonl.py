import json
import os
from typing import Any, Dict, Iterator, List, Union

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.datasets.base import DatasetPluginBase
from evalscope.perf.plugin.registry import register_dataset


@register_dataset('local_jsonl')
class LocalJsonlDatasetPlugin(DatasetPluginBase):
    """通用本地 JSONL 数据集插件。

    自动识别每行的 prompt 字段，优先级：
      1. ``question``   — 同 openqa 格式
      2. ``messages``   — 同 share_gpt 格式（直接使用消息列表）
      3. ``prompt``     — 显式 prompt 字段
      4. ``input``      — 指令/微调格式
      5. ``text``       — 纯文本
      6. 整行 JSON 作为字符串（fallback）

    支持 chat_template 包装和 prompt 长度过滤。
    """

    _PROMPT_FIELDS = ('question', 'messages', 'prompt', 'input', 'text')

    def __init__(self, query_parameters: Arguments):
        super().__init__(query_parameters)
        self._field = None  # detected field name, set on first line

    def build_messages(self) -> Iterator[Union[List[Dict], str]]:
        path = self.query_parameters.dataset_path
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(f'Dataset file not found: {path}')

        for line in self.dataset_line_by_line(path):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                # Plain text line
                prompt = line
                item = None
            else:
                prompt = self._extract_prompt(item)

            if prompt is None:
                continue

            is_valid, _ = self.check_prompt_length(prompt)
            if not is_valid:
                continue

            if self.query_parameters.apply_chat_template:
                # If the field is 'messages', use them directly
                if self._field == 'messages' and isinstance(prompt, list):
                    yield prompt
                else:
                    message = self.create_message(str(prompt))
                    yield [message]
            else:
                yield str(prompt)

    def _extract_prompt(self, item: dict) -> Union[str, List[Dict], None]:
        """从 JSON 对象中提取 prompt，自动检测字段名。"""
        if self._field is not None:
            # 已检测到字段，直接使用
            return item.get(self._field)

        # 自动检测：按优先级尝试常见字段
        for field in self._PROMPT_FIELDS:
            if field in item:
                self._field = field
                return item[field]

        # Fallback：整行 JSON 作为字符串
        self._field = '__raw__'
        return json.dumps(item, ensure_ascii=False)
