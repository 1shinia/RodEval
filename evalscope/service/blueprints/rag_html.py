"""Standalone, escaped HTML for existing MTEB result artifacts."""
import json
import os
from pathlib import Path

from flask import render_template_string


_TEMPLATE = '''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>向量检索评估报告</title>
<style>
body{font:16px/1.6 system-ui,sans-serif;color:#1f2937;background:#f8fafc;margin:0;padding:32px}
main{max-width:1100px;margin:auto;background:white;padding:28px;border-radius:16px}
h1{font-size:24px}h2{font-size:20px;margin-top:28px}p{color:#475569}
section{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px}
th,td{text-align:left;border-bottom:1px solid #e2e8f0;padding:12px}
th{background:#f1f5f9} .note{font-size:14px}
</style></head><body><main><h1>向量检索评估报告 (RAG)</h1>
<p>模型：{{ model }}</p>
<p class="note">分数含义由各任务指标决定，不等同于准确率。样本较少时结果仅供链路验证参考。</p>
{% for result in results %}<h2>{{ result.task }}</h2><section>
<table><thead><tr><th>数据划分</th><th>子集</th><th>指标</th><th>分数</th></tr></thead><tbody>
{% for row in result.rows %}<tr><td>{{ row.split }}</td><td>{{ row.subset }}</td><td>{{ row.metric }}</td><td>{{ row.score }}</td></tr>{% endfor %}
</tbody></table></section>{% endfor %}
</main></body></html>'''


def render_mteb_html(task_dir, model_name, datasets):
    """Render selected task/model results, without exposing task configuration."""
    root = Path(task_dir).resolve()
    results_dir = root / 'results'
    selected = []
    for directory, _, files in os.walk(results_dir):
        for filename in sorted(files):
            if not filename.endswith('.json') or filename == 'model_meta.json':
                continue
            path = Path(directory) / filename
            if not path.resolve().is_relative_to(root):
                continue
            relative = path.relative_to(results_dir)
            if relative.parts[0].removeprefix('eval__') != model_name:
                continue
            try:
                data = json.loads(path.read_text(encoding='utf-8'))
                task = data.get('task_name', path.stem)
                if datasets and task not in datasets:
                    continue
                rows = []
                for split, scores in data.get('scores', {}).items():
                    for score in scores:
                        for metric, value in score.items():
                            if isinstance(value, (int, float)) and not isinstance(value, bool):
                                rows.append(dict(split=split, subset=score.get('hf_subset', 'default'),
                                                 metric=metric, score=value))
                if rows:
                    selected.append(dict(task=task, rows=rows))
            except (OSError, ValueError, TypeError, AttributeError):
                continue
    if not selected:
        return None
    return render_template_string(_TEMPLATE, model=model_name, results=selected)
