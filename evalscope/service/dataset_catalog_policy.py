"""Authorize published datasets, independently of prepared cache versions.

Source sanitization and offline enforcement live in the shared eval entry policy.
Snapshot resolution remains available internally, but normal-user requests must
not carry old version selections into the runner.
"""
from . import dataset_catalog_inventory as inventory


def apply_catalog_policy(data):
    from .utils.benchmarks import load_benchmark_data

    dataset_args = data.setdefault('dataset_args', {})
    if not isinstance(dataset_args, dict):
        raise ValueError('dataset_args 必须为对象。')
    names = data.get('datasets') or []
    if not isinstance(names, list) or any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError('datasets 必须为数据集名称列表。')
    for name in names:
        meta = load_benchmark_data(name).get(name, {}).get('meta', {})
        category = meta.get('category')
        if category not in ('llm', 'vlm', 'agent'):
            raise ValueError(f'未知或不支持的数据集：{name}，请联系管理员。')
        inventory.require_dataset_published(category, name)
        if not isinstance(dataset_args.get(name, {}), dict):
            raise ValueError('数据集参数必须为对象。')
    # Also discard stale selections for unselected benchmarks in saved configs.
    for args in dataset_args.values():
        if isinstance(args, dict):
            args.pop('_catalog_selection', None)
    return data


def apply_rag_catalog_policy(config):
    args = config.setdefault('eval', {})
    if not isinstance(args, dict):
        raise ValueError('RAG MTEB eval configuration must be an object.')
    if args.get('custom_tasks'):
        raise ValueError('普通用户不可使用自定义数据路径。')
    names = args.get('task_names')
    if not isinstance(names, list) or not names or any(
            not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError('请选择管理员已发布的明确 RAG 任务，不支持按类型自动展开。')
    for name in names:
        inventory.require_dataset_published('rag', name)
    args.pop('catalog_selection', None)
    args.pop('task_types', None)
    args.pop('languages', None)
    args['offline'] = True
    args['allow_download'] = False
    return config
