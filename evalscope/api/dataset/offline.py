"""Cache-only resolution for special dataset snapshots."""
from .hub import _dataset_offline


def resolve_dataset_snapshot(dataset_id, **kwargs):
    if not _dataset_offline():
        from modelscope import dataset_snapshot_download
        return dataset_snapshot_download(dataset_id, **kwargs)
    from modelscope.hub.snapshot_download import snapshot_download
    try:
        return snapshot_download(dataset_id, repo_type='dataset', local_files_only=True, **kwargs)
    except Exception as exc:
        raise RuntimeError('数据集缓存缺失，请联系管理员准备数据集。') from exc
