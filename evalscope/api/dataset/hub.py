# Copyright (c) Alibaba, Inc. and its affiliates.

import inspect
import os
from dataclasses import dataclass
from typing import Optional

from evalscope.constants import HubType
from evalscope.utils.logger import get_logger

logger = get_logger()


@dataclass(frozen=True)
class DatasetHub:
    """Small hub handle shared by dataset loaders and benchmark-specific media resolvers."""

    data_id_or_path: str
    data_source: Optional[str] = HubType.MODELSCOPE
    revision: Optional[str] = None
    trust_remote: bool = True
    force_redownload: bool = False
    cache_dir: Optional[str] = None

    def load(self, split: str, subset: str = 'default', **kwargs):
        return load_dataset_from_hub(
            data_id_or_path=self.data_id_or_path,
            split=split,
            subset=subset,
            data_source=self.data_source,
            version=self.revision,
            trust_remote=self.trust_remote,
            force_redownload=self.force_redownload,
            **kwargs,
        )

    def download_file(self, file_path: str) -> str:
        return download_dataset_file(
            data_id_or_path=self.data_id_or_path,
            file_path=file_path,
            data_source=self.data_source,
            revision=self.revision,
            force_redownload=self.force_redownload,
            cache_dir=self.cache_dir,
        )


def load_dataset_from_hub(
    data_id_or_path: str,
    split: str,
    subset: str = 'default',
    data_source: Optional[str] = HubType.MODELSCOPE,
    version: Optional[str] = None,
    trust_remote: bool = True,
    force_redownload: bool = False,
    **kwargs,
):
    """Load a dataset split from ModelScope, Hugging Face, or a local dataset path."""
    import datasets
    from datasets import DownloadMode as HFDownloadMode
    from modelscope import MsDataset
    from modelscope.utils.constant import DownloadMode as MSDownloadMode

    data_source = data_source or HubType.MODELSCOPE
    hf_download_mode = None if not force_redownload else HFDownloadMode.FORCE_REDOWNLOAD
    ms_download_mode = None if not force_redownload else MSDownloadMode.FORCE_REDOWNLOAD

    if data_source == HubType.MODELSCOPE:
        try:
            dataset = MsDataset.load(
                dataset_name=data_id_or_path,
                split=split,
                subset_name=subset,
                version=version,
                trust_remote_code=trust_remote,
                download_mode=ms_download_mode,
                **kwargs,
            )
            if not isinstance(dataset, datasets.Dataset):
                dataset = dataset.to_hf_dataset()
            return dataset
        except datasets.exceptions.DatasetGenerationError:
            logger.warning(
                f'MsDataset.load failed for {data_id_or_path} (split={split}, subset={subset}) '
                f'with schema error. Falling back to snapshot download + parquet loading ...'
            )
            # Fallback: download raw dataset files via modelscope snapshot,
            # then load parquet directly to bypass the datasets builder schema validation.
            try:
                import glob as _glob
                import time as _time
                from modelscope import dataset_snapshot_download

                snapshot_dir = None
                last_err = None

                # Retry loop: first try local cache, then full download with backoff
                for attempt in range(3):
                    try:
                        if attempt == 0:
                            # First attempt: check local cache only (fast path)
                            snapshot_dir = dataset_snapshot_download(
                                dataset_id=data_id_or_path,
                                revision=version,
                                local_files_only=True,
                            )
                        else:
                            # Retry with full download, exponential backoff
                            delay = 2**(attempt - 1)
                            logger.info(f'Retry {attempt}/3 for {data_id_or_path} after {delay}s ...')
                            _time.sleep(delay)
                            snapshot_dir = dataset_snapshot_download(
                                dataset_id=data_id_or_path,
                                revision=version,
                            )
                        if snapshot_dir:
                            break
                    except Exception as retry_err:
                        last_err = retry_err
                        logger.warning(f'Snapshot download attempt {attempt + 1}/3 failed: {retry_err}')
                        continue

                if snapshot_dir is None:
                    raise last_err or RuntimeError('All snapshot download attempts failed')

                # Try parquet files first (standard Arrow-backed datasets)
                parquet_pattern = os.path.join(snapshot_dir, 'data', f'{split}-*.parquet')
                parquet_files = sorted(_glob.glob(parquet_pattern))
                if parquet_files:
                    logger.info(
                        f'Found {len(parquet_files)} parquet file(s) for split={split} in {snapshot_dir}. '
                        f'Loading via datasets (parquet loader) ...'
                    )
                    dataset = datasets.load_dataset('parquet', data_files=parquet_files, split='train')
                    return dataset

                # No matching parquet found – fall back to loading the snapshot dir as a local dataset
                logger.info(
                    f'No parquet files found for split={split} in {snapshot_dir}. '
                    f'Trying to load as local dataset directory ...'
                )
                load_kwargs = dict(
                    path=snapshot_dir,
                    name=subset if subset != 'default' else None,
                    split=split,
                    revision=version,
                    download_mode=hf_download_mode,
                    **kwargs,
                )
                if 'trust_remote_code' in inspect.signature(datasets.load_dataset).parameters:
                    load_kwargs['trust_remote_code'] = trust_remote
                dataset = datasets.load_dataset(**load_kwargs)
                return dataset
            except Exception as fallback_err:
                logger.error(f'Fallback snapshot download also failed: {fallback_err}')
                raise  # re-raise the original DatasetGenerationError

    if data_source in [HubType.HUGGINGFACE, HubType.LOCAL]:
        # Hugging Face datasets may fail on local mirrors that contain a stale dataset_infos.json.
        dataset_infos_path = os.path.join(data_id_or_path, 'dataset_infos.json')
        if os.path.exists(dataset_infos_path):
            logger.info(f'Removing dataset_infos.json file at {dataset_infos_path} to avoid datasets errors.')
            os.remove(dataset_infos_path)

        load_kwargs = dict(
            path=data_id_or_path,
            name=subset if subset != 'default' else None,
            split=split,
            revision=version,
            download_mode=hf_download_mode,
            **kwargs,
        )
        if 'trust_remote_code' in inspect.signature(datasets.load_dataset).parameters:
            load_kwargs['trust_remote_code'] = trust_remote
        return datasets.load_dataset(**load_kwargs)

    raise ValueError(f'Unsupported dataset hub: {data_source}')


def download_dataset_file(
    data_id_or_path: str,
    file_path: str,
    data_source: Optional[str] = HubType.MODELSCOPE,
    revision: Optional[str] = None,
    force_redownload: bool = False,
    cache_dir: Optional[str] = None,
) -> str:
    """Download or resolve a single file from a dataset hub."""
    data_source = data_source or HubType.MODELSCOPE

    if data_source == HubType.HUGGINGFACE:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(
            repo_id=data_id_or_path,
            filename=file_path,
            repo_type='dataset',
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_redownload,
        )

    if data_source == HubType.MODELSCOPE:
        from modelscope import dataset_snapshot_download

        download_kwargs = {'allow_file_pattern': file_path}
        if revision:
            download_kwargs['revision'] = revision
        snapshot_dir = dataset_snapshot_download(data_id_or_path, **download_kwargs)
        resolved_path = os.path.join(snapshot_dir, file_path)
        if not os.path.exists(resolved_path):
            raise FileNotFoundError(f'Dataset file {file_path} was not found in {snapshot_dir}.')
        return resolved_path

    if data_source == HubType.LOCAL:
        root_dir = os.path.abspath(data_id_or_path)
        resolved_path = os.path.abspath(os.path.join(root_dir, file_path))
        if os.path.commonpath([root_dir, resolved_path]) != root_dir:
            raise ValueError(f'Invalid dataset file path: {file_path}')
        if not os.path.exists(resolved_path):
            raise FileNotFoundError(f'Dataset file {file_path} was not found in {root_dir}.')
        return resolved_path

    raise ValueError(f'Unsupported dataset hub: {data_source}')
