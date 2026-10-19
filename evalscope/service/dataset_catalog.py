"""Offline-only dataset inventory, validation and publication.

No download/snapshot API is called. Discovery is not evidence of availability.
Only audited adapter schemas may be published. All rows and required few-shot
caches are checked; embedded images are decoded. Unmapped MTEB caches fail closed.
"""
import hashlib
import json
import os
from collections import Counter
from typing import Any
from pathlib import Path

from . import db
from .time_utils import utc_now_iso

PUBLIC_FIELDS = ('id', 'benchmark', 'category', 'source', 'dataset_id', 'revision',
                 'subset', 'split', 'status', 'published', 'created_at', 'updated_at', 'validated_at')


class CatalogError(ValueError):
    def __init__(self, message, status_code=409):
        super().__init__(message)
        self.status_code = status_code


def known_cache_roots():
    """Server configuration only; never derived from an HTTP request."""
    from evalscope.constants import DEFAULT_EVALSCOPE_CACHE_DIR, DEFAULT_DATASET_CACHE_ROOT
    from huggingface_hub.constants import HF_HUB_CACHE
    from datasets.config import HF_DATASETS_CACHE
    return list(dict.fromkeys(Path(p).expanduser().absolute() for p in (
        Path(DEFAULT_EVALSCOPE_CACHE_DIR) / 'datasets', DEFAULT_DATASET_CACHE_ROOT,
        HF_HUB_CACHE, HF_DATASETS_CACHE)))


def _safe_path(path):
    path = Path(path).absolute()
    # Check lexical containment as well as resolved containment. A symlinked
    # candidate cannot promote a directory outside the approved cache boundary.
    for root in known_cache_roots():
        root = Path(root).absolute()
        if path.is_relative_to(root) and path.resolve().is_relative_to(root.resolve()):
            return path
    raise CatalogError('缓存路径越界，拒绝访问。')


def _cache_specs():
    """Reproduce RemoteDataLoader's processed-cache key, without loading remotely."""
    import evalscope.benchmarks  # register metadata; no dataset acquisition
    from evalscope.api.registry import BENCHMARK_REGISTRY
    from evalscope.utils.resource_utils import load_benchmark_data
    specs = []
    for name, info in load_benchmark_data().items():
        category = info.get('meta', {}).get('category')
        meta = BENCHMARK_REGISTRY.get(name)
        if not meta or not meta.dataset_id or category not in ('llm', 'vlm', 'rag', 'agent'):
            continue
        # These audited adapters change the loader subset/split in __init__.
        subsets = [meta.default_subset] if name in ('mmlu', 'chartqa', 'pope') else meta.subset_list
        splits = meta.subset_list if name == 'pope' else [meta.eval_split, meta.train_split]
        for subset in subsets:
            for split in dict.fromkeys(s for s in splits if s):
                specs.append(dict(benchmark=name, category=category, source='modelscope',
                                  dataset_id=meta.dataset_id, revision=None, subset=subset, split=split,
                                  cache_name=_cache_name(meta.dataset_id, split, subset)))
    return specs


def _cache_name(dataset_id, split, subset, revision=None):
    from evalscope.utils.io_utils import gen_hash, safe_filename
    return f'{safe_filename(dataset_id)}-' + gen_hash(f'{dataset_id}{split}{subset}{revision}{dict()}')


def _fingerprint(manifest):
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _item(row, admin=False):
    result = {key: row[key] for key in PUBLIC_FIELDS}
    result['published'] = bool(result['published'])
    result['fingerprint'] = _fingerprint(json.loads(row['manifest_json'])) if row['status'] == 'available' else None
    rows = json.loads(row['details_json']).get('rows')
    if type(rows) is int:
        result['details'] = {'sample_count': rows}
    if admin:
        result.update(cache_path=row['cache_path'], manifest=json.loads(row['manifest_json']),
                      details=json.loads(row['details_json']))
    return result


def _get(ident):
    row = db._get_conn().execute('SELECT * FROM dataset_catalog WHERE id=?', (ident,)).fetchone()
    if row is None:
        raise CatalogError('数据集记录不存在。', 404)
    return dict(row)


def _manifest(path):
    """Full SHA256 contents, names and symlink targets, not size/mtime heuristics."""
    path = _safe_path(path)
    if not path.is_dir():
        raise FileNotFoundError('缓存目录不存在。')
    entries = []
    for directory, dirs, files in os.walk(path, followlinks=False):
        for name in sorted(dirs + files):
            child = Path(directory) / name
            _safe_path(child)
            # Internal links also fail closed: Arrow dependencies must be local,
            # immutable regular files, not untracked target trees.
            if child.is_symlink():
                raise CatalogError('暂不支持包含符号链接的缓存。')
            if child.is_dir():
                continue
            if not child.is_file():
                raise CatalogError('缓存包含非普通文件。')
            digest = hashlib.sha256()
            before = child.stat()
            with child.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(chunk)
            after = child.stat()
            if (before.st_size, before.st_mtime_ns, before.st_ino) != (
                    after.st_size, after.st_mtime_ns, after.st_ino):
                raise CatalogError('校验过程中缓存发生变化。')
            entries.append({'name': str(child.relative_to(path)), 'size': after.st_size,
                            'sha256': digest.hexdigest()})
    return {'version': 1, 'files': sorted(entries, key=lambda entry: entry['name'])}


def _fresh(row):
    # Dataset-level veto applies equally to automatic and manual cache versions,
    # including callers holding a previously published row/fingerprint.
    review = db._get_conn().execute('''SELECT passed FROM dataset_catalog_manual_reviews
        WHERE category=? AND benchmark=?''', (row['category'], row['benchmark'])).fetchone()
    if review is not None and not review['passed']:
        return False
    if row['status'] != 'available':
        return False
    try:
        if json.loads(row['details_json']).get('validator') not in ('audited-cache-v2', 'manual-cache-v1'):
            return False
        stored = json.loads(row['manifest_json'])
        return stored.get('version') == 2 and stored == _validation_manifest(row)
    except (OSError, ValueError):
        return False


def list_entries(admin=False):
    rows = db._get_conn().execute('SELECT * FROM dataset_catalog ORDER BY category,benchmark,subset,split,id').fetchall()
    items = [_item(row, admin) for row in rows if admin or (row['published'] and _fresh(row))]
    return {'items': items, 'total': len(items)}


def _hf_cache_specs():
    """Map exact HF repository identities to all tasks sharing that source."""
    from .dataset_catalog_inventory import definitions
    specs = {}
    for item in definitions():
        repo = item['dataset_id']
        if item['category'] != 'rag' or not isinstance(repo, str) or len(repo.split('/')) != 2:
            continue
        name = 'datasets--' + repo.replace('/', '--')
        specs.setdefault(name, []).append(dict(
            benchmark=item['benchmark'], category='rag', source='huggingface',
            dataset_id=repo, revision=None, subset='', split=''))
    return specs


def scan():
    specs = {spec['cache_name']: spec for spec in _cache_specs()}
    hf_specs = _hf_cache_specs()
    found = []
    for root in known_cache_roots():
        root = Path(root)
        if not root.is_dir():
            continue
        # One level only: no recursive traversal of unrelated user directories.
        for path in sorted(root.iterdir()):
            try:
                path = _safe_path(path)
                if not path.is_dir():
                    continue
            except (OSError, CatalogError):
                continue
            spec = specs.get(path.name)
            if spec:
                found.append((spec, path))
            elif path.name in hf_specs:
                found.extend((match, path) for match in hf_specs[path.name])
            elif (path / 'testset.json').is_file() or (path / 'testset.jsonl').is_file():
                found.append((dict(benchmark='ragas', category='rag', source='local',
                                   dataset_id='ragas', revision=None, subset='default', split='test'), path))
            elif (path / 'state.json').exists() or path.name.startswith('datasets--'):
                # An unknown cache is visible but is never guessed to be a usable
                # benchmark/task. HF snapshots need an explicit dependency map.
                found.append((dict(benchmark='unmapped:' + path.name, category='llm', source='cache',
                                   dataset_id=path.name, revision=None, subset='', split=''), path))
    now = utc_now_iso()
    def write(conn):
        discovered = 0
        for spec, path in found:
            cursor = conn.execute('''INSERT OR IGNORE INTO dataset_catalog
                (benchmark,category,source,dataset_id,revision,subset,split,cache_path,created_at,updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?)''',
                tuple(spec.get(k) for k in ('benchmark','category','source','dataset_id','revision','subset','split'))
                + (str(path), now, now))
            discovered += cursor.rowcount
            if spec['source'] == 'huggingface' and spec['category'] == 'rag':
                # Retire only an unpublished placeholder for this exact path.
                # No cache file, reviewed version or dataset-level verdict is removed.
                conn.execute('''DELETE FROM dataset_catalog
                    WHERE benchmark=? AND cache_path=? AND source='cache'
                    AND status='candidate' AND published=0''',
                    ('unmapped:' + path.name, str(path)))
        return discovered
    discovered = db._write(write)
    return dict(list_entries(admin=True), discovered=discovered)


def _check_arc(row, path):
    """Read every Arrow row and exercise the real ARC sample converter.

    Only this audited, text-only schema is supported. No arbitrary adapter code
    or generic string/URL heuristics may claim external dependencies complete.
    """
    if row['category'] != 'llm' or row['benchmark'] != 'arc' or row['dataset_id'] != 'allenai/ai2_arc':
        return 'unsupported', {'reason': '尚无此数据集完整依赖校验器（媒体、附属文档及 MTEB 特殊任务暂不支持发布）。'}
    if row['subset'] not in ('ARC-Easy', 'ARC-Challenge') or row['split'] not in ('test', 'train', 'validation'):
        return 'unsupported', {'reason': '不支持的 ARC 子集或分片。'}
    # Audit the Arrow file list BEFORE load_from_disk can follow metadata paths.
    state = json.loads((path / 'state.json').read_text())
    files = state.get('_data_files')
    if not isinstance(files, list) or not files:
        raise CatalogError('Arrow 文件清单缺失。')
    for entry in files:
        name = entry.get('filename') if isinstance(entry, dict) else None
        if not isinstance(name, str) or Path(name).is_absolute() or '..' in Path(name).parts:
            raise CatalogError('Arrow 文件引用越界。')
        dependency = _safe_path(path / name)
        if not dependency.resolve().is_relative_to(path.resolve()) or not dependency.is_file():
            raise CatalogError('Arrow 文件缺失或引用越界。')
    from datasets import Dataset, load_from_disk
    from evalscope.benchmarks.arc.arc_adapter import ARCAdapter
    dataset = load_from_disk(str(path))  # same processed format as RemoteDataLoader
    if not isinstance(dataset, Dataset) or len(dataset) == 0:
        raise CatalogError('缓存必须为非空的已处理 Dataset。')
    if set(dataset.column_names) - {'id', 'question', 'choices', 'answerKey'}:
        return 'unsupported', {'reason': '存在未经审计的附加字段，不能证明媒体/文档依赖完整。'}
    for record in dataset:
        if not isinstance(record.get('question'), str) or not record['question'].strip():
            raise CatalogError('ARC question 缺失。')
        choices = record.get('choices')
        if not isinstance(choices, dict) or set(choices) != {'text', 'label'}:
            raise CatalogError('ARC choices 不完整。')
        texts, labels = choices['text'], choices['label']
        if not isinstance(texts, list) or not isinstance(labels, list) or len(texts) < 2 or len(texts) != len(labels):
            raise CatalogError('ARC choices 长度不合法。')
        if not all(isinstance(v, str) and v.strip() for v in texts + labels) or record.get('answerKey') not in labels:
            raise CatalogError('ARC choices/answerKey 不合法。')
        ARCAdapter.record_to_sample(None, record)
    return 'available', {'validator': 'arc-text-v1', 'rows': len(dataset), 'dependencies': 'self-contained-text'}


def _meta(name):
    import evalscope.benchmarks
    from evalscope.api.registry import BENCHMARK_REGISTRY
    return BENCHMARK_REGISTRY.get(name)


def _dependencies(row):
    meta = _meta(row['benchmark'])
    if (row['benchmark'] in ('gsm8k', 'mmlu') and meta and row['category'] == 'llm'
            and row['dataset_id'] == meta.dataset_id and meta.few_shot_num > 0
            and row['split'] == meta.eval_split):
        return [_safe_path(Path(row['cache_path']).parent / _cache_name(
            row['dataset_id'], meta.train_split, row['subset'], row['revision']))]
    return []


def _validation_manifest(row):
    return {'version': 2, 'cache': _manifest(row['cache_path']),
            'dependencies': [{'name': p.name, 'manifest': _manifest(p)} for p in _dependencies(row)]}


def _load_arrow(path):
    # Audit before datasets can follow state references or decode media paths.
    state = json.loads((path / 'state.json').read_text())
    files = state.get('_data_files')
    if not isinstance(files, list) or not files:
        raise CatalogError('Arrow 文件清单缺失。')
    for entry in files:
        name = entry.get('filename') if isinstance(entry, dict) else None
        if not isinstance(name, str) or Path(name).is_absolute() or '..' in Path(name).parts:
            raise CatalogError('Arrow 文件引用越界。')
        dependency = _safe_path(path / name)
        if not dependency.resolve().is_relative_to(path.resolve()):
            raise CatalogError('Arrow 文件引用越界。')
        if not dependency.is_file():
            raise FileNotFoundError(name)
    from datasets import Dataset, Image, Audio, load_from_disk
    dataset = load_from_disk(str(path))
    if not isinstance(dataset, Dataset) or not len(dataset):
        raise CatalogError('缓存必须为非空的已处理 Dataset。')
    for column, feature in dataset.features.items():
        if isinstance(feature, (Image, Audio)):
            dataset = dataset.cast_column(column, type(feature)(decode=False))
    return dataset


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _check_records(row, path):
    name = row['benchmark']
    dataset = _load_arrow(path)
    allowed = {
        'gsm8k': {'question', 'answer'},
        'mmlu': {'question', 'choices', 'answer', 'subject'},
        'chartqa': {'question', 'image', 'answer', 'type'},
        'pope': {'question', 'image', 'answer', 'id', 'category', 'question_id'},
    }[name]
    if set(dataset.column_names) - allowed:
        return 'unsupported', {'reason': '存在未经审计的附加字段，无法证明依赖完整。'}
    meta = _meta(name)
    from types import SimpleNamespace
    adapter = SimpleNamespace(prompt_template=meta.prompt_template)
    subjects = {}
    for record in dataset:
        if not _text(record.get('question')):
            raise CatalogError('question 缺失。')
        if name == 'gsm8k':
            if not _text(record.get('answer')) or '####' not in record['answer'] or not _text(record['answer'].split('####')[-1]):
                raise CatalogError('GSM8K answer 不完整。')
        elif name == 'mmlu':
            choices = record.get('choices')
            if (not isinstance(choices, list) or len(choices) != 4 or not all(map(_text, choices))
                    or type(record.get('answer')) is not int or not 0 <= record['answer'] < 4
                    or record.get('subject') not in meta.subset_list):
                raise CatalogError('MMLU choices/answer/subject 不合法。')
            subjects[record['subject']] = subjects.get(record['subject'], 0) + 1
        else:
            image = record.get('image')
            # Audited adapters consume bytes, not paths/URLs. Never open media paths.
            if not isinstance(image, dict) or not isinstance(image.get('bytes'), bytes) or not image['bytes']:
                raise CatalogError('媒体必须包含完整内嵌图片 bytes；路径/URL 不足以证明可用。')
            from io import BytesIO
            from PIL import Image
            with Image.open(BytesIO(image['bytes'])) as picture:
                picture.verify()
            with Image.open(BytesIO(image['bytes'])) as picture:
                picture.load()
            if name == 'pope' and (not _text(record.get('answer')) or record['answer'].lower() not in ('yes', 'no')):
                raise CatalogError('POPE answer 不合法。')
            if name == 'chartqa':
                answer = record.get('answer')
                if (not (_text(answer) or isinstance(answer, list) and answer and all(map(_text, answer)))
                        or record.get('type') not in meta.subset_list):
                    raise CatalogError('ChartQA answer/type 不合法。')
                subjects[record['type']] = subjects.get(record['type'], 0) + 1
        meta.data_adapter.record_to_sample(adapter, record)
    return 'available', {'rows': len(dataset), 'subjects': subjects}


def _rag_file(path):
    files = [p for p in (path / 'testset.json', path / 'testset.jsonl') if p.is_file()]
    if len(files) != 1:
        raise CatalogError('RAGAS 必须有且仅有一个 testset.json 或 testset.jsonl。')
    return files[0]


def _check_ragas(path):
    file = _rag_file(path)
    rows = ([json.loads(line) for line in file.read_text().splitlines() if line.strip()]
            if file.suffix == '.jsonl' else json.loads(file.read_text()))
    rows = rows if isinstance(rows, list) else [rows]
    allowed = {'question', 'user_input', 'answer', 'response', 'contexts', 'retrieved_contexts',
               'reference', 'ground_truth'}
    if not rows:
        raise CatalogError('RAGAS testset 为空。')
    for record in rows:
        if not isinstance(record, dict):
            raise CatalogError('RAGAS testset 行必须为对象。')
        if set(record) - allowed:
            return 'unsupported', {'reason': 'RAGAS 附加字段/外部文档尚无依赖校验器。'}
        if not all(_text(record.get(key, record.get(alias))) for key, alias in (
                ('user_input', 'question'), ('response', 'answer'), ('reference', 'ground_truth'))):
            raise CatalogError('RAGAS 问题、回答、参考答案不完整。')
        contexts = record.get('retrieved_contexts', record.get('contexts'))
        if not isinstance(contexts, list) or not contexts or not all(map(_text, contexts)):
            raise CatalogError('RAGAS 已检索上下文缺失。')
    try:
        from evalscope.backend.rag_eval.ragas.task_template import _load_dataset
        _load_dataset(str(file))
    except ImportError:
        return 'unsupported', {'reason': '缺少 RAGAS 可选依赖，无法执行真实 testset loader。'}
    return 'available', {'rows': len(rows), 'dependencies': 'self-contained-ragas-testset'}


def _check_cache(row, path):
    name = row['benchmark']
    from . import dataset_catalog_mteb
    if row['source'] == dataset_catalog_mteb.SOURCE:
        return dataset_catalog_mteb.check_cache(row, path)
    if name == 'ragas' and row['category'] == 'rag' and row['source'] == 'local':
        return 'unsupported', {'reason': '本阶段仅管理文本、多模态与 MTEB 数据；RAGAS 不开放发布。'}
    elif name in ('arc', 'gsm8k', 'mmlu', 'chartqa', 'pope'):
        meta = _meta(name)
        category = 'vlm' if name in ('chartqa', 'pope') else 'llm'
        if row['category'] != category or row['dataset_id'] != meta.dataset_id or row['revision'] is not None or row['source'] not in ('modelscope', 'huggingface'):
            return 'unsupported', {'reason': '未经审计的 category/source/dataset_id/revision。'}
        subsets = [meta.default_subset] if name in ('mmlu', 'chartqa', 'pope') else meta.subset_list
        splits = meta.subset_list if name == 'pope' else [meta.eval_split, meta.train_split]
        if row['subset'] not in subsets or row['split'] not in splits:
            return 'unsupported', {'reason': '不支持的 loader 子集或分片。'}
        status, details = _check_arc(row, path) if name == 'arc' else _check_records(row, path)
        if status == 'available':
            for dependency in _dependencies(row):
                dep_status, dep_details = _check_records(row, dependency)
                if dep_status != 'available':
                    return dep_status, dep_details
                if name == 'mmlu':
                    counts = dep_details['subjects']
                    if any(counts.get(subject, 0) < meta.few_shot_num for subject in details['subjects']):
                        raise CatalogError('MMLU dev 缺少所选 subject 的 few-shot 样本。')
                elif dep_details['rows'] < meta.few_shot_num:
                    raise CatalogError('训练缓存 few-shot 样本不足。')
            if _dependencies(row):
                details['few_shot_num'] = meta.few_shot_num
    else:
        return 'unsupported', {'reason': '无完整依赖校验器；MTEB 任务变换、corpus/queries/qrels、CLIP 和快照暂不支持。'}
    if status == 'available':
        details['validator'] = 'audited-cache-v2'
    return status, details


def validate(ident):
    row = _get(ident)
    manifest = {}
    try:
        path = _safe_path(row['cache_path'])
        before = _validation_manifest(row)
        status, details = _check_cache(row, path)
        after = _validation_manifest(row)
        if before != after:
            raise CatalogError('校验过程中缓存发生变化。')
        if status == 'available':
            manifest = after
    except FileNotFoundError:
        status, details = 'missing', {'reason': '缓存或必需文件缺失。'}
    except Exception as exc:
        # Diagnostic only in admin payload; never exposed in public responses.
        status, details = 'invalid', {'reason': str(exc)}
    previous = json.loads(row['details_json'])
    if previous.get('manual_history'):
        details['manual_history'] = previous['manual_history']
    now = utc_now_iso()
    db._write(lambda conn: conn.execute('''UPDATE dataset_catalog SET status=?,published=0,
        manifest_json=?,details_json=?,validated_at=?,updated_at=? WHERE id=?''',
        (status, json.dumps(manifest), json.dumps(details, ensure_ascii=False), now, now, ident)))
    return {'item': _item(_get(ident), True)}


def manual_review(ident, passed, actor):
    """Administrator attestation, pinned to a loadable, unchanged cache identity."""
    if type(passed) is not bool:
        raise CatalogError('passed 必须为布尔值。', 400)
    if type(actor) is not int or actor <= 0:
        raise CatalogError('需要已登录管理员身份。', 403)

    def write(conn):
        row = _get(ident)
        previous = json.loads(row['details_json'])
        manifest = {}
        details: dict[str, Any] = {}
        if passed:
            before = _validation_manifest(row)
            path = _safe_path(row['cache_path'])
            from .dataset_catalog_mteb import SOURCE
            if row['source'] == SOURCE:
                status, details = _check_cache(row, path)
                if status != 'available':
                    raise CatalogError('固定 RAG 版本结构无效，不能人工放行。')
            else:
                spec = next((spec for spec in _cache_specs() if all(
                    row[key] == spec[key] for key in (
                        'benchmark', 'category', 'source', 'dataset_id', 'revision', 'subset', 'split'))
                    and path.name == spec['cache_name']), None)
                if spec is None:
                    raise CatalogError('缓存尚未映射到可执行版本，不能人工放行。')
                dataset = _load_arrow(path)
                details = {'rows': len(dataset)}
                if row['benchmark'] in ('mmlu', 'chartqa'):
                    column = 'subject' if row['benchmark'] == 'mmlu' else 'type'
                    details['subjects'] = dict(Counter(dataset[column]))
                dependencies = _dependencies(row)
                if dependencies:
                    meta = _meta(row['benchmark'])
                    if meta is None:
                        raise CatalogError('数据集元信息缺失。')
                    for dependency in dependencies:
                        training = _load_arrow(dependency)
                        if len(training) < meta.few_shot_num:
                            raise CatalogError('少样本训练缓存不足，不能人工放行。')
                        if row['benchmark'] == 'mmlu':
                            counts = Counter(training['subject'])
                            if any(counts[subject] < meta.few_shot_num for subject in details['subjects']):
                                raise CatalogError('少样本训练子集不完整，不能人工放行。')
                    details['few_shot_num'] = meta.few_shot_num
            if before != _validation_manifest(row):
                raise CatalogError('人工确认过程中缓存发生变化，请重试。')
            manifest = before
            details['validator'] = 'manual-cache-v1'
        now = utc_now_iso()
        history = previous.get('manual_history', [])
        review = {'passed': passed, 'user_id': actor, 'at': now}
        details.update(manual_review=review, manual_history=[*history, review],
                       automatic_validation=previous.get('automatic_validation', {
                           key: value for key, value in previous.items()
                           if key not in ('manual_review', 'manual_history', 'validator')
                           or key == 'validator' and value != 'manual-cache-v1'}))
        conn.execute('''UPDATE dataset_catalog SET status=?,published=0,manifest_json=?,
            details_json=?,validated_at=?,updated_at=? WHERE id=?''',
            ('available' if passed else 'invalid', json.dumps(manifest),
             json.dumps(details, ensure_ascii=False), now, now, ident))
    try:
        db._write(write)
    except (OSError, ValueError, KeyError) as exc:
        if isinstance(exc, CatalogError):
            raise
        raise CatalogError('缓存缺失或结构无效，无法人工确认通过。') from exc
    return {'item': _item(_get(ident), True)}


def publish(ident, published):
    if type(published) is not bool:
        raise CatalogError('published 必须为布尔值。', 400)
    # Under the DB write lock: concurrent validations/publications cannot publish
    # an older manifest or overwrite a failed revalidation.
    def write(conn):
        row = _get(ident)
        if published and not _fresh(row):
            raise CatalogError('请先完成完整校验；缓存缺失、变化或不受支持时禁止发布。')
        conn.execute('UPDATE dataset_catalog SET published=?,updated_at=? WHERE id=?',
                     (int(published), utc_now_iso(), ident))
    db._write(write)
    return {'item': _item(_get(ident), True)}


def resolve_published(benchmark=None, category=None, *, task=None, subset=None, split=None):
    """Internal evaluation guard. Never accepts caller-supplied path/source.

    Returns one exact published, freshly fingerprinted server-side source. With
    multiple subsets/splits callers MUST pass subset/split; ambiguity fails shut.
    Admin bypass, if desired, belongs explicitly in the evaluation entrypoint.
    """
    name = benchmark or task
    if not isinstance(name, str) or not name or category not in ('llm', 'vlm', 'rag', 'agent'):
        raise CatalogError('必须指定 benchmark 或 task 以及 category。', 400)
    if benchmark and task and benchmark != task:
        raise CatalogError('benchmark 与 task 不匹配。', 400)
    rows = db._get_conn().execute('''SELECT * FROM dataset_catalog
        WHERE benchmark=? AND category=? AND published=1 AND status='available' ORDER BY id''',
        (name, category)).fetchall()
    matches = [dict(row) for row in rows if (subset is None or row['subset'] == subset)
               and (split is None or row['split'] == split) and _fresh(row)]
    if not matches:
        raise CatalogError('数据集未发布、不可用或缓存已变化，请联系管理员。')
    if len(matches) != 1:
        raise CatalogError('存在多个已发布来源，请精确指定 subset 和 split。')
    return _resolved(matches[0])


def resolve_selection(ident, fingerprint):
    """Resolve an exact published record and the manifest selected by the caller."""
    if type(ident) is not int or ident <= 0 or not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise CatalogError('必须提供有效记录 id 和 manifest 指纹。', 400)
    row = _get(ident)
    if not row['published'] or not _fresh(row) or _fingerprint(json.loads(row['manifest_json'])) != fingerprint:
        raise CatalogError('所选记录已撤销、缓存变化或 manifest 指纹不匹配。')
    return _resolved(row)


def _resolved(row):
    result = _item(row, True)
    if row['source'] == 'mteb-snapshot-v1':
        return result
    # Keep original dataset ID/hash. Do not replace it with an Arrow directory
    # (LocalDataLoader expects CSV/JSONL). dataset_dir selects the server cache.
    result['dataset_hub'] = row['source']
    result['dataset_dir'] = str(Path(row['cache_path']).parent.parent)
    meta = _meta(row['benchmark'])
    if row['benchmark'] == 'ragas':
        result['dataset_args'] = {'testset_file': str(_rag_file(Path(row['cache_path']))),
                                  '_catalog_selection': {'id': row['id'], 'fingerprint': result['fingerprint']}}
        result['loader_kwargs'] = {'testset_file': result['dataset_args']['testset_file']}
        return result
    if row['split'] == meta.train_split and row['split'] != meta.eval_split:
        raise CatalogError('训练分片仅用于 few-shot 依赖，不能直接选择执行评测。')
    subsets = (sorted(result['details'].get('subjects', {})) if row['benchmark'] in ('mmlu', 'chartqa')
               else [row['split']] if row['benchmark'] == 'pope' else [row['subset']])
    result['dataset_args'] = dict(dataset_id=row['dataset_id'], subset_list=subsets,
                                  eval_split=row['split'], force_redownload=False,
                                  _catalog_selection={'id': row['id'], 'fingerprint': result['fingerprint']})
    # Preserve benchmark defaults; validated dependencies cover default few-shot.
    if meta.few_shot_num:
        result['dataset_args']['few_shot_num'] = meta.few_shot_num
    result['loader_kwargs'] = dict(data_id_or_path=row['dataset_id'], data_source=row['source'],
                                   version=row['revision'], subset=row['subset'], split=row['split'],
                                   dataset_dir=result['dataset_dir'], force_redownload=False)
    return result
