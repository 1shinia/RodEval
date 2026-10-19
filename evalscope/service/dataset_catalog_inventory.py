"""Administrator support inventory, separate from publishable cache identities."""
from functools import lru_cache
import json
from pathlib import Path
from . import dataset_catalog as catalog


@lru_cache(maxsize=1)
def definitions():
    from evalscope.utils.resource_utils import load_benchmark_data
    import mteb
    items = []
    for name, info in load_benchmark_data().items():
        meta = info.get('meta', {})
        if meta.get('category') in ('llm', 'vlm', 'agent'):
            items.append(dict(benchmark=name, category=meta['category'],
                              dataset_id=meta.get('dataset_id') or name,
                              validation_supported=name in ('arc', 'gsm8k', 'mmlu', 'chartqa', 'pope')))
    from .dataset_catalog_mteb import SUPPORTED_TYPES
    tasks = list(mteb.get_tasks(modalities=['text'], exclusive_modality_filter=True, exclude_aggregate=True))
    # These legacy tasks remain selectable in the RAG evaluation form.
    names = {task.metadata.name for task in tasks}
    tasks.extend(task for task in mteb.get_tasks(tasks=[
        'STSBenchmark', 'ToxicConversationsClassification']) if task.metadata.name not in names)
    for task in tasks:
        meta = task.metadata
        items.append(dict(benchmark=meta.name, category='rag', dataset_id=meta.dataset.get('path', meta.name),
                          task_type=meta.type, validation_supported=meta.type in SUPPORTED_TYPES))
    return sorted(items, key=lambda item: (item['category'], item['benchmark'].lower()))


def report_evidence():
    """Read sample-bearing historical reports; never grant cache authorization."""
    if not catalog.db._db_path:
        return {}
    root = Path(catalog.db._db_path).parent.resolve()
    evidence = {}
    for path in root.glob('*/reports/*/*.json'):
        try:
            if not path.resolve().is_relative_to(root) or path.stat().st_size > 10_000_000:
                continue
            report = json.loads(path.read_text())
            name, samples = report.get('dataset_name'), report.get('num')
            if not isinstance(name, str) or type(samples) is not int or samples <= 0:
                continue
            modified = path.stat().st_mtime
            if name not in evidence or modified > evidence[name]['timestamp']:
                evidence[name] = {'samples': samples, 'timestamp': modified,
                                  'report': str(path.relative_to(root))}
        except (OSError, ValueError, AttributeError):
            continue
    return evidence


def _review_item(row):
    return {'passed': bool(row['passed']), 'user_id': row['user_id'], 'at': row['at']}


def _definition(category, benchmark):
    if (not isinstance(category, str) or category not in ('llm', 'vlm', 'rag', 'agent')
            or not isinstance(benchmark, str) or not benchmark or len(benchmark) > 200
            or benchmark != benchmark.strip()):
        raise catalog.CatalogError('category 或 benchmark 无效。', 400)
    for item in definitions():
        if item['category'] == category and item['benchmark'] == benchmark:
            return item
    raise catalog.CatalogError('目录数据集不存在。', 404)


def _publication(row):
    return {'category': row['category'], 'benchmark': row['benchmark'],
            'published': bool(row['published']) and bool(row['passed']),
            'manual_review': _review_item(row)}


def _published_item(definition, row):
    return dict(_publication(row), dataset_id=definition['dataset_id'],
                task_type=definition.get('task_type'))


def publish(category, benchmark, published):
    """Publish a manually approved directory dataset, independently of cache versions."""
    _definition(category, benchmark)
    if type(published) is not bool:
        raise catalog.CatalogError('published 必须为布尔值。', 400)

    def write(conn):
        row = conn.execute('''SELECT * FROM dataset_catalog_manual_reviews
            WHERE category=? AND benchmark=?''', (category, benchmark)).fetchone()
        if published and (row is None or not row['passed']):
            raise catalog.CatalogError('数据集须经管理员人工通过后才能发布。', 409)
        if row is None:
            return {'category': category, 'benchmark': benchmark,
                    'published': False, 'manual_review': None}
        conn.execute('''UPDATE dataset_catalog_manual_reviews SET published=?
            WHERE category=? AND benchmark=?''', (int(published), category, benchmark))
        return dict(_publication(row), published=published)

    return catalog.db._write(write)


def list_published_datasets():
    rows = {(row['category'], row['benchmark']): row for row in catalog.db._get_conn().execute(
        'SELECT * FROM dataset_catalog_manual_reviews WHERE published=1 AND passed=1').fetchall()}
    items = [_published_item(item, rows[(item['category'], item['benchmark'])])
             for item in definitions() if (item['category'], item['benchmark']) in rows]
    return {'items': items, 'total': len(items)}


def require_dataset_published(category, benchmark):
    """Authorize by directory identity; never inspect or require a cached version."""
    definition = _definition(category, benchmark)
    row = catalog.db._get_conn().execute('''SELECT * FROM dataset_catalog_manual_reviews
        WHERE category=? AND benchmark=? AND published=1 AND passed=1''',
        (category, benchmark)).fetchone()
    if row is None:
        raise catalog.CatalogError('数据集尚未发布或未通过管理员人工审核。', 403)
    return _published_item(definition, row)


def manual_review(category, benchmark, passed, actor):
    """Attest to a directory dataset, without reading or manufacturing caches."""
    _definition(category, benchmark)
    if type(passed) is not bool:
        raise catalog.CatalogError('passed 必须为布尔值。', 400)
    if type(actor) is not int or actor <= 0:
        raise catalog.CatalogError('需要已登录管理员身份。', 403)
    now = catalog.utc_now_iso()

    def write(conn):
        conn.execute('''INSERT INTO dataset_catalog_manual_reviews(category,benchmark,passed,user_id,at)
            VALUES (?,?,?,?,?) ON CONFLICT(category,benchmark) DO UPDATE SET
            passed=excluded.passed,user_id=excluded.user_id,at=excluded.at''',
            (category, benchmark, int(passed), actor, now))
        # The verdict and all revocations commit (or roll back) together.
        if not passed:
            conn.execute('''UPDATE dataset_catalog_manual_reviews SET published=0
                WHERE category=? AND benchmark=?''', (category, benchmark))
            conn.execute('''UPDATE dataset_catalog SET published=0,updated_at=?
                WHERE category=? AND benchmark=?''', (now, category, benchmark))

    catalog.db._write(write)
    return {'category': category, 'benchmark': benchmark,
            'manual_review': {'passed': passed, 'user_id': actor, 'at': now}}


def list_inventory():
    historical = report_evidence()
    reviews = {(row['category'], row['benchmark']): row
               for row in catalog.db._get_conn().execute('SELECT * FROM dataset_catalog_manual_reviews').fetchall()}
    registered = catalog.list_entries(admin=True)['items']
    grouped = {}
    unmapped = []
    for version in registered:
        if version['benchmark'].startswith('unmapped:'):
            unmapped.append(version)
        else:
            grouped.setdefault((version['category'], version['benchmark']), []).append(version)
    dependencies = {}
    for version in unmapped[:]:
        if not version['benchmark'].startswith('unmapped:princeton-nlp_SWE-bench_oracle-'):
            continue
        try:
            path = catalog._safe_path(version['cache_path'])
            info = json.loads((path / 'dataset_info.json').read_text())
            sources = info.get('download_checksums', {})
            if (not any(key.startswith('hf://datasets/princeton-nlp/SWE-bench_oracle@') for key in sources)
                    or not {'instance_id', 'text', 'problem_statement'} <= set(info.get('features', {}))):
                continue
        except (OSError, ValueError, catalog.CatalogError):
            continue
        # Auxiliary context belongs to single-turn adapters, not agentic tasks.
        for definition in definitions():
            if definition['benchmark'] in ('swe_bench', 'swe_bench_lite', 'swe_bench_verified', 'swe_bench_verified_mini'):
                dependencies.setdefault((definition['category'], definition['benchmark']), []).append(version)
        if dependencies:
            unmapped.remove(version)
    items = []
    for definition in definitions():
        versions = grouped.pop((definition['category'], definition['benchmark']), [])
        status = 'not_registered'
        for candidate in ('available', 'candidate', 'stale', 'invalid', 'missing', 'unsupported'):
            if any(version['status'] == candidate for version in versions):
                status = candidate
                break
        row = reviews.get((definition['category'], definition['benchmark']))
        review = _review_item(row) if row is not None else None
        items.append(dict(definition, versions=versions, dependencies=dependencies.get((definition['category'], definition['benchmark']), []), status=status, manual_review=review,
                          historical_test=historical.get(definition['benchmark']),
                          published=bool(row is not None and row['passed'] and row['published'])))
    for versions in grouped.values():
        unmapped.extend(versions)
    for version in unmapped:
        if not version.get('details', {}).get('reason'):
            reason = ('数据源未关联到当前目录任务。' if version.get('dataset_id', '').startswith('datasets--')
                      else '缓存格式或用途尚未确认，可能为评测辅助数据。')
            version['details'] = dict(version.get('details', {}), reason=reason)
    return {'items': items, 'unmapped': unmapped, 'total': len(items)}
