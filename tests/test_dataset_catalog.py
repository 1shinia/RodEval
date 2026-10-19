"""Offline catalogue tests. Only temporary SQLite/cache paths are used."""
import json
import sqlite3
import socket
from pathlib import Path

import pytest
from flask import Flask

from evalscope.service import db
from evalscope.service import dataset_catalog as catalog
from evalscope.service.blueprints import dataset_catalog as routes
from evalscope.service.blueprints import auth


@pytest.fixture
def env(tmp_path, monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError('catalogue must never access the network')
    monkeypatch.setattr(socket.socket, 'connect', no_network)
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    for version, _, sql in db._MIGRATIONS:
        if version in (24, 25, 26):
            conn.executescript(sql)
    monkeypatch.setattr(db, '_get_conn', lambda: conn)
    root = tmp_path / 'cache' / 'datasets'
    root.mkdir(parents=True)
    monkeypatch.setenv('EVALSCOPE_DATASET_OFFLINE', '1')
    monkeypatch.setattr(catalog, 'known_cache_roots', lambda: [root])
    monkeypatch.setattr(catalog, '_cache_specs', lambda: [dict(
        benchmark='arc', category='llm', source='modelscope', dataset_id='allenai/ai2_arc',
        revision=None, subset='ARC-Easy', split='test', cache_name='arc-cache')])
    app = Flask(__name__)
    app.register_blueprint(routes.bp_dataset_catalog, url_prefix='/api/v1/dataset-catalog')
    monkeypatch.setattr(auth, 'get_current_role', lambda: 'admin')
    yield app.test_client(), root
    conn.close()


def saved(root, rows=None):
    from datasets import Dataset
    path = root / 'arc-cache'
    Dataset.from_list(rows or [{'question': 'q', 'choices': {'text': ['a', 'b'], 'label': ['A', 'B']},
                                'answerKey': 'A'}]).save_to_disk(str(path))
    return path


def scan(client):
    response = client.post('/api/v1/dataset-catalog/scan', json={})
    assert response.status_code == 200, response.json
    return response.json['items']


def test_discovery_does_not_publish_and_is_idempotent(env):
    client, root = env
    saved(root)
    items = scan(client)
    assert len(items) == 1
    assert items[0]['status'] == 'candidate'
    assert items[0]['published'] is False
    assert scan(client)[0]['id'] == items[0]['id']
    assert client.get('/api/v1/dataset-catalog').json == {'items': [], 'total': 0}


def test_full_lifecycle_and_safe_resolution(env):
    client, root = env
    path = saved(root)
    ident = scan(client)[0]['id']
    base = f'/api/v1/dataset-catalog/{ident}'
    assert client.post(base + '/publish', json={'published': True}).status_code == 409
    item = client.post(base + '/validate').json['item']
    assert item['status'] == 'available'
    assert item['manifest']['version'] == 2
    assert client.post(base + '/publish', json={'published': True}).status_code == 200
    public = client.get('/api/v1/dataset-catalog').json['items'][0]
    assert not {'cache_path', 'manifest'} & set(public)
    assert public['details'] == {'sample_count': 1}
    resolved = catalog.resolve_published('arc', 'llm', subset='ARC-Easy', split='test')
    assert resolved['cache_path'] == str(path)
    assert resolved['dataset_id'] == 'allenai/ai2_arc'
    assert resolved['dataset_args']['dataset_id'] == 'allenai/ai2_arc'
    assert client.post(base + '/publish', json={'published': False}).status_code == 200
    with pytest.raises(catalog.CatalogError):
        catalog.resolve_published('arc', 'llm')


def test_manifest_detects_same_size_content_change(env):
    client, root = env
    path = saved(root)
    ident = scan(client)[0]['id']
    base = f'/api/v1/dataset-catalog/{ident}'
    client.post(base + '/validate')
    client.post(base + '/publish', json={'published': True})
    info = path / 'dataset_info.json'
    original = info.read_text()
    info.write_text(original.replace('question', 'questioN'))
    assert client.get('/api/v1/dataset-catalog').json['items'] == []
    with pytest.raises(catalog.CatalogError):
        catalog.resolve_published('arc', 'llm')
    assert client.post(base + '/publish', json={'published': True}).status_code == 409


@pytest.mark.parametrize('endpoint,method', [('/admin', 'get'), ('/scan', 'post'),
                                            ('/1/validate', 'post'), ('/1/publish', 'post')])
def test_admin_only(env, monkeypatch, endpoint, method):
    client, _ = env
    monkeypatch.setattr(auth, 'get_current_role', lambda: 'user')
    assert getattr(client, method)('/api/v1/dataset-catalog' + endpoint, json={}).status_code == 403


@pytest.mark.parametrize('body', [{'path': '/tmp'}, {'roots': ['/tmp']}, [], None])
def test_scan_rejects_custom_paths(env, body):
    client, _ = env
    assert client.post('/api/v1/dataset-catalog/scan', json=body).status_code == 400


@pytest.mark.parametrize('body', [{}, {'published': 'true'}, {'published': 1}, {'published': True, 'path': '/tmp'}])
def test_publish_requires_strict_bool(env, body):
    client, _ = env
    assert client.post('/api/v1/dataset-catalog/1/publish', json=body).status_code == 400


def test_files_are_not_proof_of_complete_dataset(env):
    client, root = env
    path = root / 'arc-cache'
    path.mkdir()
    (path / 'random.txt').write_text('not a dataset')
    ident = scan(client)[0]['id']
    item = client.post(f'/api/v1/dataset-catalog/{ident}/validate').json['item']
    assert item['status'] != 'available'


def test_escape_symlink_is_rejected(env, tmp_path):
    client, root = env
    outside = tmp_path / 'outside'
    outside.mkdir()
    (root / 'arc-cache').symlink_to(outside, target_is_directory=True)
    assert scan(client) == []


def test_nested_escape_symlink_is_rejected(env, tmp_path):
    client, root = env
    path = saved(root)
    outside = tmp_path / 'secret'
    outside.write_text('secret')
    (path / 'attachment').symlink_to(outside)
    ident = scan(client)[0]['id']
    assert client.post(f'/api/v1/dataset-catalog/{ident}/validate').json['item']['status'] != 'available'


@pytest.mark.parametrize('category', ['vlm', 'rag'])
def test_special_unverified_categories_cannot_publish(env, monkeypatch, category):
    client, root = env
    saved(root)
    specs = catalog._cache_specs()
    specs[0]['category'] = category
    monkeypatch.setattr(catalog, '_cache_specs', lambda: specs)
    ident = scan(client)[0]['id']
    base = f'/api/v1/dataset-catalog/{ident}'
    assert client.post(base + '/validate').json['item']['status'] == 'unsupported'
    assert client.post(base + '/publish', json={'published': True}).status_code == 409


def test_media_reference_in_text_cache_not_silently_accepted(env):
    client, root = env
    saved(root, [{'question': 'q', 'image': 'https://invalid.example/a.png'}])
    ident = scan(client)[0]['id']
    assert client.post(f'/api/v1/dataset-catalog/{ident}/validate').json['item']['status'] != 'available'


def test_unknown_entry(env):
    client, _ = env
    assert client.post('/api/v1/dataset-catalog/999/validate').status_code == 404


def test_schema_constraints(env):
    with pytest.raises(sqlite3.IntegrityError):
        db._get_conn().execute("INSERT INTO dataset_catalog (benchmark,category,source,dataset_id,cache_path,created_at,updated_at) VALUES ('a','bad','x','x','x','x','x')")


def test_arrow_metadata_cannot_escape_root(env):
    client, root = env
    path = saved(root)
    state_path = path / 'state.json'
    state = json.loads(state_path.read_text())
    state['_data_files'][0]['filename'] = '../outside.arrow'
    state_path.write_text(json.dumps(state))
    ident = scan(client)[0]['id']
    assert client.post(f'/api/v1/dataset-catalog/{ident}/validate').json['item']['status'] == 'invalid'


def test_resolver_rejects_ambiguous_or_mismatched_identity(env):
    client, root = env
    saved(root)
    ident = scan(client)[0]['id']
    client.post(f'/api/v1/dataset-catalog/{ident}/validate')
    client.post(f'/api/v1/dataset-catalog/{ident}/publish', json={'published': True})
    with pytest.raises(catalog.CatalogError):
        catalog.resolve_published('arc', 'vlm')
    with pytest.raises(catalog.CatalogError):
        catalog.resolve_published('arc', 'llm', task='other')
    assert catalog.resolve_published(task='arc', category='llm')['id'] == ident


def test_real_specs_match_remote_loader_key():
    from evalscope.utils.io_utils import gen_hash, safe_filename
    spec = next(spec for spec in catalog._cache_specs()
                if spec['benchmark'] == 'arc' and spec['subset'] == 'ARC-Easy' and spec['split'] == 'test')
    assert spec['cache_name'] == safe_filename('allenai/ai2_arc') + '-' + gen_hash(
        'allenai/ai2_arctestARC-EasyNone{}')


def prepare(env, monkeypatch, name, rows, *, train_rows=None):
    from datasets import Dataset
    client, root = env
    meta = catalog._meta(name)
    subset = meta.default_subset if name in ('mmlu', 'chartqa', 'pope') else meta.subset_list[0]
    split = meta.subset_list[0] if name == 'pope' else meta.eval_split
    path = root / catalog._cache_name(meta.dataset_id, split, subset)
    Dataset.from_list(rows).save_to_disk(str(path))
    if train_rows is not None:
        Dataset.from_list(train_rows).save_to_disk(str(root / catalog._cache_name(
            meta.dataset_id, meta.train_split, subset)))
    monkeypatch.setattr(catalog, '_cache_specs', lambda: [dict(
        benchmark=name, category='vlm' if name in ('chartqa', 'pope') else 'llm', source='modelscope',
        dataset_id=meta.dataset_id, revision=None, subset=subset, split=split, cache_name=path.name)])
    ident = scan(client)[0]['id']
    return ident, path


@pytest.mark.parametrize('name', ['gsm8k', 'mmlu'])
@pytest.mark.parametrize('train_count,expected', [(None, 'missing'), (1, 'invalid'), (5, 'available')])
def test_text_requires_real_fewshot_cache(env, monkeypatch, name, train_count, expected):
    record = ({'question': 'q', 'answer': 'reason #### 2'} if name == 'gsm8k' else
              {'question': 'q', 'choices': ['a', 'b', 'c', 'd'], 'answer': 0, 'subject': 'anatomy'})
    train = [record] * train_count if train_count else None
    ident, path = prepare(env, monkeypatch, name, [record], train_rows=train)
    item = catalog.validate(ident)['item']
    assert item['status'] == expected, item
    if expected != 'available':
        with pytest.raises(catalog.CatalogError):
            catalog.publish(ident, True)
        return
    catalog.publish(ident, True)
    resolved = catalog.resolve_selection(ident, item['fingerprint'])
    assert resolved['dataset_args']['few_shot_num'] == (4 if name == 'gsm8k' else 5)
    assert resolved['dataset_args']['subset_list'] == (['main'] if name == 'gsm8k' else ['anatomy'])
    assert resolved['dataset_args']['_catalog_selection'] == {'id': ident, 'fingerprint': item['fingerprint']}
    # Exercise the actual cache key and record converter without hub access.
    from evalscope.api.dataset import RemoteDataLoader
    result = RemoteDataLoader(**resolved['loader_kwargs'], sample_fields=catalog._meta(name).data_adapter.record_to_sample.__get__(
        object(), catalog._meta(name).data_adapter)).load()
    assert len(result) == 1
    from evalscope.service.dataset_catalog_policy import apply_catalog_policy
    from evalscope.service.utils import benchmarks
    monkeypatch.setattr(benchmarks, 'load_benchmark_data', lambda _: {
        name: {'meta': {'category': 'llm', 'few_shot_num': catalog._meta(name).few_shot_num,
                        'train_split': catalog._meta(name).train_split}}})
    from evalscope.service import dataset_catalog_inventory as inventory
    monkeypatch.setattr(inventory, 'require_dataset_published', lambda *args: {})
    policy_result = apply_catalog_policy({'datasets': [name], 'dataset_args': {name: {
        'subset_list': resolved['dataset_args']['subset_list'],
        '_catalog_selection': [{'id': ident, 'fingerprint': item['fingerprint']}]}}})
    assert policy_result['dataset_args'][name]['subset_list'] == resolved['dataset_args']['subset_list']
    assert '_catalog_selection' not in policy_result['dataset_args'][name]
    assert apply_catalog_policy(policy_result) == policy_result
    dependency = catalog._dependencies(catalog._get(ident))[0] / 'dataset_info.json'
    dependency.write_text(dependency.read_text() + ' ')
    with pytest.raises(catalog.CatalogError):
        catalog.resolve_selection(ident, item['fingerprint'])


def image_bytes():
    from io import BytesIO
    from PIL import Image
    buffer = BytesIO()
    Image.new('RGB', (2, 2), color='red').save(buffer, format='PNG')
    return buffer.getvalue()


@pytest.mark.parametrize('name', ['chartqa', 'pope'])
@pytest.mark.parametrize('media,expected', [('valid', 'available'), ('corrupt', 'invalid'), ('path', 'invalid')])
def test_vlm_checks_every_embedded_image(env, monkeypatch, name, media, expected):
    record = {'question': 'q', 'answer': 'yes' if name == 'pope' else ['2'],
              'image': {'bytes': image_bytes(), 'path': None}}
    if name == 'chartqa':
        record['type'] = 'human_test'
    broken = dict(record, image={'bytes': b'broken' if media == 'corrupt' else None,
                                 'path': '/must/not/read.png'})
    ident, _ = prepare(env, monkeypatch, name, [record, record if media == 'valid' else broken])
    item = catalog.validate(ident)['item']
    assert item['status'] == expected, item
    if expected == 'available':
        catalog.publish(ident, True)
        assert catalog.resolve_selection(ident, item['fingerprint'])['category'] == 'vlm'
    else:
        with pytest.raises(catalog.CatalogError):
            catalog.publish(ident, True)


@pytest.mark.parametrize('jsonl', [False, True])
def test_ragas_processed_testset_uses_real_loader(env, jsonl):
    client, root = env
    path = root / 'prepared-rag'
    path.mkdir()
    rows = [{'question': 'q', 'answer': 'a', 'contexts': ['complete context'], 'ground_truth': 'a'}]
    file = path / ('testset.jsonl' if jsonl else 'testset.json')
    file.write_text(json.dumps(rows[0] if jsonl else rows))
    item = scan(client)[0]
    assert item['category'] == 'rag'
    validated = catalog.validate(item['id'])['item']
    assert validated['status'] == 'unsupported', validated
    with pytest.raises(catalog.CatalogError):
        catalog.publish(item['id'], True)


def test_exact_selection_rejects_old_fingerprint_and_revoked_record(env):
    client, root = env
    saved(root)
    ident = scan(client)[0]['id']
    item = catalog.validate(ident)['item']
    catalog.publish(ident, True)
    assert catalog.resolve_selection(ident, item['fingerprint']) == catalog.resolve_published('arc', 'llm')
    with pytest.raises(catalog.CatalogError):
        catalog.resolve_selection(ident, '0' * 64)
    with pytest.raises(catalog.CatalogError):
        catalog.resolve_selection(True, item['fingerprint'])
    catalog.publish(ident, False)
    with pytest.raises(catalog.CatalogError):
        catalog.resolve_selection(ident, item['fingerprint'])


def test_scan_specs_use_authoritative_metadata_and_adapter_keys(monkeypatch):
    from evalscope.utils import resource_utils
    original = resource_utils.load_benchmark_data
    metadata = original()
    metadata['gsm8k']['meta']['category'] = 'rag'
    monkeypatch.setattr(resource_utils, 'load_benchmark_data', lambda: metadata)
    specs = catalog._cache_specs()
    assert {s['category'] for s in specs if s['benchmark'] == 'gsm8k'} == {'rag'}
    assert {s['subset'] for s in specs if s['benchmark'] == 'mmlu'} == {'all'}
    assert {s['subset'] for s in specs if s['benchmark'] == 'chartqa'} == {'default'}
    assert {s['split'] for s in specs if s['benchmark'] == 'pope'} == {'popular', 'adversarial', 'random'}
    assert any(s['benchmark'] == 'pope' for s in specs)  # not in UI default list


def test_unknown_processed_cache_stays_unsupported(env):
    client, root = env
    from datasets import Dataset
    Dataset.from_list([{'text': 'file is not proof'}]).save_to_disk(str(root / 'unknown'))
    ident = scan(client)[0]['id']
    item = catalog.validate(ident)['item']
    assert item['status'] == 'unsupported'
    with pytest.raises(catalog.CatalogError):
        catalog.publish(ident, True)
