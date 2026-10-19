import sqlite3

import pytest

from evalscope.service import dataset_catalog_inventory as inventory


@pytest.fixture(autouse=True)
def temporary_review_store(monkeypatch):
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    conn.executescript(next(sql for version, _, sql in inventory.catalog.db._MIGRATIONS if version == 25))
    monkeypatch.setattr(inventory.catalog.db, '_get_conn', lambda: conn)
    yield
    conn.close()


def test_report_evidence_requires_samples_and_never_authorizes(tmp_path, monkeypatch):
    import json
    report = tmp_path / 'task' / 'reports' / 'model'
    report.mkdir(parents=True)
    (report / 'arc.json').write_text(json.dumps({'dataset_name': 'arc', 'num': 3, 'score': 0}))
    (report / 'empty.json').write_text(json.dumps({'dataset_name': 'empty', 'num': 0}))
    (report / 'bad.json').write_text('broken')
    monkeypatch.setattr(inventory.catalog.db, '_db_path', str(tmp_path / 'metadata.db'))
    evidence = inventory.report_evidence()
    assert evidence['arc']['samples'] == 3
    assert 'empty' not in evidence
    monkeypatch.setattr(inventory, 'definitions', lambda: [dict(benchmark='arc', category='llm', dataset_id='arc')])
    monkeypatch.setattr(inventory.catalog, 'list_entries', lambda **kw: {'items': []})
    item = inventory.list_inventory()['items'][0]
    assert item['historical_test']['samples'] == 3
    assert not item['published'] and item['versions'] == []


def test_empty_cache_keeps_all_categories(monkeypatch):
    monkeypatch.setattr(inventory, 'definitions', lambda: [
        {'benchmark': name, 'category': category, 'dataset_id': name}
        for name, category in [('text', 'llm'), ('vision', 'vlm'), ('search', 'rag')]])
    monkeypatch.setattr(inventory.catalog, 'list_entries', lambda **kw: {'items': []})
    result = inventory.list_inventory()
    assert len(result['items']) == 3
    assert {item['category'] for item in result['items']} == {'llm', 'vlm', 'rag'}
    assert all(item['versions'] == [] and item['status'] == 'not_registered' for item in result['items'])
    assert all('id' not in item for item in result['items'])


def test_versions_merge_and_unknown_separate(monkeypatch):
    monkeypatch.setattr(inventory, 'definitions', lambda: [dict(benchmark='text', category='llm', dataset_id='source')])
    known = dict(id=1, benchmark='text', category='llm', status='available', published=True)
    unknown = dict(id=2, benchmark='unmapped:cache', category='llm', status='candidate', published=False)
    monkeypatch.setattr(inventory.catalog, 'list_entries', lambda **kw: {'items': [known, unknown]})
    result = inventory.list_inventory()
    assert result['items'][0]['versions'] == [known]
    assert result['items'][0]['published'] is False  # Legacy version publication is not dataset approval.
    assert result['unmapped'] == [unknown]


def test_real_definitions_cover_registered_metadata():
    from evalscope.utils.resource_utils import load_benchmark_data
    actual = {(item['category'], item['benchmark']) for item in inventory.definitions()}
    expected = {(info['meta']['category'], name) for name, info in load_benchmark_data().items()
                if info.get('meta', {}).get('category') in ('llm', 'vlm', 'agent')}
    assert expected <= actual
    assert any(category == 'rag' for category, _ in actual)
    assert len(actual) == len(inventory.definitions())


def test_inventory_route_is_admin_only(monkeypatch):
    from flask import Flask
    from evalscope.service.blueprints.dataset_catalog import bp_dataset_catalog
    from evalscope.service.blueprints import auth
    app = Flask(__name__)
    app.register_blueprint(bp_dataset_catalog, url_prefix='/api/v1/dataset-catalog')
    monkeypatch.setattr(inventory, 'list_inventory', lambda: {'items': [], 'unmapped': [], 'total': 0})
    monkeypatch.setattr(auth, 'get_current_role', lambda: 'user')
    assert app.test_client().get('/api/v1/dataset-catalog/inventory').status_code == 403
    monkeypatch.setattr(auth, 'get_current_role', lambda: 'admin')
    response = app.test_client().get('/api/v1/dataset-catalog/inventory')
    assert response.status_code == 200 and response.is_json
    assert response.json['unmapped'] == []
