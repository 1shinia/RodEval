import json
from evalscope.service import dataset_catalog_inventory as inventory


def test_legacy_tasks_keep_own_repository_identity():
    rows = {i['benchmark']: i for i in inventory.definitions() if i['category'] == 'rag'}
    assert rows['STSBenchmark']['dataset_id'] == 'mteb/stsbenchmark-sts'
    assert rows['ToxicConversationsClassification']['dataset_id'] == 'mteb/toxic_conversations_50k'
    assert len(rows) == len([i for i in inventory.definitions() if i['category'] == 'rag'])


def test_oracle_is_dependency_not_approved_version(tmp_path, monkeypatch):
    import sqlite3
    conn = sqlite3.connect(':memory:'); conn.row_factory = sqlite3.Row
    inventory.catalog.db._migrate(conn)
    monkeypatch.setattr(inventory.catalog.db, '_get_conn', lambda: conn)
    monkeypatch.setattr(inventory, 'report_evidence', lambda: {})
    monkeypatch.setattr(inventory, 'definitions', lambda: [dict(category='llm', benchmark='swe_bench_verified', dataset_id='main')])
    info = dict(download_checksums={'hf://datasets/princeton-nlp/SWE-bench_oracle@master/file': {}}, features={k:{} for k in ('text','instance_id','problem_statement')})
    (tmp_path/'dataset_info.json').write_text(json.dumps(info))
    row = dict(id=7, benchmark='unmapped:princeton-nlp_SWE-bench_oracle-old', cache_path=str(tmp_path), dataset_id='oracle', status='candidate')
    monkeypatch.setattr(inventory.catalog, '_safe_path', lambda p: tmp_path)
    monkeypatch.setattr(inventory.catalog, 'list_entries', lambda **kw: {'items':[row]})
    result = inventory.list_inventory()
    assert result['unmapped'] == []
    item = result['items'][0]
    assert item['dependencies'] == [row] and item['versions'] == []
    assert not item['published'] and item['manual_review'] is None
    info['download_checksums'] = {}
    (tmp_path/'dataset_info.json').write_text(json.dumps(info))
    assert inventory.list_inventory()['unmapped']
    conn.close()
