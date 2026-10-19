import sqlite3
from evalscope.service import dataset_catalog as catalog
from evalscope.service import dataset_catalog_inventory as inventory


def test_hf_scan_reconciles_old_rows_without_publication(tmp_path, monkeypatch):
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    catalog.db._migrate(conn)
    monkeypatch.setattr(catalog.db, '_get_conn', lambda: conn)
    monkeypatch.setattr(catalog.db, '_write', lambda fn: fn(conn))
    monkeypatch.setattr(catalog, 'known_cache_roots', lambda: [tmp_path])
    monkeypatch.setattr(catalog, '_cache_specs', lambda: [])
    definitions = [dict(category='rag', benchmark=n, dataset_id='mteb/shared') for n in ('TaskA', 'TaskB')]
    monkeypatch.setattr(inventory, 'definitions', lambda: definitions)
    path = tmp_path / 'datasets--mteb--shared'
    path.mkdir()
    conn.execute('''INSERT INTO dataset_catalog
        (benchmark,category,source,dataset_id,cache_path,created_at,updated_at)
        VALUES (?,?,?,?,?,?,?)''', ('unmapped:'+path.name, 'llm', 'cache', path.name, str(path), 'old', 'old'))
    catalog.scan()
    rows = conn.execute('SELECT * FROM dataset_catalog').fetchall()
    assert {r['benchmark'] for r in rows} == {'TaskA', 'TaskB'}
    assert all(r['category']=='rag' and r['source']=='huggingface' and not r['published'] and r['status']=='candidate' for r in rows)
    catalog.scan()
    assert conn.execute('SELECT count(*) FROM dataset_catalog').fetchone()[0] == 2
    assert conn.execute('SELECT count(*) FROM dataset_catalog_manual_reviews').fetchone()[0] == 0


def test_hf_matching_uses_exact_repository_identity(monkeypatch):
    monkeypatch.setattr(inventory, 'definitions', lambda: [dict(category='rag', benchmark='DifferentName', dataset_id='mteb/repo')])
    specs = catalog._hf_cache_specs()
    assert specs['datasets--mteb--repo'][0]['benchmark'] == 'DifferentName'
    assert 'datasets--mteb--REPO' not in specs
