"""Dataset-level verdicts use temporary SQLite and a blueprint-only Flask app."""
import json
import sqlite3
from datetime import datetime

import pytest

from tests.test_dataset_catalog import env, saved, scan
from evalscope.service import db, dataset_catalog as catalog, dataset_catalog_inventory as inventory
from evalscope.service.blueprints import auth

URL = '/api/v1/dataset-catalog/inventory/manual-review'


@pytest.fixture
def review_env(env, monkeypatch):
    monkeypatch.setattr(auth, 'get_current_user_id', lambda: 7)
    monkeypatch.setattr(inventory, 'definitions', lambda: [
        {'category': category, 'benchmark': 'arc', 'dataset_id': 'arc'}
        for category in ('llm', 'vlm', 'rag')])
    monkeypatch.setattr(inventory, 'report_evidence', lambda: {})
    return env


def verdict(client, passed, category='llm'):
    return client.post(URL, json={'category': category, 'benchmark': 'arc', 'passed': passed})


def test_no_cache_verdict_persists_and_never_creates_or_publishes_version(review_env, monkeypatch):
    client, _ = review_env
    monkeypatch.setattr(catalog, '_manifest', lambda *a: pytest.fail('must not inspect cache'))
    monkeypatch.setattr(catalog, 'scan', lambda: pytest.fail('must not scan'))
    assert all(item['manual_review'] is None for item in inventory.list_inventory()['items'])
    for passed in (True, False, True):
        response = verdict(client, passed)
        assert response.status_code == 200
        review = response.json['manual_review']
        assert review['passed'] is passed and review['user_id'] == 7
        assert datetime.fromisoformat(review['at']).utcoffset() is not None
        item = inventory.list_inventory()['items'][0]
        assert item['manual_review'] == review
        assert item['status'] == 'not_registered' and item['versions'] == [] and not item['published']
        stored = db._get_conn().execute('SELECT * FROM dataset_catalog_manual_reviews').fetchall()
        assert len(stored) == 1 and bool(stored[0]['passed']) is passed
        assert not db._get_conn().execute('SELECT * FROM dataset_catalog').fetchall()
    assert catalog.list_entries() == {'items': [], 'total': 0}
    assert all(item['manual_review'] is None for item in inventory.list_inventory()['items'][1:])


@pytest.mark.parametrize('body', [None, [], 'bad', {},
    {'category': 'llm', 'benchmark': 'arc', 'passed': 1},
    {'category': 'llm', 'benchmark': 'arc', 'passed': 'false'},
    {'category': [], 'benchmark': 'arc', 'passed': True},
    {'category': 'llm', 'benchmark': {}, 'passed': True},
    {'category': 'other', 'benchmark': 'arc', 'passed': True},
    {'category': 'llm', 'benchmark': ' arc', 'passed': True},
    {'category': 'llm', 'benchmark': '', 'passed': True},
    {'category': 'llm', 'benchmark': 'a' * 201, 'passed': True},
    {'category': 'llm', 'benchmark': 'arc', 'passed': True, 'user_id': 9}])
def test_strict_body(review_env, body):
    client, _ = review_env
    assert client.post(URL, json=body).status_code == 400
    assert not db._get_conn().execute('SELECT * FROM dataset_catalog_manual_reviews').fetchall()


def test_unknown_dataset_and_admin_identity(review_env, monkeypatch):
    client, _ = review_env
    assert client.post(URL, json={'category': 'llm', 'benchmark': 'missing', 'passed': True}).status_code == 404
    for role in ('user', None):
        monkeypatch.setattr(auth, 'get_current_role', lambda: role)
        assert verdict(client, True).status_code == 403
    monkeypatch.setattr(auth, 'get_current_role', lambda: 'admin')
    for actor in (None, True, 0, -1, '7'):
        monkeypatch.setattr(auth, 'get_current_user_id', lambda: actor)
        assert verdict(client, True).status_code == 403


def test_veto_revokes_all_versions_and_blocks_manual_auto_and_stale_resolve(review_env):
    client, root = review_env
    saved(root)
    ident = scan(client)[0]['id']
    assert client.post(f'/api/v1/dataset-catalog/{ident}/validate').status_code == 200
    assert catalog.publish(ident, True)['item']['published']
    original = catalog._get(ident)
    fingerprint = catalog._item(original)['fingerprint']
    assert catalog.resolve_selection(ident, fingerprint)['id'] == ident
    assert catalog.resolve_published('arc', 'llm')['id'] == ident

    def clone(conn, category, validator):
        details = json.loads(original['details_json'])
        details['validator'] = validator
        return conn.execute('''INSERT INTO dataset_catalog
            (benchmark,category,source,dataset_id,subset,split,cache_path,status,published,
             manifest_json,details_json,created_at,updated_at)
            VALUES (?,?,?,?,?,?,?,'available',1,?,?,?,?)''',
            ('arc', category, original['source'], original['dataset_id'], 'another', 'test',
             original['cache_path'], original['manifest_json'], json.dumps(details),
             original['created_at'], original['updated_at'])).lastrowid
    manual_id = db._write(lambda conn: clone(conn, 'llm', 'manual-cache-v1'))
    other_id = db._write(lambda conn: clone(conn, 'vlm', 'audited-cache-v2'))
    assert verdict(client, False).status_code == 200
    assert not catalog._get(ident)['published'] and not catalog._get(manual_id)['published']
    assert catalog._get(other_id)['published']
    assert not catalog._fresh(original)  # even a previously fetched/published row is vetoed
    assert inventory.list_inventory()['items'][0]['manual_review']['passed'] is False
    for version_id in (ident, manual_id):
        assert client.post(f'/api/v1/dataset-catalog/{version_id}/publish', json={'published': True}).status_code == 409
    # Existing version-level manual reapproval cannot override dataset veto.
    assert catalog.manual_review(ident, True, 7)['item']['status'] == 'available'
    assert client.post(f'/api/v1/dataset-catalog/{ident}/publish', json={'published': True}).status_code == 409
    # Defense in depth if an old/external writer leaves a stale publication flag.
    db._write(lambda conn: conn.execute('UPDATE dataset_catalog SET published=1 WHERE category=?', ('llm',)))
    assert not any(item['category'] == 'llm' for item in catalog.list_entries()['items'])
    with pytest.raises(catalog.CatalogError):
        catalog.resolve_selection(ident, fingerprint)
    with pytest.raises(catalog.CatalogError):
        catalog.resolve_published('arc', 'llm')
    verdict(client, False)
    assert verdict(client, True).status_code == 200
    assert not catalog._get(ident)['published'] and not catalog._get(manual_id)['published']
    assert catalog.publish(ident, True)['item']['published']
    assert catalog.resolve_selection(ident, fingerprint)['id'] == ident


def test_failed_revocation_rolls_back_verdict_and_every_publication(review_env):
    client, root = review_env
    saved(root)
    ident = scan(client)[0]['id']
    catalog.validate(ident)
    catalog.publish(ident, True)
    assert verdict(client, True).status_code == 200
    conn = db._get_conn()
    conn.execute('''CREATE TRIGGER refuse_revocation BEFORE UPDATE OF published ON dataset_catalog
                    BEGIN SELECT RAISE(ABORT, 'forced rollback'); END''')
    with pytest.raises(sqlite3.IntegrityError, match='forced rollback'):
        inventory.manual_review('llm', 'arc', False, 9)
    assert catalog._get(ident)['published']
    stored = conn.execute('SELECT * FROM dataset_catalog_manual_reviews').fetchone()
    assert stored['passed'] == 1 and stored['user_id'] == 7


def test_v24_upgrade_preserves_cache_and_version_review(tmp_path):
    conn = sqlite3.connect(str(tmp_path / 'upgrade.sqlite'))
    conn.row_factory = sqlite3.Row
    try:
        # Apply the actual migration runner to an actual v24 database, never production.
        original = db._MIGRATIONS
        db._MIGRATIONS = [migration for migration in original if migration[0] <= 24]
        try:
            db._migrate(conn)
        finally:
            db._MIGRATIONS = original
        conn.execute('''INSERT INTO dataset_catalog
            (benchmark,category,source,dataset_id,cache_path,details_json,created_at,updated_at)
            VALUES ('arc','llm','modelscope','arc','/temporary/cache',
                    '{"manual_review":{"passed":true}}','now','now')''')
        conn.commit()
        db._migrate(conn)
        assert db._get_schema_version(conn) == db._MIGRATIONS[-1][0]
        assert json.loads(conn.execute('SELECT details_json FROM dataset_catalog').fetchone()[0]) == {
            'manual_review': {'passed': True}}
        assert conn.execute('SELECT count(*) FROM dataset_catalog_manual_reviews').fetchone()[0] == 0
        db._migrate(conn)
        assert conn.execute('SELECT count(*) FROM dataset_catalog').fetchone()[0] == 1
    finally:
        conn.close()
