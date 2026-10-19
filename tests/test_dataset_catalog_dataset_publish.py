"""Dataset publication contract; only isolated SQLite and blueprint requests."""
import sqlite3

import pytest
from flask import Flask

from evalscope.service import db, dataset_catalog as catalog, dataset_catalog_inventory as inventory
from evalscope.service.blueprints import auth
from evalscope.service.blueprints.dataset_catalog import bp_dataset_catalog

BASE = '/api/v1/dataset-catalog'


@pytest.fixture
def client(monkeypatch):
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    for version, _, sql in db._MIGRATIONS:
        if version >= 24:
            conn.executescript(sql)
    monkeypatch.setattr(db, '_get_conn', lambda: conn)
    monkeypatch.setattr(auth, 'get_current_role', lambda: 'admin')
    monkeypatch.setattr(auth, 'get_current_user_id', lambda: 7)
    monkeypatch.setattr(inventory, 'definitions', lambda: [
        dict(category=category, benchmark='shared', dataset_id=f'{category}/data',
             task_type='Retrieval' if category == 'rag' else None)
        for category in ('llm', 'vlm', 'rag')])
    monkeypatch.setattr(inventory, 'report_evidence', lambda: {})
    monkeypatch.setattr(catalog, '_manifest', lambda *a: pytest.fail('no cache inspection'))
    monkeypatch.setattr(catalog, 'scan', lambda: pytest.fail('no cache scan'))
    app = Flask(__name__)
    app.register_blueprint(bp_dataset_catalog, url_prefix=BASE)
    yield app.test_client()
    conn.close()


def review(client, passed, category='llm'):
    return client.post(BASE + '/inventory/manual-review', json=dict(
        category=category, benchmark='shared', passed=passed))


def publish(client, published, category='llm'):
    return client.post(BASE + '/inventory/publish', json=dict(
        category=category, benchmark='shared', published=published))


def test_no_cache_publication_lifecycle_and_safe_directory(client, monkeypatch):
    assert client.get(BASE + '/published-datasets').json == {'items': [], 'total': 0}
    assert publish(client, True).status_code == 409
    assert review(client, True).status_code == 200
    assert client.get(BASE + '/published-datasets').json['items'] == []
    response = publish(client, True)
    assert response.status_code == 200
    assert response.json == dict(category='llm', benchmark='shared', published=True,
                                 manual_review=review_data())
    assert publish(client, True).json == response.json
    expected = dict(response.json, dataset_id='llm/data', task_type=None)
    assert inventory.require_dataset_published('llm', 'shared') == expected
    monkeypatch.setattr(auth, 'get_current_role', lambda: 'user')
    listing = client.get(BASE + '/published-datasets')
    assert listing.status_code == 200
    assert listing.json == {'items': [expected], 'total': 1}
    assert set(expected) == {'benchmark', 'category', 'dataset_id', 'task_type', 'published', 'manual_review'}
    assert inventory.list_inventory()['items'][0]['published'] is True
    assert not db._get_conn().execute('SELECT * FROM dataset_catalog').fetchall()
    monkeypatch.setattr(auth, 'get_current_role', lambda: 'admin')
    assert publish(client, False).json['published'] is False
    assert publish(client, False).status_code == 200
    assert client.get(BASE + '/published-datasets').json['items'] == []
    with pytest.raises(catalog.CatalogError) as exc:
        inventory.require_dataset_published('llm', 'shared')
    assert exc.value.status_code == 403


def review_data(category='llm'):
    row = db._get_conn().execute('SELECT * FROM dataset_catalog_manual_reviews WHERE category=?',
                                (category,)).fetchone()
    return dict(passed=bool(row['passed']), user_id=row['user_id'], at=row['at'])


def test_veto_is_scoped_and_reapproval_never_restores(client):
    for category in ('llm', 'vlm', 'rag'):
        assert review(client, True, category).status_code == 200
        assert publish(client, True, category).status_code == 200
    assert review(client, False).status_code == 200
    row = db._get_conn().execute("SELECT * FROM dataset_catalog_manual_reviews WHERE category='llm'").fetchone()
    assert row['passed'] == 0 and row['published'] == 0
    assert publish(client, True).status_code == 409
    assert {i['category'] for i in client.get(BASE + '/published-datasets').json['items']} == {'vlm', 'rag'}
    assert review(client, True).status_code == 200
    assert not inventory.list_inventory()['items'][0]['published']
    with pytest.raises(catalog.CatalogError):
        inventory.require_dataset_published('llm', 'shared')
    assert publish(client, True).status_code == 200
    # Defense against stale flags left by another writer.
    db._write(lambda conn: conn.execute("UPDATE dataset_catalog_manual_reviews SET passed=0 WHERE category='llm'"))
    assert 'llm' not in {i['category'] for i in client.get(BASE + '/published-datasets').json['items']}
    with pytest.raises(catalog.CatalogError):
        inventory.require_dataset_published('llm', 'shared')


def test_veto_and_publication_revocation_are_atomic(client):
    review(client, True)
    publish(client, True)
    conn = db._get_conn()
    conn.execute('''CREATE TRIGGER refuse_legacy_revocation BEFORE UPDATE OF published ON dataset_catalog
                    BEGIN SELECT RAISE(ABORT, 'rollback'); END''')
    conn.execute('''INSERT INTO dataset_catalog
        (benchmark,category,source,dataset_id,cache_path,status,published,created_at,updated_at)
        VALUES ('shared','llm','local','data','/unused','available',1,'now','now')''')
    conn.commit()
    with pytest.raises(sqlite3.IntegrityError, match='rollback'):
        inventory.manual_review('llm', 'shared', False, 9)
    row = conn.execute('SELECT * FROM dataset_catalog_manual_reviews').fetchone()
    assert row['passed'] == 1 and row['published'] == 1 and row['user_id'] == 7
    assert conn.execute('SELECT published FROM dataset_catalog').fetchone()[0] == 1


@pytest.mark.parametrize('body', [None, [], {},
    dict(category='llm', benchmark='shared', published=1),
    dict(category='llm', benchmark='shared', published='false'),
    dict(category=[], benchmark='shared', published=True),
    dict(category='llm', benchmark={}, published=True),
    dict(category='other', benchmark='shared', published=True),
    dict(category='llm', benchmark=' shared', published=True),
    dict(category='llm', benchmark='', published=True),
    dict(category='llm', benchmark='x' * 201, published=True),
    dict(category='llm', benchmark='shared', published=True, cache_path='/unused')])
def test_strict_publish_body(client, body):
    assert client.post(BASE + '/inventory/publish', json=body).status_code == 400
    assert not db._get_conn().execute('SELECT * FROM dataset_catalog_manual_reviews').fetchall()


def test_publish_admin_unknown_dataset_and_guard_inputs(client, monkeypatch):
    for role in ('user', None):
        monkeypatch.setattr(auth, 'get_current_role', lambda: role)
        assert publish(client, True).status_code == 403
    monkeypatch.setattr(auth, 'get_current_role', lambda: 'admin')
    assert client.post(BASE + '/inventory/publish', json=dict(
        category='llm', benchmark='unknown', published=True)).status_code == 404
    for category, benchmark in [([], 'shared'), ('llm', {}), ('other', 'shared')]:
        with pytest.raises(catalog.CatalogError) as exc:
            inventory.require_dataset_published(category, benchmark)
        assert exc.value.status_code == 400
    with pytest.raises(catalog.CatalogError) as exc:
        inventory.require_dataset_published('llm', 'unknown')
    assert exc.value.status_code == 404


def test_unreviewed_unpublish_and_legacy_publish_route_remain_separate(client):
    response = publish(client, False)
    assert response.status_code == 200
    assert response.json == dict(category='llm', benchmark='shared', published=False, manual_review=None)
    assert not db._get_conn().execute('SELECT * FROM dataset_catalog_manual_reviews').fetchall()
    ident = db._write(lambda conn: conn.execute('''INSERT INTO dataset_catalog
        (benchmark,category,source,dataset_id,cache_path,status,published,created_at,updated_at)
        VALUES ('shared','llm','local','data','/unused','candidate',0,'now','now')''').lastrowid)
    assert client.post(BASE + f'/{ident}/publish', json={'published': False}).status_code == 200
    assert client.post(BASE + f'/{ident}/publish', json={'published': True}).status_code == 409
    assert not client.get(BASE + '/published-datasets').json['items']


def test_failed_dataset_revocation_rolls_back_review(client):
    review(client, True)
    publish(client, True)
    conn = db._get_conn()
    conn.execute('''CREATE TRIGGER refuse_dataset_revocation
        BEFORE UPDATE OF published ON dataset_catalog_manual_reviews
        WHEN NEW.published=0 BEGIN SELECT RAISE(ABORT, 'dataset rollback'); END''')
    with pytest.raises(sqlite3.IntegrityError, match='dataset rollback'):
        inventory.manual_review('llm', 'shared', False, 9)
    row = conn.execute('SELECT * FROM dataset_catalog_manual_reviews').fetchone()
    assert row['passed'] == 1 and row['published'] == 1 and row['user_id'] == 7


def test_v25_upgrade_defaults_unpublished_and_preserves_review(tmp_path, monkeypatch):
    conn = sqlite3.connect(str(tmp_path / 'upgrade.sqlite'))
    try:
        with monkeypatch.context() as m:
            m.setattr(db, 'SCHEMA_VERSION', 25)
            db._migrate(conn)
        conn.execute('''INSERT INTO dataset_catalog_manual_reviews(category,benchmark,passed,user_id,at)
                        VALUES ('llm','shared',1,7,'original')''')
        conn.commit()
        db._migrate(conn)
        assert db._get_schema_version(conn) == 27
        assert conn.execute('SELECT passed,user_id,at,published FROM dataset_catalog_manual_reviews').fetchone() == (
            1, 7, 'original', 0)
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute('UPDATE dataset_catalog_manual_reviews SET published=2')
        conn.rollback()
        db._migrate(conn)
        assert conn.execute('SELECT count(*) FROM dataset_catalog_manual_reviews').fetchone()[0] == 1
    finally:
        conn.close()
