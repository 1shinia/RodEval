"""Manual review uses only temporary cache/SQLite, never real datasets."""
import pytest
from tests.test_dataset_catalog import env, saved, scan
from evalscope.service import dataset_catalog as catalog
from evalscope.service.blueprints import auth


@pytest.fixture
def review_env(env, monkeypatch):
    monkeypatch.setattr(auth, 'get_current_user_id', lambda: 7)
    return env


def test_manual_lifecycle_revoke_and_reapprove(review_env):
    client, root = review_env
    saved(root)
    ident = scan(client)[0]['id']
    base = f'/api/v1/dataset-catalog/{ident}'
    item = client.post(base + '/manual-review', json={'passed': True}).json['item']
    assert item['status'] == 'available' and not item['published']
    assert item['details']['manual_review']['user_id'] == 7
    fingerprint = item['fingerprint']
    assert client.post(base + '/publish', json={'published': True}).status_code == 200
    assert catalog.resolve_selection(ident, fingerprint)['id'] == ident
    item = client.post(base + '/manual-review', json={'passed': False}).json['item']
    assert item['status'] == 'invalid' and not item['published']
    assert client.get('/api/v1/dataset-catalog').json['items'] == []
    with pytest.raises(catalog.CatalogError):
        catalog.resolve_selection(ident, fingerprint)
    assert client.post(base + '/publish', json={'published': True}).status_code == 409
    item = client.post(base + '/manual-review', json={'passed': True}).json['item']
    assert item['status'] == 'available' and not item['published']
    assert len(item['details']['manual_history']) == 3
    assert client.post(base + '/publish', json={'published': True}).status_code == 200
    public = client.get('/api/v1/dataset-catalog').json['items'][0]
    assert 'manual_history' not in public.get('details', {})
    (root / 'arc-cache' / 'dataset_info.json').write_text('{}')
    assert client.get('/api/v1/dataset-catalog').json['items'] == []
    assert client.post(base + '/publish', json={'published': True}).status_code == 409


def test_manual_pass_does_not_require_automatic_validator(review_env, monkeypatch):
    client, root = review_env
    saved(root)
    ident = scan(client)[0]['id']
    monkeypatch.setattr(catalog, '_check_cache', lambda *a: ('unsupported', {}))
    response = client.post(f'/api/v1/dataset-catalog/{ident}/manual-review', json={'passed': True})
    assert response.status_code == 200
    assert response.json['item']['details']['validator'] == 'manual-cache-v1'


@pytest.mark.parametrize('body', [{}, {'passed': 1}, {'passed': 'true'}, {'passed': True, 'user_id': 9}])
def test_manual_strict_input(review_env, body):
    client, _ = review_env
    assert client.post('/api/v1/dataset-catalog/1/manual-review', json=body).status_code == 400


def test_manual_admin_identity_required(review_env, monkeypatch):
    client, _ = review_env
    monkeypatch.setattr(auth, 'get_current_role', lambda: 'user')
    assert client.post('/api/v1/dataset-catalog/1/manual-review', json={'passed': True}).status_code == 403
    monkeypatch.setattr(auth, 'get_current_role', lambda: 'admin')
    monkeypatch.setattr(auth, 'get_current_user_id', lambda: None)
    assert client.post('/api/v1/dataset-catalog/1/manual-review', json={'passed': True}).status_code == 403


@pytest.mark.parametrize('failure', ['missing', 'empty', 'unmapped', 'changed'])
def test_manual_pass_cannot_bypass_cache_guards(review_env, monkeypatch, failure):
    client, root = review_env
    path = saved(root)
    ident = scan(client)[0]['id']
    if failure == 'missing':
        monkeypatch.setattr(catalog, '_manifest', lambda *a: (_ for _ in ()).throw(FileNotFoundError()))
    elif failure == 'empty':
        monkeypatch.setattr(catalog, '_load_arrow', lambda *a: (_ for _ in ()).throw(catalog.CatalogError('empty')))
    elif failure == 'unmapped':
        monkeypatch.setattr(catalog, '_cache_specs', lambda: [])
    else:
        original = catalog._load_arrow
        def changed(p):
            result = original(p)
            (path / 'extra').write_text('changed')
            return result
        monkeypatch.setattr(catalog, '_load_arrow', changed)
    response = client.post(f'/api/v1/dataset-catalog/{ident}/manual-review', json={'passed': True})
    assert response.status_code == 409
    assert not catalog._get(ident)['published']


def test_negative_review_works_when_cache_missing(review_env, monkeypatch):
    client, root = review_env
    saved(root)
    ident = scan(client)[0]['id']
    monkeypatch.setattr(catalog, '_manifest', lambda *a: (_ for _ in ()).throw(FileNotFoundError()))
    response = client.post(f'/api/v1/dataset-catalog/{ident}/manual-review', json={'passed': False})
    assert response.status_code == 200
    assert response.json['item']['status'] == 'invalid'
