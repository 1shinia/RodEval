"""JSON response contract for catalog endpoints, independent of UI rendering."""
import pytest
from flask import Flask
from evalscope.service.blueprints.dataset_catalog import bp_dataset_catalog
from evalscope.service import dataset_catalog
from evalscope.service.blueprints import auth


@pytest.mark.parametrize('role, expected', [('admin', 200), ('user', 403)])
def test_admin_list_returns_json(role, expected, monkeypatch):
    app = Flask(__name__)
    app.register_blueprint(bp_dataset_catalog, url_prefix='/api/v1/dataset-catalog')
    monkeypatch.setattr(auth, 'get_current_role', lambda: role)
    monkeypatch.setattr(dataset_catalog, 'list_entries', lambda **kwargs: {'items': [], 'total': 0})
    response = app.test_client().get('/api/v1/dataset-catalog/admin')
    assert response.status_code == expected
    assert response.is_json


@pytest.mark.parametrize('value', [[], 'invalid', {'published': 'yes'}, {'published': True, 'extra': 1}])
def test_invalid_publish_body_is_json_error(value, monkeypatch):
    app = Flask(__name__)
    app.register_blueprint(bp_dataset_catalog, url_prefix='/api/v1/dataset-catalog')
    monkeypatch.setattr(auth, 'get_current_role', lambda: 'admin')
    response = app.test_client().post('/api/v1/dataset-catalog/1/publish', json=value)
    assert response.status_code == 400
    assert response.is_json
