"""Dataset inventory HTTP API. Authentication is enforced by the service app."""
from functools import wraps

from flask import Blueprint, jsonify, request

from evalscope.service import dataset_catalog as catalog

bp_dataset_catalog = Blueprint('dataset_catalog', __name__)


def _endpoint(admin=False):
    def decorate(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            from .auth import get_current_role
            if admin and get_current_role() != 'admin':
                return jsonify(error='仅管理员可管理数据集。'), 403
            try:
                return jsonify(fn(*args, **kwargs))
            except catalog.CatalogError as exc:
                return jsonify(error=str(exc)), exc.status_code
        return wrapped
    return decorate


def _body():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise catalog.CatalogError('请求体必须为 JSON 对象。', 400)
    return data


@bp_dataset_catalog.get('')
@bp_dataset_catalog.get('/')
@_endpoint()
def public_list():
    return catalog.list_entries()


@bp_dataset_catalog.get('/admin')
@_endpoint(admin=True)
def admin_list():
    return catalog.list_entries(admin=True)


@bp_dataset_catalog.get('/inventory')
@_endpoint(admin=True)
def inventory():
    from ..dataset_catalog_inventory import list_inventory
    return list_inventory()


@bp_dataset_catalog.post('/inventory/manual-review')
@_endpoint(admin=True)
def inventory_manual_review():
    from .auth import get_current_user_id
    from ..dataset_catalog_inventory import manual_review
    body = _body()
    if set(body) != {'category', 'benchmark', 'passed'}:
        raise catalog.CatalogError('必须且只能提供 category、benchmark、passed。', 400)
    return manual_review(body['category'], body['benchmark'], body['passed'], get_current_user_id())


@bp_dataset_catalog.post('/inventory/publish')
@_endpoint(admin=True)
def inventory_publish():
    from ..dataset_catalog_inventory import publish
    body = _body()
    if set(body) != {'category', 'benchmark', 'published'}:
        raise catalog.CatalogError('必须且只能提供 category、benchmark、published。', 400)
    return publish(body['category'], body['benchmark'], body['published'])


@bp_dataset_catalog.get('/published-datasets')
@_endpoint()
def published_datasets():
    from ..dataset_catalog_inventory import list_published_datasets
    return list_published_datasets()


@bp_dataset_catalog.post('/scan')
@_endpoint(admin=True)
def scan():
    if _body():
        raise catalog.CatalogError('扫描不接受任何参数或自定义路径。', 400)
    return catalog.scan()


@bp_dataset_catalog.post('/<int:ident>/validate')
@_endpoint(admin=True)
def validate(ident):
    if request.content_length and _body():
        raise catalog.CatalogError('校验不接受参数。', 400)
    return catalog.validate(ident)


@bp_dataset_catalog.post('/<int:ident>/manual-review')
@_endpoint(admin=True)
def manual_review(ident):
    from .auth import get_current_user_id
    body = _body()
    if set(body) != {'passed'} or type(body['passed']) is not bool:
        raise catalog.CatalogError('必须且只能提供 passed 布尔值。', 400)
    return catalog.manual_review(ident, body['passed'], get_current_user_id())


@bp_dataset_catalog.post('/<int:ident>/publish')
@_endpoint(admin=True)
def publish(ident):
    body = _body()
    if set(body) != {'published'} or type(body['published']) is not bool:
        raise catalog.CatalogError('必须且只能提供 published 布尔值。', 400)
    return catalog.publish(ident, body['published'])
