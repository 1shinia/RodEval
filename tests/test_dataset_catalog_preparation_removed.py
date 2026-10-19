"""Version generation is retired; existing cache readers remain available."""
from flask import Flask

from evalscope.service import dataset_catalog, dataset_catalog_mteb
from evalscope.service.blueprints.dataset_catalog import bp_dataset_catalog


def test_version_generation_entrypoints_removed():
    app = Flask(__name__)
    app.register_blueprint(bp_dataset_catalog, url_prefix='/api/v1/dataset-catalog')
    assert not any(rule.rule.endswith('/prepare-mteb') for rule in app.url_map.iter_rules())
    assert not hasattr(dataset_catalog, 'prepare_mteb')
    assert not hasattr(dataset_catalog_mteb, 'prepare_snapshot')
    assert callable(dataset_catalog.scan)
    assert callable(dataset_catalog_mteb.load_snapshot)
    assert callable(dataset_catalog_mteb.load_published_tasks)
