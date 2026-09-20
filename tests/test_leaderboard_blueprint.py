"""Leaderboard blueprint regression tests.

构造一个最小化的 OpenCompass 快照作为 fixture：LLM（新旧两版周期）+ image-vlm
多模态。只挂载 leaderboard 蓝图 + Flask test_client。不触碰生产 DB / 真实快照。
"""
import json
from pathlib import Path

import pytest
from flask import Flask

from evalscope.service.blueprints import leaderboard as lb


def _write(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False))


@pytest.fixture()
def snapshot_root(tmp_path: Path) -> Path:
    """Build a minimal snapshot; return the dir that contains full_snapshot/."""
    snap = tmp_path / 'full_snapshot'
    snap.mkdir(parents=True)

    # LLM —— 新式（TabConfig）
    _write(snap / 'llm-rank__llm-column-v2.26-07.20260724.json', {
        'OverallColumn': {'columns': [
            {'title': {'zh-CN': '模型', 'en-US': 'Model'}, 'width': 220, 'key': 'model', 'fixed': 'left'},
            {'title': {'zh-CN': '均分', 'en-US': 'Average'}, 'width': 100, 'key': 'Average'},
        ], 'groups': {'all': ['model', 'Average']}},
        'TabConfig': [{'key': 'Overall', 'index': 'Overall', 'name': {'zh-CN': '综合评分', 'en-US': 'Overall'},
                       'tableData': 'OverallTable', 'tableColumn': 'OverallColumn'}],
    })
    _write(snap / 'llm-rank__llm-data-v2.26-07.20260724.json', {
        'OverallTable': [{'model': 'A', 'Average': 90}, {'model': 'B', 'Average': 80}], 'models': {},
    })
    # LLM —— 老式（无 TabConfig）
    _write(snap / 'llm-rank__llm-column-v2.24-02.20240220.json', {
        'OverallColumn': {'columns': [{'key': 'model', 'title': {'zh-CN': '模型', 'en-US': 'Model'}}]},
        'MathColumn': {'columns': [{'key': 'Math_College', 'title': {'zh-CN': '大学数学', 'en-US': 'M'}}]},
    })
    _write(snap / 'llm-rank__llm-data-v2.24-02.20240220.json', {
        'OverallTable': [{'model': 'A'}], 'MathTable': [{'Math_College': 1}],
    })

    # 多模态 —— image-vlm 官方榜
    vlm = snap / 'image-vlm'
    vlm.mkdir(parents=True)
    _write(vlm / 'data-image-vlm-ability_official.json', {
        'OverallTable': [
            {'key': 0, 'model': 'Qwen3.7-Plus', 'Avg_Score_wo_Agent': 72.96, 'Perception': 71.0, 'date': '2026/5/20'},
            {'key': 1, 'model': 'Gemini-3.1', 'Avg_Score_wo_Agent': 60.5, 'Perception': 55.0, 'date': '2026/2/19'},
        ],
        'globalData': {},
    })
    _write(vlm / 'column-image-vlm-ability_official.json', {
        'TabConfig': [{'key': 'Overall', 'index': 'Overall', 'name': {'zh-CN': '综合评分', 'en-US': 'Overall'},
                       'tableData': 'OverallTable', 'tableColumn': 'OverallColumn'}],
        'OverallColumn': {'columns': [
            {'key': 'model', 'title': {'zh-CN': '模型', 'en-US': 'Model'}},
            {'key': 'Avg_Score_wo_Agent', 'title': {'zh-CN': '平均分数（不含智能体）', 'en-US': 'Avg (wo agent)'}},
        ], 'groups': {'all': ['model', 'Avg_Score_wo_Agent']}},
    })

    # 智能体 —— agent 榜（带布局文件，验证布局驱动页签 + 只暴露有 data 的页签）
    agent = snap / 'agent'
    agent.mkdir(parents=True)
    _write(agent / 'rankLayout-agent.json', {
        'zh-CN': {
            'hero': {'title': '智能体评测榜单'},
            'ranks': {'sections': [
                {'id': 'ability', 'lineTabs': [
                    {'key': 'openclaw', 'name': {'zh-CN': 'OpenClaw 模型榜'}},
                    {'key': 'hermes', 'name': {'zh-CN': 'Hermes 模型榜'}},
                ]},
            ]},
        },
    })
    _write(agent / 'data-agent-ability_openclaw.json', {
        'OverallTable': [{'key': 0, 'model': 'OpenClaw-1', 'Score': 88.5}],
        'globalData': {},
    })
    _write(agent / 'column-agent-ability_openclaw.json', {
        'TabConfig': [{'key': 'Overall', 'index': 'Overall', 'name': {'zh-CN': '综合评分', 'en-US': 'Overall'},
                       'tableData': 'OverallTable', 'tableColumn': 'OverallColumn'}],
        'OverallColumn': {'columns': [
            {'key': 'model', 'title': {'zh-CN': '模型', 'en-US': 'Model'}},
            {'key': 'Score', 'title': {'zh-CN': '分数', 'en-US': 'Score'}},
        ]},
    })
    return tmp_path


@pytest.fixture()
def client(snapshot_root: Path):
    original = lb._OPENCOMPASS_DIR
    lb._OPENCOMPASS_DIR = str(snapshot_root)   # 指向含 full_snapshot 的目录
    app = Flask(__name__)
    app.register_blueprint(lb.bp_leaderboard)
    yield app.test_client()
    lb._OPENCOMPASS_DIR = original


def test_meta(client):
    data = client.get('/api/v1/leaderboard/meta').get_json()
    assert data['llm_periods'][0]['value'] == '26-07.20260724'
    assert data['default_llm'] == '26-07.20260724'
    assert data['mm_periods'] == []                       # image-vlm 无历史期
    assert data['vlm_tabs'] == [{'value': 'ability_official', 'label': '官方评测榜'}]
    assert data['default_mm'] is None


def test_llm_table_and_tabs(client):
    d = client.get('/api/v1/leaderboard/llm').get_json()
    assert d['tabs'] == [{'key': 'Overall', 'zh': '综合评分', 'en': 'Overall'}]
    tab = d['tables'][0]
    assert tab['key'] == 'Overall'
    assert len(tab['rows']) == 2
    assert [c['key'] for c in tab['columns']] == ['model', 'Average']


def test_multimodal_serves_image_vlm(client):
    d = client.get('/api/v1/leaderboard/multimodal').get_json()
    assert d['tab'] == 'ability_official'
    assert len(d['rows']) == 2
    assert d['rows'][0]['model'] == 'Qwen3.7-Plus'
    assert [c['key'] for c in d['columns']] == ['model', 'Avg_Score_wo_Agent']
    assert d['name'] == '官方评测榜'


def test_multimodal_unknown_tab_falls_back(client):
    d = client.get('/api/v1/leaderboard/multimodal', query_string={'tab': 'debug_xyz'}).get_json()
    assert d['tab'] == 'ability_official'


def test_old_style_llm_period_falls_back_to_flat_tabs(client):
    d = client.get('/api/v1/leaderboard/llm', query_string={'period': '24-02.20240220'}).get_json()
    assert d['period'] == '24-02.20240220'
    assert [t['key'] for t in d['tabs']] == ['Overall', 'Math']


def test_path_traversal_period_falls_back(client):
    r = client.get('/api/v1/leaderboard/llm', query_string={'period': '..%2F..%2Fsecret'})
    assert r.status_code == 200
    assert r.get_json()['period'] == '26-07.20260724'


def test_meta_boards_lists_all(client):
    data = client.get('/api/v1/leaderboard/meta').get_json()
    boards = {b['id']: b for b in data['boards']}
    assert list(boards) == ['llm', 'multimodal', 'agent', 'ai4science', 'physical-intelligence']
    # 多模态（无布局文件，回退到旧中文名表）
    assert boards['multimodal']['tabs'] == [{'value': 'ability_official', 'label': '官方评测榜'}]
    # 智能体（有布局文件，只暴露有 data 的页签：hermes 无 data 故不出现）
    assert boards['agent']['tabs'] == [{'value': 'ability_openclaw', 'label': 'OpenClaw 模型榜'}]
    # 未落盘的新榜：出现在清单里但页签为空
    assert boards['ai4science']['tabs'] == []
    assert boards['physical-intelligence']['tabs'] == []


def test_board_serves_agent(client):
    d = client.get('/api/v1/leaderboard/board', query_string={'board': 'agent'}).get_json()
    assert d['board'] == 'agent'
    assert d['tab'] == 'ability_openclaw'
    assert d['name'] == 'OpenClaw 模型榜'
    assert len(d['rows']) == 1
    assert d['rows'][0]['model'] == 'OpenClaw-1'
    assert [c['key'] for c in d['columns']] == ['model', 'Score']


def test_board_unknown_404(client):
    r = client.get('/api/v1/leaderboard/board', query_string={'board': 'nope'})
    assert r.status_code == 404


def test_board_unknown_tab_falls_back_to_first(client):
    d = client.get('/api/v1/leaderboard/board',
                   query_string={'board': 'agent', 'tab': 'ability_hermes'}).get_json()
    # hermes 无 data → 不在页签清单 → 回退到第一个有效页签 openclaw
    assert d['tab'] == 'ability_openclaw'


def test_extend_cols_surfaces_sorter():
    """列配置里 customizeRender.sorter 是前端默认排序的权威信号，须透出。"""
    config = {'columns': [
        {'key': 'model', 'title': {'zh-CN': '模型'}},
        {'key': 'Score', 'title': {'zh-CN': '总体评分'}, 'customizeRender': {'sorter': 'descend'}},
        {'key': 'Knowledge', 'title': {'zh-CN': '知识'}, 'customizeRender': {'sorter': ''}},
    ]}
    cols = lb._extend_cols(config)
    assert cols[0]['sorter'] is None
    assert cols[1]['sorter'] == 'descend'
    assert cols[2]['sorter'] == ''