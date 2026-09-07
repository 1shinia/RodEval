"""OpenCompass 榜单数据访问蓝图（只读）。

数据来自 opencompass_crawler.py 抓到并落盘的快照目录，前端按
rank.opencompass.org.cn 的「榜单类型 + 分类维度 + 时间段」下拉框交互渲染。
本模块只做：定位最新批次 → 按周期读出 JSON → 归一化成前端友好的结构。
不写库、不落冗余副本、对文件系统只读。
"""
import json
import logging
import os
import re
from pathlib import Path

from flask import Blueprint, jsonify, request

logger = logging.getLogger(__name__.replace('evalscope', 'evalperf'))

bp_leaderboard = Blueprint('leaderboard', __name__, url_prefix='/api/v1/leaderboard')

# 快照目录：默认项目内 leaderboard_data/（随项目走），可用环境变量覆盖。
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_OPENCOMPASS_DIR = os.environ.get('OPENCOMPASS_DATA_DIR', os.path.join(_PROJECT_ROOT, 'leaderboard_data'))

# 文件名模式：full_snapshot 下的归一化文件名。
_LLM_DATA_PAT = re.compile(r'^llm-rank__llm-data-v2\.(.+)\.json$')
_LLM_COL_PAT = re.compile(r'^llm-rank__llm-column-v2\.(.+)\.json$')
_DATE_RE = re.compile(r'(\d{8})')           # 取文件名里最后的 YYYYMMDD 作排序键
_LLM_VARIANT_RE = re.compile(r'^(\d\d)-(\d\d)\.(\d{8})$')


def _latest_batch_dir() -> Path | None:
    """定位榜单纯数据目录 full_snapshot（存在则返回）。"""
    base = Path(_OPENCOMPASS_DIR)
    if (base / 'full_snapshot').is_dir():
        return base / 'full_snapshot'
    # 允许 OPENCOMPASS_DATA_DIR 直接指向 full_snapshot 本身
    if base.is_dir() and base.name == 'full_snapshot':
        return base
    return None


def _period_variants(pattern: re.Pattern) -> list[str]:
    """扫描 full_snapshot，按文件名模式收集周期，按版本日期降序。"""
    root = _latest_batch_dir()
    if root is None:
        return []
    found = []
    for f in root.glob('*.json'):
        m = pattern.match(f.name)
        if not m:
            continue
        variant = m.group(1)
        dm = _DATE_RE.findall(variant)
        sort_key = int(dm[-1]) if dm else 0
        found.append({'variant': variant, 'sort_key': sort_key})
    found.sort(key=lambda e: e['sort_key'], reverse=True)
    return [e['variant'] for e in found]


def _period_label(variant: str) -> str:
    m = _LLM_VARIANT_RE.match(variant)
    if m:
        yy, mm, _ = m.groups()
        return f'20{yy}-{mm}'
    dm = _DATE_RE.findall(variant)
    if dm:
        d = dm[-1]
        return f'{d[:4]}-{d[4:6]}-{d[6:8]}'
    return variant


def _resolve_variant(period: str | None, allowed: list[str]) -> str:
    """period 必须命中已经扫描到的变体，否则用最新的。"""
    if period in allowed:
        return period
    return allowed[0] if allowed else ''


def _read_file(root: Path, prefix: str, variant: str) -> dict | None:
    """在 root 里找前缀+变体对应的 json，不存在返回 None。"""
    # variant 来自目录扫描结果，本身就是真实文件名的一部分，天然安全；
    # 再用 basename 过滤一次防止任何形式拼接逃逸。
    fname = f'{prefix}{variant}.json'
    if fname != os.path.basename(fname):
        return None
    fp = root / fname
    if not fp.is_file():
        return None
    try:
        return json.loads(fp.read_text(encoding='utf-8'))
    except Exception as e:  # 单个文件损坏不拖垮接口
        logger.warning('leaderboard parse failed %s: %s', fp, e)
        return None


def _extend_cols(config) -> list[dict]:
    """把列配置展平为前端友好结构。config.columns: [{key,title{zh-CN,en-US},width}]"""
    out = []
    if not isinstance(config, dict):
        return out
    for c in config.get('columns') or []:
        if not isinstance(c, dict):
            continue
        title = c.get('title') or {}
        out.append({
            'key': c.get('key', ''),
            'zh': title.get('zh-CN') if isinstance(title, dict) else title,
            'en': title.get('en-US') if isinstance(title, dict) else title,
            'width': c.get('width'),
        })
    return out


_VLM_TABS = [
    ('ability_official', '官方评测榜'),
    ('ability_open_source', '开源评测榜'),
    ('ability_arena_votes', '竞技场投票榜'),
    ('security_overall', '安全综合榜'),
]


def _vlm_tabs() -> list[dict]:
    """枚举 image-vlm 目录里已有的有效榜单（官方/开源/竞技场/安全）。"""
    out = []
    for value, label in _VLM_TABS:
        got = _read_image_vlm(value)
        if got is not None and got[1] and got[1][0]['rows']:
            out.append({'value': value, 'label': label})
    return out


def _read_image_vlm(base: str) -> tuple[list, list] | None:
    """读 image-vlm 某榜单的数据+列，返回 (tabs, tables)；文件缺失返回 None。"""
    root = _latest_batch_dir()
    if root is None:
        return None
    sub = root / 'image-vlm'
    data = _read_file(sub, '', f'data-image-vlm-{base}') if sub.is_dir() else None
    col = _read_file(sub, '', f'column-image-vlm-{base}') if sub.is_dir() else None
    if data is None:
        return None
    col = col or {}
    return _derive_tabs(data, col)


@bp_leaderboard.route('/meta', methods=['GET'])
def leaderboard_meta():
    """返回榜单类型 + 各自的时间段（下拉框数据源）。"""
    llm_periods = _period_variants(_LLM_DATA_PAT)
    return jsonify({
        'llm_periods': [{'value': v, 'label': _period_label(v)} for v in llm_periods],
        'mm_periods': [],                       # image-vlm 无历史期，改用 vlm_tabs
        'vlm_tabs': _vlm_tabs(),
        'default_llm': llm_periods[0] if llm_periods else None,
        'default_mm': None,
        'source': f'file://{_OPENCOMPASS_DIR}',
    })


_FALLBACK_TAB_ORDER = ['Overall', 'Subjective', 'Language', 'Knowledge', 'Reason', 'Math', 'Code', 'Agent']
_FALLBACK_TAB_LABELS_ZH = {
    'Overall': '综合评分', 'Subjective': '主观', 'Language': '语言',
    'Knowledge': '知识', 'Reason': '推理', 'Math': '数学', 'Code': '代码', 'Agent': '智能体',
}


def _derive_tabs(data: dict, col: dict) -> tuple[list, list]:
    """构造 {tabs, tables}。

    新版本列文件带 TabConfig（综合/知识/推理/数学/代码）；老版本没有，
    用 data 文件里的 *Table 键 + 列文件的 *Column 键平铺派生。
    """
    tabs, tables = [], []
    seen = set()

    def push(key: str, zh: str, en: str, data_key: str, col_key: str):
        if key in seen or key is None:
            return
        seen.add(key)
        rows = data.get(data_key)
        cols = _extend_cols(col.get(col_key)) if col_key and col_key in col else []
        tabs.append({'key': key, 'zh': zh, 'en': en})
        tables.append({'key': key, 'columns': cols, 'rows': rows if isinstance(rows, list) else []})

    tab_cfg = col.get('TabConfig') or []
    if tab_cfg:
        for t in tab_cfg:
            if not isinstance(t, dict):
                continue
            key = t.get('key') or t.get('index')
            name = t.get('name') or {}
            push(key, name.get('zh-CN') if isinstance(name, dict) else name,
                 name.get('en-US') if isinstance(name, dict) else name,
                 t.get('tableData'), t.get('tableColumn'))
        return tabs, tables
    # 老版本没有 TabConfig：按固定顺序平铺
    for base in _FALLBACK_TAB_ORDER:
        data_key = f'{base}Table'
        if data_key not in data:
            continue
        push(base, _FALLBACK_TAB_LABELS_ZH.get(base, base), base,
             data_key, f'{base}Column')
    # 兜底：data 里还有其它 Table 键也带上
    for k in data:
        if k.endswith('Table') and k not in {f'{b}Table' for b in _FALLBACK_TAB_ORDER}:
            base = k[:-5]
            push(base, _FALLBACK_TAB_LABELS_ZH.get(base, base), base, k, f'{base}Column')
    return tabs, tables


@bp_leaderboard.route('/llm', methods=['GET'])
def leaderboard_llm():
    root = _latest_batch_dir()
    if root is None:
        return jsonify({'error': 'No leaderboard snapshot found'}), 404
    period = _resolve_variant(request.args.get('period'), _period_variants(_LLM_DATA_PAT))
    if not period:
        return jsonify({'error': 'No LLM leaderboard snapshot found'}), 404

    data = _read_file(root, 'llm-rank__llm-data-v2.', period)
    col = _read_file(root, 'llm-rank__llm-column-v2.', period)
    if data is None or col is None:
        return jsonify({'error': f'Snapshot missing for period {period}'}), 404

    tabs, tables = _derive_tabs(data, col)
    return jsonify({'period': period, 'tabs': tabs, 'tables': tables})


@bp_leaderboard.route('/multimodal', methods=['GET'])
def leaderboard_multimodal():
    """多模态榜单 = OpenCompass 图片理解 VLM（image-vlm）官方评测榜。

    通过 ?tab= 切换 官方/开源/竞技场/安全 子榜（默认 官方评测榜）。
    """
    labels = dict(_VLM_TABS)
    tab = request.args.get('tab') or 'ability_official'
    if tab not in {v for v, _ in _VLM_TABS}:
        tab = 'ability_official'
    got = _read_image_vlm(tab)
    if got is None:
        return jsonify({'error': 'No multimodal (image-vlm) snapshot found'}), 404
    tabs, tables = got
    table = tables[0] if tables else {'columns': [], 'rows': []}
    return jsonify({
        'tab': tab,
        'name': labels.get(tab, tab),
        'tabs': tabs,
        'columns': table['columns'],
        'rows': table['rows'],
    })