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
    """把列配置展平为前端友好结构。config.columns: [{key,title{zh-CN,en-US},width,customizeRender{sorter}}]"""
    out = []
    if not isinstance(config, dict):
        return out
    for c in config.get('columns') or []:
        if not isinstance(c, dict):
            continue
        title = c.get('title') or {}
        render = c.get('customizeRender')
        out.append({
            'key': c.get('key', ''),
            'zh': title.get('zh-CN') if isinstance(title, dict) else title,
            'en': title.get('en-US') if isinstance(title, dict) else title,
            'width': c.get('width'),
            'sorter': render.get('sorter') if isinstance(render, dict) else None,
        })
    return out


# 榜种注册表：顶层「榜单类型」下拉的权威清单。
#   llm 由 gateway 月份 + CDN 驱动（无布局文件，period 维度）；
#   其余「OSS 榜」由 rankLayout-<dir>.json 布局文件驱动页签清单（tab 维度）。
# 未来上游新增榜单：在下载脚本的 OSS_BOARDS 加一项 + 此处加一行即可，无需再改接口/前端。
_BOARDS = [
    {'id': 'llm', 'dir': None, 'label': 'LLM 榜单'},
    {'id': 'multimodal', 'dir': 'image-vlm', 'label': '多模态榜单'},
    {'id': 'agent', 'dir': 'agent', 'label': '智能体榜单'},
    {'id': 'ai4science', 'dir': 'ai4science', 'label': '科学智能榜单'},
    {'id': 'physical-intelligence', 'dir': 'physical-intelligence', 'label': '物理智能榜单'},
]

# 无布局文件时（旧快照/测试 fixture）image-vlm 页签的中文名回退表。
_LEGACY_VLM_LABELS = {
    'ability_official': '官方评测榜',
    'ability_open_source': '开源评测榜',
    'ability_arena_votes': '竞技场投票榜',
    'security_overall': '安全综合榜',
}


def _read_layout(dir_name: str) -> dict | None:
    """读 OSS 榜的布局文件 rankLayout-<dir>.json；缺失/损坏返回 None。"""
    root = _latest_batch_dir()
    if root is None:
        return None
    fp = root / dir_name / f'rankLayout-{dir_name}.json'
    if not fp.is_file():
        return None
    try:
        return json.loads(fp.read_text(encoding='utf-8'))
    except Exception as e:
        logger.warning('leaderboard layout parse failed %s: %s', fp, e)
        return None


def _oss_board_tabs(dir_name: str) -> list[dict]:
    """枚举某 OSS 榜的页签清单（value=section_tab，label=中文名）。

    优先读布局文件（权威页签 + 中文名）；无布局时回退为按 data 文件命名扫描。
    只返回「data 文件确实存在」的页签，与前端下拉一一对应。
    """
    root = _latest_batch_dir()
    if root is None:
        return []
    sub = root / dir_name
    if not sub.is_dir():
        return []
    prefix = f'data-{dir_name}-'

    layout = _read_layout(dir_name)
    if layout:
        out = []
        zh = layout.get('zh-CN') or {}
        sections = (zh.get('ranks') or {}).get('sections') or []
        for s in sections:
            if not isinstance(s, dict):
                continue
            sec = s.get('id')
            for t in s.get('lineTabs') or []:
                if not isinstance(t, dict):
                    continue
                key = t.get('key')
                if not sec or not key:
                    continue
                value = f'{sec}_{key}'
                if not (sub / f'{prefix}{value}.json').is_file():
                    continue
                nm = t.get('name') or {}
                label = nm.get('zh-CN') if isinstance(nm, dict) else (nm or value)
                out.append({'value': value, 'label': label or value})
        return out

    out = []
    for f in sorted(sub.glob(f'{prefix}*.json')):
        value = f.name[len(prefix):-len('.json')]
        if not value:
            continue
        label = _LEGACY_VLM_LABELS.get(value, value) if dir_name == 'image-vlm' else value
        out.append({'value': value, 'label': label})
    return out


def _read_oss_board(dir_name: str, tab: str) -> tuple[list, list] | None:
    """读某 OSS 榜某页签的数据+列，返回 (tabs, tables)；文件缺失返回 None。"""
    root = _latest_batch_dir()
    if root is None:
        return None
    sub = root / dir_name
    if not sub.is_dir():
        return None
    data = _read_file(sub, '', f'data-{dir_name}-{tab}')
    if data is None:
        return None
    col = _read_file(sub, '', f'column-{dir_name}-{tab}') or {}
    return _derive_tabs(data, col)


@bp_leaderboard.route('/meta', methods=['GET'])
def leaderboard_meta():
    """返回榜单类型清单（boards）+ 各自的时间段/页签（下拉框数据源）。"""
    llm_periods = _period_variants(_LLM_DATA_PAT)
    vlm_tabs: list[dict] = []
    boards = []
    for b in _BOARDS:
        entry = {'id': b['id'], 'label': b['label']}
        if b['dir']:
            tabs = _oss_board_tabs(b['dir'])
            entry['tabs'] = tabs
            if b['id'] == 'multimodal':
                vlm_tabs = tabs
        boards.append(entry)
    return jsonify({
        'llm_periods': [{'value': v, 'label': _period_label(v)} for v in llm_periods],
        'mm_periods': [],                       # OSS 榜无历史期，改用 boards[].tabs
        'vlm_tabs': vlm_tabs,                   # 向后兼容：= multimodal 榜的页签
        'boards': boards,
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


def _board_by_id(board_id: str):
    """按 id 查 OSS 榜注册项（llm 无 dir，不在此列）。"""
    return next((b for b in _BOARDS if b['id'] == board_id and b['dir']), None)


def _serve_oss_board(board_id: str):
    """通用 OSS 榜取数：按 board + tab 读出数据+列，归一化后返回。"""
    b = _board_by_id(board_id)
    if b is None:
        return jsonify({'error': f'Unknown leaderboard board: {board_id}'}), 404
    tabs = _oss_board_tabs(b['dir'])
    allowed = {t['value'] for t in tabs}
    tab = request.args.get('tab')
    if tab not in allowed:
        tab = tabs[0]['value'] if tabs else ''
    if not tab:
        return jsonify({'error': f'No snapshot for board {board_id}'}), 404
    got = _read_oss_board(b['dir'], tab)
    if got is None:
        return jsonify({'error': f'No snapshot for board {board_id}'}), 404
    tabs_d, tables = got
    table = tables[0] if tables else {'columns': [], 'rows': []}
    labels = {t['value']: t['label'] for t in tabs}
    return jsonify({
        'board': board_id,
        'tab': tab,
        'name': labels.get(tab, tab),
        'tabs': tabs_d,
        'columns': table['columns'],
        'rows': table['rows'],
    })


@bp_leaderboard.route('/board', methods=['GET'])
def leaderboard_board():
    """通用榜单取数：?board=<id>&tab=<页签>。"""
    return _serve_oss_board(request.args.get('board') or 'multimodal')


@bp_leaderboard.route('/multimodal', methods=['GET'])
def leaderboard_multimodal():
    """向后兼容别名：多模态榜单 = board 'multimodal'（image-vlm）。"""
    return _serve_oss_board('multimodal')