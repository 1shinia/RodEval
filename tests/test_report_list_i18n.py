"""Guard report-list UI copy against bypassing locale translations."""
import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / 'evalscope/web/src'
TARGETS = [
    'pages/LLMReportsTab.tsx',
    'pages/RAGReportsTab.tsx',
    'pages/AIGCReportsTab.tsx',
    'pages/AudioReportsTab.tsx',
    'components/reports/CompareBar.tsx',
]


@pytest.mark.parametrize('relative_path', TARGETS)
def test_report_list_copy_uses_locale(relative_path):
    source = (SRC / relative_path).read_text()
    # Ignore comments, but retain literal JSX, attributes and callback messages.
    source = re.sub(r'/\*.*?\*/|//[^\n]*', '', source, flags=re.S)
    assert not re.search(r'[\u4e00-\u9fff]', source), relative_path
    assert 'useLocale' in source
