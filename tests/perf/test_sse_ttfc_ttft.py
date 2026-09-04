"""SSE synthetic two-chunk test: TTFC (first chunk) must differ from TTFT (first token).

The first SSE chunk may carry only the role delta (no generated content). That
chunk establishes TTFC but must NOT be recorded as TTFT; only the later chunk
with actual content establishes TTFT.
"""
import asyncio
import time
from unittest.mock import AsyncMock

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.api.default_api import DefaultApiPlugin


class _FakeContent:
    def __init__(self, chunks: list[bytes]):
        self._chunks = chunks

    async def iter_any(self):
        for c in self._chunks:
            yield c


class _FakeResponse:
    def __init__(self, chunks: list[bytes]):
        self.status = 200
        self.headers = {'Content-Type': 'text/event-stream'}
        self.content = _FakeContent(chunks)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeSession:
    def __init__(self, chunks: list[bytes]):
        self._chunks = chunks

    def post(self, url: str, data: str, headers: dict):  # sync: aiohttp.post() returns an async CM
        return _FakeResponse(self._chunks)


def test_sse_role_only_first_chunk_ttfc_neq_ttft():
    plugin = DefaultApiPlugin(Arguments(url='http://test.local/v1', api='openai', model='test-model'))
    role_only = b'data: {"choices":[{"index":0,"delta":{"role":"assistant"}}]}\n\n'
    content = b'data: {"choices":[{"index":0,"delta":{"content":"Hello"}}]}\n\n'
    usage = b'data: {"usage":{"prompt_tokens":5,"completion_tokens":1}}\n\n'
    session = _FakeSession([role_only, content, usage])

    data = asyncio.run(plugin.process_request(session, 'http://test.local/v1', {}, {'messages': []}))

    assert data.success is True
    assert data.generated_text == 'Hello'
    assert data.first_chunk_latency > 0.0
    assert data.first_token_latency is not None
    # The role-only chunk arrives first and must not be counted as TTFT.
    assert data.first_token_latency > data.first_chunk_latency
    # A single generated delta means no inter-chunk interval.
    assert data.inter_chunk_latency == []


def test_sse_multiple_content_chunks_produce_icl():
    plugin = DefaultApiPlugin(Arguments(url='http://test.local/v1', api='openai', model='test-model'))
    c1 = b'data: {"choices":[{"index":0,"delta":{"content":"Hello"}}]}\n\n'
    c2 = b'data: {"choices":[{"index":0,"delta":{"content":" world"}}]}\n\n'
    c3 = b'data: {"choices":[{"index":0,"delta":{"content":"!"}}]}\n\n'
    usage = b'data: {"usage":{"prompt_tokens":5,"completion_tokens":3}}\n\n'
    session = _FakeSession([c1, c2, c3, usage])

    data = asyncio.run(plugin.process_request(session, 'http://test.local/v1', {}, {'messages': []}))

    assert data.generated_text == 'Hello world!'
    assert len(data.inter_chunk_latency) == 2
    assert all(x > 0 for x in data.inter_chunk_latency)
    assert data.first_token_latency == data.first_chunk_latency  # first chunk already carried content
