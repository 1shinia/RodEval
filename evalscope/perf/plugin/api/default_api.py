import aiohttp
import codecs
import json
import sys
import time
import traceback
from typing import Any, Dict

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.api.base import ApiPluginBase
from evalscope.perf.utils.benchmark_util import BenchmarkData
from evalscope.utils.logger import get_logger

logger = get_logger()


class StreamedResponseHandler:
    """Handles streaming HTTP responses by accumulating chunks until complete
    messages are available."""

    def __init__(self):
        self.buffer = ''
        # Keep decoder state across chunks to handle split multibyte sequences
        self.decoder = codecs.getincrementaldecoder('utf-8')()

    def add_chunk(self, chunk_bytes: bytes) -> list[str]:
        """Add a chunk of bytes to the buffer and return any complete
        messages."""
        # Use incremental decoding so incomplete multibyte sequences don't error
        try:
            chunk_str = self.decoder.decode(chunk_bytes, final=False)
        except UnicodeDecodeError:
            # Bad bytes: drop them and reset decoder state to avoid corruption
            self.decoder.reset()
            chunk_str = chunk_bytes.decode('utf-8', errors='ignore')
        # Normalize CRLF (common in SSE implementations) to LF so downstream
        # splitting on "\n\n" works consistently.
        self.buffer += chunk_str.replace('\r\n', '\n')

        messages = []

        # Split by double newlines (SSE message separator)
        while '\n\n' in self.buffer:
            message, self.buffer = self.buffer.split('\n\n', 1)
            message = message.strip()
            if message:
                messages.append(message)

        # if self.buffer is not empty, check if it is a complete message
        # by removing data: prefix and check if it is a valid JSON
        if self.buffer.startswith('data:'):
            message_content = self.buffer.removeprefix('data:').strip()
            if message_content == '[DONE]':
                messages.append(self.buffer.strip())
                self.buffer = ''
            elif message_content:
                try:
                    json.loads(message_content)
                    messages.append(self.buffer.strip())
                    self.buffer = ''
                except json.JSONDecodeError:
                    # Incomplete JSON, wait for more chunks.
                    pass

        return messages


class DefaultApiPlugin(ApiPluginBase):
    """Default implementation of API plugin with common HTTP handling methods."""

    def __init__(self, param: Arguments):
        super().__init__(param)

    async def process_request(
        self, client_session: aiohttp.ClientSession, url: str, headers: Dict, body: Dict
    ) -> BenchmarkData:
        """Process the HTTP request and handle the response.

        Args:
            client_session: The aiohttp client session
            url: The request URL
            headers: The request headers
            body: The request body

        Returns:
            BenchmarkData: Aggregated benchmarking data for the request/response.
        """
        headers = {'Content-Type': 'application/json', **headers}
        data = json.dumps(body, ensure_ascii=False)  # serialize to JSON

        output = BenchmarkData()
        first_generated_timestamp = None
        generated_text = ''
        st = time.perf_counter()
        output.start_time = st
        output.request = data
        most_recent_generated_timestamp = None
        try:
            async with client_session.post(url=url, data=data, headers=headers) as response:
                content_type = response.headers.get('Content-Type', '')
                if response.status == 200:
                    # Handle streaming responses (SSE)
                    if 'text/event-stream' in content_type:
                        handler = StreamedResponseHandler()
                        async for chunk_bytes in response.content.iter_any():

                            if not chunk_bytes:
                                continue

                            messages = handler.add_chunk(chunk_bytes)
                            for message in messages:
                                # NOTE: SSE comments (often used as pings) start with
                                # a colon. These are not JSON data payload and should
                                # be skipped.
                                if message.startswith(':'):
                                    continue

                                chunk = message.removeprefix('data:').strip()

                                if chunk != '[DONE]':
                                    timestamp = time.perf_counter()
                                    data = json.loads(chunk)

                                    if choices := data.get('choices'):
                                        # TTFC is a transport/protocol diagnostic: the first
                                        # SSE choice chunk may be role-only and contain no token.
                                        if output.first_chunk_latency == 0.0:
                                            output.first_chunk_latency = timestamp - st

                                        first_choice = choices[0]
                                        if data.get('object') == 'text_completion':
                                            content = first_choice.get('text') or ''
                                            has_generated_delta = bool(content)
                                        else:
                                            delta = first_choice.get('delta', {}) or {}
                                            content = (delta.get('content') or '') + (delta.get('reasoning_content') or '')
                                            # Tool/function-call deltas are generated model output
                                            # even when there is no textual content.
                                            has_generated_delta = bool(
                                                content or delta.get('tool_calls') or delta.get('function_call')
                                            )

                                        if has_generated_delta:
                                            if first_generated_timestamp is None:
                                                first_generated_timestamp = timestamp
                                                output.first_token_latency = timestamp - st
                                            elif most_recent_generated_timestamp is not None:
                                                output.inter_chunk_latency.append(
                                                    timestamp - most_recent_generated_timestamp
                                                )
                                            output.chunk_times.append(timestamp)
                                            most_recent_generated_timestamp = timestamp

                                        generated_text += content
                                        output.response_messages.append(data)
                                    elif usage := data.get('usage'):
                                        output.prompt_tokens = usage.get('prompt_tokens')
                                        output.completion_tokens = usage.get('completion_tokens')
                                        # Extract real cached tokens from prompt_tokens_details
                                        _details = usage.get('prompt_tokens_details')
                                        if _details and isinstance(_details, dict):
                                            _cached = _details.get('cached_tokens')
                                            if _cached is not None:
                                                output.real_cached_tokens = _cached

                        output.generated_text = generated_text
                        output.success = True
                        # E2E completes when the response stream is actually
                        # exhausted; TPOT uses last_generated_time separately
                        # so usage-only / DONE tail latency is not decode time.
                        response_end_timestamp = time.perf_counter()
                        output.completed_time = response_end_timestamp
                        output.query_latency = response_end_timestamp - st
                        output.last_generated_time = most_recent_generated_timestamp
                        if output.first_chunk_latency == 0.0:
                            output.first_chunk_latency = output.query_latency

                    # Handle non-stream JSON responses
                    elif 'application/json' in content_type or 'application/' in content_type:
                        payload: Any
                        try:
                            payload = await response.json()
                        except Exception:
                            # Fallback to text if JSON parsing fails
                            payload = await response.text()

                        timestamp = time.perf_counter()
                        output.completed_time = timestamp
                        output.query_latency = timestamp - st
                        # For non-stream, first chunk/token can only be observed
                        # when the complete response is available.
                        output.first_chunk_latency = output.query_latency
                        output.first_token_latency = output.query_latency

                        if isinstance(payload, dict):
                            # Extract generated text from choices
                            text = ''
                            if choices := payload.get('choices'):
                                first = choices[0] if choices else {}
                                # Chat Completions format
                                msg = first.get('message') or {}
                                if isinstance(msg, dict) and msg.get('content') is not None:
                                    text = (msg.get('content') or '') + (msg.get('reasoning_content') or '')
                                else:
                                    # Legacy Completions format
                                    text = first.get('text') or ''
                            generated_text = text

                            # Extract usage if provided
                            if usage := payload.get('usage'):
                                output.prompt_tokens = usage.get('prompt_tokens')
                                output.completion_tokens = usage.get('completion_tokens')
                                # Extract real cached tokens from prompt_tokens_details
                                _details = usage.get('prompt_tokens_details')
                                if _details and isinstance(_details, dict):
                                    _cached = _details.get('cached_tokens')
                                    if _cached is not None:
                                        output.real_cached_tokens = _cached

                            output.response_messages.append(payload)
                        else:
                            generated_text = str(payload)

                        output.generated_text = generated_text
                        output.success = True

                    else:
                        # Unknown successful content-type: read as text
                        raw = await response.text()
                        timestamp = time.perf_counter()
                        output.completed_time = timestamp
                        output.query_latency = timestamp - st
                        output.first_chunk_latency = output.query_latency
                        output.first_token_latency = output.query_latency
                        output.generated_text = raw
                        output.response_messages.append(raw)
                        output.success = True
                else:
                    # Try to parse structured error, fallback to reason/text
                    output.status_code = response.status
                    try:
                        err_payload = await response.json()
                        output.error = json.dumps(err_payload, ensure_ascii=False)
                    except Exception:
                        try:
                            output.error = await response.text()
                        except Exception:
                            output.error = response.reason or ''
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = ''.join(traceback.format_exception(*exc_info))
            logger.error(output.error)
        finally:
            # Every request attempt, including HTTP errors, connection errors,
            # JSON/SSE parse failures and timeouts caught here, must have a
            # valid lifecycle.  Throughput/wall-time calculations rely on this
            # invariant and must never see the default 0/0 timestamps.
            if output.completed_time <= output.start_time:
                end = time.perf_counter()
                output.completed_time = end
                output.query_latency = max(0.0, end - output.start_time)

        return output
