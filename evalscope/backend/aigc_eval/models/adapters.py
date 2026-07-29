"""AIGC API adapters — decouple 'how to call this API' from model logic.

Architecture:
    Model class ──uses──▶ AIGCApiAdapter ──HTTP──▶ API provider
                          ├── OpenAICompatibleAdapter (default)
                          └── (future: Volcengine, DashScope, etc.)

Each adapter handles: URL resolution, payload construction, response parsing,
and async polling. Returns raw media bytes; model classes handle PIL/frame conversion.
"""

import base64
import json
import logging
import time as _time
from abc import ABC, abstractmethod
from io import BytesIO
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__.replace('evalscope', 'evalperf'))


# ═══════════════════════════════════════════════════════════════════════
# Abstract base
# ═══════════════════════════════════════════════════════════════════════

class AIGCApiAdapter(ABC):
    """Base for API provider adapters.

    Subclasses implement the 'wire protocol' for a specific provider:
    URL construction, payload format, response extraction, async polling.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_base: str = config.get('api_base', '')
        self.api_key: str = config.get('api_key', '')
        self.model_name: str = config.get('model_name_or_path', '')

    # ── Public API ─────────────────────────────────────────────────

    def generate(self, prompt: str, **params) -> bytes:
        """Make API call and return raw media bytes (PNG/JPEG/MP4).

        Args:
            prompt: Text prompt.
            **params: tool, width, height, seed, num_frames, fps,
                      resolution, ratio, negative_prompt, etc.
        Returns:
            Raw bytes of the generated image or video.
        """
        headers = self._build_headers()
        url = self._resolve_url(params.get('tool', 'txt2img'))
        payload = self._build_payload(prompt, **params)
        logger.debug('Adapter POST %s payload keys=%s', url, list(payload.keys()))

        response = requests.post(url, json=payload, headers=headers, timeout=300)

        # Allow subclass to retry on failure (e.g. size format fallback)
        if not response.ok:
            response = self._handle_retry(response, url, headers, payload, **params)

        response.raise_for_status()
        data = response.json()

        # Handle async video generation (task_id + status pattern)
        if self._is_async_response(data):
            data = self._poll_async(data, url, headers)

        return self._extract_bytes(data)

    # ── Subclass overrides ────────────────────────────────────────

    @abstractmethod
    def _resolve_url(self, tool: str) -> str:
        """Build the full API endpoint URL."""
        ...

    @abstractmethod
    def _build_payload(self, prompt: str, **params) -> Dict[str, Any]:
        """Construct the JSON payload for the API request."""
        ...

    @abstractmethod
    def _extract_bytes(self, data: Dict[str, Any]) -> bytes:
        """Extract raw media bytes from the API response JSON."""
        ...

    def _build_headers(self) -> Dict[str, str]:
        """Build HTTP headers."""
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        return headers

    def _is_async_response(self, data: Dict[str, Any]) -> bool:
        """Check if the response indicates an async task (task_id + status)."""
        has_id = 'task_id' in data or 'id' in data
        has_status = 'status' in data
        return has_id and has_status

    def _handle_retry(
        self, response, url: str, headers: dict, payload: dict, **params
    ) -> requests.Response:
        """Optional retry hook. Subclass can override to e.g. fall back size format."""
        return response

    def _poll_async(
        self, data: Dict[str, Any], post_url: str, headers: Dict[str, str]
    ) -> Dict[str, Any]:
        """Poll an async task until complete. Returns the final response data.

        Subclass must override for providers with custom polling logic.
        Default: simple /{task_id} polling with 5s interval."""
        task_id = data.get('task_id', data.get('id', ''))
        status = data.get('status', '')
        logger.info('Async task %s: status=%s', task_id, status)

        if status in ('completed', 'succeeded', 'done'):
            return data

        poll_url = self._resolve_poll_url(post_url, task_id)
        max_attempts = 60  # 5 minutes max

        for attempt in range(max_attempts):
            _time.sleep(5)
            resp = requests.get(poll_url, headers=headers, timeout=30)
            if not resp.ok:
                logger.warning('Poll %d: HTTP %d', attempt + 1, resp.status_code)
                continue
            data = resp.json()
            # Unwrap common API wrappers: {code, message, data: {...}}
            data = self._unwrap_response(data)
            status = data.get('status', '')
            progress = data.get('progress', 0)
            logger.info('Task %s: status=%s progress=%d%%', task_id, status, progress)

            if status in ('completed', 'succeeded', 'done'):
                logger.info('Task %s completed', task_id)
                return self._fetch_async_content(data, task_id, headers)
            elif status in ('failed', 'error', 'cancelled'):
                raise RuntimeError(f'Task {task_id} failed: {data}')

        raise TimeoutError(f'Task {task_id} did not complete within {max_attempts * 5}s')

    def _resolve_poll_url(self, post_url: str, task_id: str) -> str:
        """Build the polling URL from the POST URL and task ID.

        Default: append task_id to the POST URL path (matching old behavior).
        """
        return post_url.rstrip('/') + '/' + task_id

    @staticmethod
    def _unwrap_response(data: Dict[str, Any]) -> Dict[str, Any]:
        """Unwrap common API response wrappers like {code, message, data: {...}}."""
        # {data: {...}} pattern — unwrap until we find the inner payload
        while 'data' in data and isinstance(data['data'], dict):
            if 'status' in data['data']:
                data = data['data']
            else:
                break
        return data

    def _fetch_async_content(
        self, data: Dict[str, Any], task_id: str, headers: Dict[str, str]
    ) -> Dict[str, Any]:
        """Download async task content. Returns data dict with b64_json or url key."""
        # Try common response fields
        for key in ('url', 'video_url', 'b64_json', 'content'):
            if key in data:
                return data
        for key in ('result', 'video', 'output'):
            if key in data and isinstance(data[key], (str, dict)):
                return data
        return data


# ═══════════════════════════════════════════════════════════════════════
# OpenAI-compatible adapter (default, covers 90% of API platforms)
# ═══════════════════════════════════════════════════════════════════════

class OpenAICompatibleAdapter(AIGCApiAdapter):
    """Standard OpenAI-compatible images/video generation API.

    Supports standard path appending: {base}/images/generations or /video/generations.
    Custom overrides via config: endpoint_template, param_aliases, response_path,
    async_poll_url, async_content_url.
    """

    # ── URL resolution ────────────────────────────────────────────

    def _resolve_url(self, tool: str) -> str:
        # Check for custom endpoint template first
        template = self.config.get('endpoint_template')
        if template:
            if template == 'passthrough':
                return self.api_base
            return self.api_base.rstrip('/') + '/' + template.lstrip('/')

        base = self.api_base.rstrip('/')
        # Heuristic: if base already looks like an endpoint, use as-is
        keywords = ['/generations', '/variations', '/edits', '/video', '/images']
        if any(k in base for k in keywords):
            return base

        if tool == 'txt2video':
            return f'{base}/video/generations'
        return f'{base}/images/generations'

    # ── Payload construction ──────────────────────────────────────

    def _build_payload(self, prompt: str, **params) -> Dict[str, Any]:
        tool = params.get('tool', 'txt2img')
        width = params.get('width', 1024)
        height = params.get('height', 1024)

        payload: Dict[str, Any] = {
            'model': self.model_name,
            'prompt': prompt,
            'n': 1,
            'size': f'{width}x{height}',
        }

        if tool == 'txt2video':
            num_frames = params.get('num_frames', 16)
            fps = params.get('fps', 8)
            duration = num_frames // fps if fps > 0 else 5
            if duration >= 2:
                payload['duration'] = duration
            if params.get('resolution'):
                payload['resolution'] = params['resolution']
            if params.get('ratio'):
                payload['ratio'] = params['ratio']
        elif tool == 'img2img':
            if params.get('strength'):
                payload['strength'] = params['strength']
            ref_b64 = params.get('reference_image_b64')
            if ref_b64:
                payload['image'] = f'data:image/png;base64,{ref_b64}'

        # Apply param aliases (e.g. duration → seconds)
        aliases: dict = self.config.get('param_aliases') or {}
        for std_name, alias_name in aliases.items():
            if std_name in payload:
                payload[alias_name] = payload.pop(std_name)

        return payload

    # ── Response extraction ───────────────────────────────────────

    def _extract_bytes(self, data: Dict[str, Any]) -> bytes:
        """Try known response formats in order."""
        # Custom response path (JMESPath-style, simplified)
        path = self.config.get('response_path')
        if path:
            return self._extract_by_path(data, path)

        logger.debug('Extracting bytes from response keys: %s', list(data.keys()))

        # OpenAI format: {"data": [{"b64_json": "..."}]} or {"data": [{"url": "..."}]}
        if 'data' in data and isinstance(data['data'], list) and len(data['data']) > 0:
            item = data['data'][0]
        elif 'b64_json' in data:
            item = data
        elif 'url' in data:
            item = data
        elif 'content' in data:
            # Direct content URL (e.g. Doubao/Volcengine seedance response)
            content = data['content']
            item = {'url': content} if isinstance(content, str) else content
        elif 'video' in data:
            item = {'url': data['video']} if isinstance(data['video'], str) else data['video']
        elif 'result' in data:
            result = data['result']
            item = result if isinstance(result, dict) else (
                result[0] if isinstance(result, list) and len(result) > 0 else {}
            )
        else:
            raise ValueError(
                f'Unexpected API response format: keys={list(data.keys())}. '
                f'First 200 chars: {str(data)[:200]}'
            )

        if 'b64_json' in item:
            return base64.b64decode(item['b64_json'])
        if 'url' in item:
            resp = requests.get(item['url'], timeout=120)
            resp.raise_for_status()
            return resp.content
        if 'video_url' in item:
            resp = requests.get(item['video_url'], timeout=120)
            resp.raise_for_status()
            return resp.content

        raise ValueError(
            f'Cannot extract media from response: keys={list(item.keys())}'
        )

    def _extract_by_path(self, data: Dict[str, Any], path: str) -> bytes:
        """Simple dot-notation path extraction: 'data.0.b64_json'."""
        parts = path.replace('[', '.').replace(']', '').split('.')
        current: Any = data
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            elif isinstance(current, list):
                try:
                    current = current[int(part)]
                except (ValueError, IndexError):
                    raise ValueError(f'Invalid path segment {part!r} in {path}')
            else:
                raise ValueError(f'Cannot index into {type(current)} at {part!r}')
            if current is None:
                raise ValueError(f'Path {path} resolved to None at {part!r}')
        if isinstance(current, str):
            return base64.b64decode(current)
        return current  # type: ignore

    # ── Async polling ─────────────────────────────────────────────

    def _resolve_poll_url(self, post_url: str, task_id: str) -> str:
        """Use custom async_poll_url template if configured."""
        template = self.config.get('async_poll_url')
        if template:
            return self.api_base.rstrip('/') + '/' + template.replace('{id}', task_id).lstrip('/')
        return super()._resolve_poll_url(post_url, task_id)

    def _fetch_async_content(
        self, data: Dict[str, Any], task_id: str, headers: Dict[str, str]
    ) -> Dict[str, Any]:
        """Download content from custom content URL template."""
        template = self.config.get('async_content_url')
        if template:
            url = self.api_base.rstrip('/') + '/' + template.replace('{id}', task_id).lstrip('/')
            logger.info('Downloading async content from %s', url)
            resp = requests.get(url, headers=headers, timeout=120)
            if resp.status_code == 200 and len(resp.content) > 0:
                logger.info('Downloaded async content: %d bytes', len(resp.content))
                return {'b64_json': base64.b64encode(resp.content).decode('utf-8')}
            logger.warning('Async content download failed: HTTP %d', resp.status_code)
        return super()._fetch_async_content(data, task_id, headers)

    # ── Retry ─────────────────────────────────────────────────────

    def _handle_retry(
        self, response, url: str, headers: dict, payload: dict, **params
    ) -> requests.Response:
        """On 400 response, try falling back size from WxH to resolution string."""
        if response.status_code == 400:
            err_text = response.text.lower()
            if any(kw in err_text for kw in ('size', 'resolution', 'invalid')):
                width = params.get('width', 1024)
                height = params.get('height', 1024)
                alt = _w_h_to_resolution(width, height)
                logger.info('Size %s rejected (400), retrying with %s', payload.get('size'), alt)
                payload['size'] = alt
                return requests.post(url, json=payload, headers=headers, timeout=300)
        return response


# ═══════════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════════

def create_adapter(config: Dict[str, Any]) -> AIGCApiAdapter:
    """Factory: create the appropriate API adapter from model config.

    Args:
        config: Model config dict with keys: provider, api_base, api_key, etc.

    Returns:
        AIGCApiAdapter subclass instance.
    """
    provider = config.get('provider', 'openai') or 'openai'

    if provider in ('openai', 'custom'):
        return OpenAICompatibleAdapter(config)

    raise ValueError(f'Unsupported AIGC API provider: {provider}')


def _w_h_to_resolution(width: int, height: int) -> str:
    """Convert WxH to resolution string for size fallback."""
    if height <= 480:
        return '480p'
    if height <= 720:
        return '720p'
    return '1080p'
