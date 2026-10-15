"""AIGC API adapters — decouple 'how to call this API' from model logic.

Architecture:
    Model class ──uses──▶ AIGCApiAdapter ──HTTP──▶ API provider
                          ├── OpenAICompatibleAdapter (default)
                          └── (future: Volcengine, DashScope, etc.)

Each adapter handles: URL resolution, payload construction, response parsing,
and async polling. Returns raw media bytes; model classes handle PIL/frame conversion.
"""

import base64
import copy
import json
import logging
import re
import requests
import time as _time
from abc import ABC, abstractmethod
from io import BytesIO
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__.replace('evalscope', 'evalperf'))

# 日志里回显供应商响应体前，先抹掉形似凭据的串（sk-xxx / Bearer xxx）
_SECRET_RE = re.compile(r'(?:sk|Bearer)[- _]?[A-Za-z0-9_\-]{6,}', re.I)

# ═══════════════════════════════════════════════════════════════════════
# Abstract base
# ═══════════════════════════════════════════════════════════════════════


class AIGCApiAdapter(ABC):
    """Base for API provider adapters.

    Subclasses implement the 'wire protocol' for a specific provider:
    URL construction, payload format, response extraction, async polling.
    """

    # ── 异步轮询预算（本默认实现服务 OpenAI 兼容来源；两个原生供应商子类各自覆盖）──
    # 视频单条生成常需数分钟（排队 + 推理），旧的 60×5s=300s 上限会把成功任务误判为失败；
    # 这里与 DashScope / Volcengine 同源取 15s × 180 = 45 分钟。
    POLL_INTERVAL_SECONDS = 15
    POLL_TIMEOUT_SECONDS = 2700
    POLL_ERROR_BODY_CHARS = 300  # 非 2xx 轮询回显响应体的截断长度
    # 状态词大小写不敏感（网关有 SUCCEEDED / Success / finished 等写法）
    SUCCESS_STATUSES = ('completed', 'succeeded', 'success', 'finished', 'done')
    FAILURE_STATUSES = ('failed', 'failure', 'error', 'cancelled', 'canceled', 'expired')
    FATAL_POLL_STATUSES = (401, 403)  # 鉴权/权限类不会自愈：立即失败
    POLL_NOT_FOUND_GRACE = 3  # 连续 404 达该次数即判为轮询地址配错，不再空转到超时

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
        if params.get('tool') == 'txt2video':
            logger.info(
                'Video request payload: %s',
                {
                    key: value
                    for key, value in payload.items()
                    if key != 'prompt'
                },
            )

        image_request_started = _time.monotonic() if params.get('tool', 'txt2img') == 'txt2img' else None
        response = requests.post(url, json=payload, headers=headers, timeout=300)
        if image_request_started is not None:
            image_request = getattr(self, 'last_image_request', None)
            if image_request is not None:
                image_request['request_seconds'] = _time.monotonic() - image_request_started

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

    def _handle_retry(self, response, url: str, headers: dict, payload: dict, **params) -> requests.Response:
        """Optional retry hook. Subclass can override to e.g. fall back size format."""
        return response

    def _poll_async(self, data: Dict[str, Any], post_url: str, headers: Dict[str, str]) -> Dict[str, Any]:
        """轮询异步任务直到终态，返回最终响应数据（供 _extract_bytes 取媒体）。

        默认实现服务 OpenAI 兼容来源（DashScope / Volcengine 各自覆盖为原生协议）：

        * 状态词归一：大小写不敏感，并覆盖 success / finished 等同义终态词
        * 非 2xx 轮询回显脱敏后的响应体；401/403 立即失败，连续 404 判为查询地址配错
        * 超时错误带 task_id / 最后状态 / 查询 URL，便于人工复查而不必重建任务
        * 创建响应即终态时同样走 _fetch_async_content（该网关可能只给内容模板地址）
        """
        data = self._unwrap_response(data)
        task_id = data.get('task_id', data.get('id', ''))
        status = self._normalize_status(data.get('status', ''))
        logger.info('Async task %s: status=%s', task_id, status)

        if status in self.SUCCESS_STATUSES:
            return self._fetch_async_content(data, task_id, headers)

        poll_url = self._resolve_poll_url(post_url, task_id)
        max_attempts = max(1, self.POLL_TIMEOUT_SECONDS // self.POLL_INTERVAL_SECONDS)
        last_status = status
        not_found_streak = 0
        logger.info(
            'Polling %s every %ds, up to %d min',
            task_id,
            self.POLL_INTERVAL_SECONDS,
            self.POLL_TIMEOUT_SECONDS // 60,
        )

        for attempt in range(max_attempts):
            _time.sleep(self.POLL_INTERVAL_SECONDS)
            resp = requests.get(poll_url, headers=headers, timeout=30)
            if not resp.ok:
                logger.warning(
                    'Poll %d: HTTP %d body=%s',
                    attempt + 1,
                    resp.status_code,
                    self._redact_secrets(resp.text)[:self.POLL_ERROR_BODY_CHARS],
                )
                # 创建任务已用同一密钥成功，轮询再遇鉴权拒绝不会自愈：立即失败并回显原因。
                if resp.status_code in self.FATAL_POLL_STATUSES:
                    raise RuntimeError(
                        f'OpenAI 兼容异步轮询被拒（不可重试，task_id={task_id}）：'
                        f'HTTP {resp.status_code} {self._poll_error_detail(resp.text)}。'
                        '请确认该 api_key 有效、且已开通对应视频模型的查询权限。'
                    )
                if resp.status_code == 404:
                    not_found_streak += 1
                    if not_found_streak >= self.POLL_NOT_FOUND_GRACE:
                        raise RuntimeError(
                            f'异步任务查询地址连续 {not_found_streak} 次 404'
                            f'（task_id={task_id}）：{poll_url}。该网关的查询地址与创建地址不同形，'
                            '请用配置项 async_poll_url 指定（支持 {id} 占位，'
                            '如 "video/generations/{id}"）。'
                        )
                else:
                    not_found_streak = 0
                continue
            not_found_streak = 0
            data = self._unwrap_response(resp.json())
            status = self._normalize_status(data.get('status', ''))
            last_status = status or last_status
            logger.info('Task %s: status=%s progress=%s%%', task_id, status, data.get('progress', 0))

            if status in self.SUCCESS_STATUSES:
                logger.info('Task %s completed', task_id)
                return self._fetch_async_content(data, task_id, headers)
            if status in self.FAILURE_STATUSES:
                detail = self._task_failure_detail(data)
                logger.error('Task %s failed: %s', task_id, detail)
                raise RuntimeError(f'Task {task_id} failed: {detail}')

        raise TimeoutError(
            f'Task {task_id} did not complete within {self.POLL_TIMEOUT_SECONDS}s '
            f'(last status={last_status}). 任务可能仍在供应商侧生成，可用 '
            f'"GET {poll_url}" 手工复查结果，请勿重复创建任务。'
        )

    @staticmethod
    def _normalize_status(status: Any) -> str:
        """状态词归一：网关大小写混用（SUCCEEDED / Success / In_Progress），统一小写去空白。"""
        return str(status or '').strip().lower()

    @staticmethod
    def _task_failure_detail(data: Dict[str, Any]) -> str:
        """从失败响应里取出可读原因（code + message），脱敏截断后交给用户与日志。"""
        error = data.get('error')
        if isinstance(error, dict):
            code = error.get('code') or ''
            message = error.get('message') or error.get('msg') or ''
        elif error:
            code, message = '', str(error)
        else:
            code, message = '', ''
        if not message:
            message = data.get('message') or data.get('reason') or data.get('fail_reason') or ''
        detail = ' '.join(str(part) for part in (code, message) if part)
        if not detail:
            detail = 'keys=' + ','.join(str(key) for key in data.keys())
        return _SECRET_RE.sub('***', detail)[:300]

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

    @staticmethod
    def _redact_secrets(text: str) -> str:
        """抹掉响应体里形似凭据的串，再写日志。"""
        return _SECRET_RE.sub('***', text or '')

    @staticmethod
    def _poll_error_detail(text: str) -> str:
        """从供应商错误响应体里取出 code/message/request_id，便于用户与工单定位。

        兼容两种形态：百炼把 code/message/request_id 放在顶层；
        火山方舟把 code/message 包在 error 里，request_id 在顶层。
        """
        raw = (text or '').strip()
        try:
            payload = json.loads(text)
        except (ValueError, TypeError):
            return raw[:120]
        if not isinstance(payload, dict):
            return raw[:120]
        error = payload.get('error')
        merged: Dict[str, Any] = dict(error) if isinstance(error, dict) else {}
        merged.update({key: value for key, value in payload.items() if key != 'error'})
        return ' '.join(str(merged[key]) for key in ('code', 'message', 'request_id') if merged.get(key)) or raw[:120]

    def _fetch_async_content(self, data: Dict[str, Any], task_id: str, headers: Dict[str, str]) -> Dict[str, Any]:
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

    def generate(self, prompt: str, **params) -> bytes:
        if params.get('tool', 'txt2img') != 'txt2img':
            return super().generate(prompt, **params)
        self.last_image_request = {}
        result = super().generate(prompt, **params)
        self.last_image_request['parameter_support'] = 'unconfirmed'
        return result

    # 文生视频时长域（秒）：前端同区间，非法值明确报错而非静默替换
    VIDEO_DURATION_RANGE = (5, 15)

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
        }

        if tool == 'txt2img':
            if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in (width, height)):
                raise ValueError('文生图尺寸须为正整数像素')

        if tool == 'txt2video':
            # 网关契约（UniToken / new-api 系：POST {base}/video/generations）：
            # resolution、ratio、duration 必须落在 metadata 里；顶层同名字段和 size
            # 会被网关忽略，缺少 metadata.resolution 时直接 400 "resolution is required"。
            # 字段依据（2026-10 核对官方文档 unitoken.apifox.cn → API文档/视频生成）：
            #   metadata{content 必需, ratio 必需, resolution 必需(480p/720p/1080p/4k，默认720p),
            #            duration 可选(默认5), generate_audio 必需(Seedance 2.0 系列/1.5 pro),
            #            watermark 可选}
            # 文生视频没有参考素材：content 留空数组，网关会自行补 text 段。
            duration = self._validate_video_duration(params.get('duration'))
            resolution = str(params.get('resolution') or _w_h_to_resolution(width, height)).strip().lower()
            ratio = str(params.get('ratio') or _wh_to_aspect_ratio(width, height)).strip()
            payload['metadata'] = {
                'content': [],
                'resolution': resolution,
                'ratio': ratio,
                'duration': duration,
                'generate_audio': bool(params.get('generate_audio', False)),
            }
            seed = params.get('seed')
            if seed is not None:
                # 种子支持度因网关而异（图片侧的 seed 走顶层，视频按契约只能落在 metadata）：
                # 网关认识就生效、不认识多半随 metadata 被忽略；若网关明确以 400 点名 seed，
                # 由 _handle_retry 去掉该字段重发一次 —— 「网关不支持种子」不该拖垮整批视频生成。
                payload['metadata']['seed'] = seed
        else:
            payload['n'] = 1
            payload['size'] = f'{width}x{height}'
            for name in ('num_inference_steps', 'guidance_scale', 'seed'):
                if params.get(name) is not None:
                    payload[name] = params[name]
            if params.get('negative_prompt'):
                payload['negative_prompt'] = params['negative_prompt']

        if tool == 'img2img':
            ref_b64 = params.get('reference_image_b64')
            if ref_b64:
                payload['image'] = f'data:image/png;base64,{ref_b64}'

        # Apply param aliases (e.g. duration → seconds)
        aliases: dict = self.config.get('param_aliases') or {}
        for std_name, alias_name in aliases.items():
            if std_name in payload:
                payload[alias_name] = payload.pop(std_name)

        if tool in ('txt2img', 'img2img'):
            self.last_image_request = {
                'parameters': {
                    key: value
                    for key, value in payload.items()
                    if key != 'prompt' and (tool != 'img2img' or key != aliases.get('image', 'image'))
                },
            }
        return payload

    def _validate_video_duration(self, value: Any) -> int:
        """文生视频时长必须是 5~15 的整数秒（前端同区间）。

        非法值明确报错而非静默替换：静默 clamp 会让「我填了 3 秒」在结果里变成 5 秒，
        评测数据无法解释；超出网关能力范围的值应由用户看到错误后自己改。
        """
        low, high = self.VIDEO_DURATION_RANGE
        raw = 5 if value is None or value == '' else value
        invalid = isinstance(raw, bool) or (isinstance(raw, float) and not raw.is_integer())
        if not invalid:
            try:
                duration = int(raw)
            except (TypeError, ValueError):
                invalid = True
        if invalid:
            raise ValueError(f'文生视频时长须为 {low}~{high} 的整数秒，收到 {value!r}')
        if not low <= duration <= high:  # type: ignore[possibly-undefined]
            raise ValueError(f'文生视频时长须为 {low}~{high} 的整数秒，收到 {value!r}')
        return duration

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
            item = result if isinstance(result,
                                        dict) else (result[0] if isinstance(result, list) and len(result) > 0 else {})
        elif 'result_url' in data:
            # 网关轮询终态：{code,message,data:{status:"SUCCESS", result_url:"..."}}
            item = {'url': data['result_url']}
        else:
            raise ValueError(
                f'Unexpected API response format: keys={list(data.keys())}. '
                f'First 200 chars: {str(data)[:200]}'
            )

        if 'b64_json' in item:
            return base64.b64decode(item['b64_json'])
        if 'url' in item:
            return self._download_media(item['url'])
        if 'video_url' in item:
            return self._download_media(item['video_url'])

        raise ValueError(f'Cannot extract media from response: keys={list(item.keys())}')

    def _download_media(self, url: str) -> bytes:
        """下载成品媒体字节。

        先裸 GET（兼容预签名直链 / CDN）；401/403 表示内容端点要求鉴权
        （部分 OpenAI 兼容网关的 /content 类地址如此），此时带上 Authorization 重试一次，
        失败则把状态码与脱敏响应体写进错误信息，而不是抛一个无上下文的 HTTPError。
        """
        resp = requests.get(url, timeout=120)
        if resp.status_code in (401, 403) and self.api_key:
            logger.info('Media download HTTP %d, retrying with Authorization header', resp.status_code)
            retry = requests.get(url, headers={'Authorization': f'Bearer {self.api_key}'}, timeout=120)
            if retry.ok:
                return retry.content
            logger.warning(
                'Media download failed: HTTP %d body=%s',
                retry.status_code,
                self._redact_secrets(retry.text)[:200],
            )
            resp = retry
        resp.raise_for_status()
        return resp.content

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

    def _is_async_response(self, data: Dict[str, Any]) -> bool:
        """兼容 {code, message, data: {task_id, status}} 包裹形态的网关。"""
        return super()._is_async_response(self._unwrap_response(data))

    def _resolve_poll_url(self, post_url: str, task_id: str) -> str:
        """Use custom async_poll_url template if configured."""
        template = self.config.get('async_poll_url')
        if template:
            return self.api_base.rstrip('/') + '/' + template.replace('{id}', task_id).lstrip('/')
        return super()._resolve_poll_url(post_url, task_id)

    def _fetch_async_content(self, data: Dict[str, Any], task_id: str, headers: Dict[str, str]) -> Dict[str, Any]:
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

    def _handle_retry(self, response, url: str, headers: dict, payload: dict, **params) -> requests.Response:
        """两处窄口兜底，只为不让可选字段拖垮整批评测；其余 400 一律原样上抛。

        - 视频 seed：支持度因网关而异（有的原样透传给上游并生效，有的 400 拒绝未知字段）。
          仅在服务端**自己点名 seed** 时去掉该字段重发一次，实现「不支持则静默失效，不影响出片」。
        - 图片 size：个别网关只认「720p」这类档位而非 WxH，按宽高推导后重发一次。
          图片专用——视频的 size 网关从不解析（旧实现把「顶层 size 被拒」改成「顶层 resolution」
          再发一次，两次请求对网关等价，只会掩盖真实错误，视频 400 必须原样回显）。
        """
        if response.status_code != 400:
            return response
        err_text = response.text.lower()
        metadata = payload.get('metadata') or {}
        if params.get('tool') == 'txt2video' and 'seed' in err_text and metadata.get('seed') is not None:
            retry_payload = copy.deepcopy(payload)
            retry_payload['metadata'].pop('seed')
            logger.warning(
                '网关不接受 seed，去掉该字段重发一次（响应：%s）',
                self._redact_secrets(response.text)[:200],
            )
            return requests.post(url, json=retry_payload, headers=headers, timeout=300)
        if params.get('tool', 'txt2img') == 'txt2img':
            raise ValueError(f'文生图 API HTTP 400: {self._redact_secrets(response.text)[:1000]}')
        if params.get('tool') != 'txt2video' and any(kw in err_text for kw in ('size', 'resolution', 'invalid')):
            width = params.get('width', 1024)
            height = params.get('height', 1024)
            alt = _w_h_to_resolution(width, height)
            logger.info('Size %s rejected (400), retrying with %s', payload.get('size'), alt)
            payload['size'] = alt
            return requests.post(url, json=payload, headers=headers, timeout=300)
        logger.warning('API request failed with HTTP 400: %s', self._redact_secrets(response.text)[:1000])
        return response


# ═══════════════════════════════════════════════════════════════════════
# ── DashScope 视频模型能力表 ────────────────────────────────────────────
# 百炼文生视频按模型分三套参数形态，字段名与取值域各不相同：
#   wan2.7 新协议      → parameters.resolution（"720P"/"1080P"）+ ratio
#   wan2.6~2.1 旧协议  → parameters.size（必须是具体像素值，如 "1280*720"）
#   kling              → parameters.mode（std=720P / pro=1080P）+ aspect_ratio
# 数据来源（官方文档，2026-10 核对）：
#   https://help.aliyun.com/zh/model-studio/text-to-video-api-reference            (wan2.7)
#   https://help.aliyun.com/zh/model-studio/legacy-wan-text-to-video-api-reference (wan2.1~2.6)
#   https://help.aliyun.com/zh/model-studio/kling-video-generation-api-reference   (kling)
# 改值域只需改这张表；前端 evalscope/web/src/components/eval/AIGCEvalForm.tsx 的
# VIDEO_FAMILIES 与之一一对应（tests/benchmark/test_dashscope_adapter.py 有对齐断言）。

# 旧协议档位 →「宽高比 → 像素值」穷举表（官方文档原文，未列出的组合一律报错，不静默替换）
_DASHSCOPE_LEGACY_SIZE_TABLE = {
    '480p': {
        '16:9': '832*480',
        '9:16': '480*832',
        '1:1': '624*624',
    },
    '720p': {
        '16:9': '1280*720',
        '9:16': '720*1280',
        '1:1': '960*960',
        '4:3': '1088*832',
        '3:4': '832*1088',
    },
    '1080p': {
        '16:9': '1920*1080',
        '9:16': '1080*1920',
        '1:1': '1440*1440',
        '4:3': '1632*1248',
        '3:4': '1248*1632',
    },
}

# wan2.7 用 resolution + ratio 表达，不传 size；此处仅用于校验档位/比例是否配套
_DASHSCOPE_WAN27_RATIOS = {
    '720p': ('16:9', '9:16', '1:1', '4:3', '3:4'),
    '1080p': ('16:9', '9:16', '1:1', '4:3', '3:4'),
}

# kling 文生视频只支持 3 种宽高比；分辨率用 mode 表达，不支持 seed
_DASHSCOPE_KLING_RATIOS = ('16:9', '9:16', '1:1')

_DASHSCOPE_VIDEO_FAMILIES = (
    {
        'name': 'wan2.7',
        'prefixes': ('wan2.7', ),
        'protocol': 'resolution',
        'tiers': ('720p', '1080p'),
        'duration': ('range', 2, 15),
        'seed': True,
        'default_tier': '1080p'
    },
    {
        'name': 'wan2.6',
        'prefixes': ('wan2.6', ),
        'protocol': 'size',
        'tiers': ('720p', '1080p'),
        'duration': ('range', 2, 15),
        'seed': True,
        'default_tier': '1080p'
    },
    {
        'name': 'wan2.5',
        'prefixes': ('wan2.5', ),
        'protocol': 'size',
        'tiers': ('480p', '720p', '1080p'),
        'duration': ('choice', (5, 10)),
        'seed': True,
        'default_tier': '1080p'
    },
    {
        'name': 'wan2.2',
        'prefixes': ('wan2.2', ),
        'protocol': 'size',
        'tiers': ('480p', '1080p'),
        'duration': ('fixed', 5),
        'seed': True,
        'default_tier': '1080p'
    },
    {
        'name': 'wan2.1-t2v-turbo',
        'prefixes': ('wan2.1-t2v-turbo', 'wanx2.1-t2v-turbo'),
        'protocol': 'size',
        'tiers': ('480p', '720p'),
        'duration': ('fixed', 5),
        'seed': True,
        'default_tier': '720p'
    },
    {
        'name': 'wan2.1-t2v-plus',
        'prefixes': ('wan2.1-t2v-plus', 'wanx2.1-t2v-plus'),
        'protocol': 'size',
        'tiers': ('720p', ),
        'duration': ('fixed', 5),
        'seed': True,
        'default_tier': '720p'
    },
    {
        'name': 'kling',
        'prefixes': ('kling', ),
        'protocol': 'mode',
        'tiers': ('720p', '1080p'),
        'duration': ('range', 3, 15),
        'seed': False,
        'default_tier': '1080p'
    },
)

_KLING_MODE_BY_TIER = {'720p': 'std', '1080p': 'pro'}
_KLING_TIER_BY_MODE = {value: key for key, value in _KLING_MODE_BY_TIER.items()}


def _dashscope_video_family(model: str) -> Dict[str, Any]:
    """按模型 id 前缀匹配百炼视频模型族；无法识别时报错而不是猜参数形态。"""
    name = (model or '').strip().lower()
    for family in _DASHSCOPE_VIDEO_FAMILIES:
        if any(name.startswith(prefix) for prefix in family['prefixes']):
            return family
    raise ValueError(
        f'无法识别 DashScope 视频模型 {model!r}：'
        '目前支持 wan2.7 / wan2.6 / wan2.5 / wan2.2 / wan2.1-t2v-* / kling 系列'
    )


def _dashscope_ratios(family: Dict[str, Any], tier: str) -> tuple:
    """返回该模型该档位下可用的宽高比（旧协议由 size 表穷举得出）。"""
    if family['protocol'] == 'mode':
        return _DASHSCOPE_KLING_RATIOS
    if family['protocol'] == 'resolution':
        return _DASHSCOPE_WAN27_RATIOS[tier]
    return tuple(_DASHSCOPE_LEGACY_SIZE_TABLE[tier].keys())


def _dashscope_duration(family: Dict[str, Any], model: str, duration: int) -> int:
    """按模型族校验时长，非法值直接报错（不再静默 snap 成 5/10）。"""
    kind, expected = family['duration'][0], family['duration'][1:]
    if kind == 'range':
        low, high = expected
        if not low <= duration <= high:
            raise ValueError(f'{model} 时长需为 {low}～{high} 秒整数，收到 {duration}')
        return duration
    if kind == 'choice':
        values = tuple(expected[0])
        if duration not in values:
            joined = ' 或 '.join(str(value) for value in values)
            raise ValueError(f'{model} 只支持时长 {joined} 秒，收到 {duration}')
        return duration
    value = int(expected[0])
    if duration != value:
        raise ValueError(f'{model} 时长固定为 {value} 秒，收到 {duration}')
    return value


# DashScope video generation adapter (Aliyun 百炼, kling / wanx 等视频模型)
# ═══════════════════════════════════════════════════════════════════════


class DashScopeVideoAdapter(AIGCApiAdapter):
    """Aliyun DashScope video generation API (video-synthesis).

    Wire protocol（三套协议共用同一端点与异步轮询，仅 parameters 形态不同）:
      wan2.7      https://help.aliyun.com/zh/model-studio/text-to-video-api-reference
      wan2.1~2.6  https://help.aliyun.com/zh/model-studio/legacy-wan-text-to-video-api-reference
      kling       https://help.aliyun.com/zh/model-studio/kling-video-generation-api-reference
      POST {api_root}/services/aigc/video-generation/video-synthesis
        headers: X-DashScope-Async: enable + Authorization: Bearer <key>
        body (wan2.7): {"model": ..., "input": {"prompt": ..., "negative_prompt": ...},
                        "parameters": {"resolution": "1080P", "ratio": "16:9",
                                       "duration": 5, "seed": 42}}
        body (wan2.6~2.1): "parameters": {"size": "1280*720", "duration": 5, "seed": 42}
        body (kling): "parameters": {"mode": "pro", "aspect_ratio": "16:9",
                                     "duration": 5, "watermark": False}
        resp: {"output": {"task_id": ..., "task_status": "PENDING"}, "request_id": ...}
      GET  {api_root}/tasks/{task_id}
        resp: {"output": {"task_id": ..., "task_status": "SUCCEEDED", "video_url": ...}}

    Differs from OpenAI-compatible APIs in 5 ways (each handled below):
      1. nested payload (input.prompt / parameters.*) instead of flat prompt/size
      2. async detection via output.task_id + output.task_status (not top-level)
      3. response wrapped in `output` (not `data`)
      4. requires X-DashScope-Async: enable header
      5. aspect_ratio + duration semantics instead of size/num_frames
    """

    SYNTHESIS_PATH = 'services/aigc/video-generation/video-synthesis'
    SUCCESS_STATUSES = ('SUCCEEDED', )
    FAILURE_STATUSES = ('FAILED', 'CANCELED', 'UNKNOWN')

    # 异步轮询预算。官方口径：文生视频通常耗时 1~5 分钟，建议查询间隔 15s，
    # 查询接口默认 RPS 20；task_id 有效期 24 小时，超时后可用同一 task_id 复查，
    # 切勿重复创建任务（https://help.aliyun.com/zh/model-studio/text-to-video-api-reference）。
    POLL_INTERVAL_SECONDS = 15
    # 20 分钟会对真实负载误杀：实测 wan2.7-t2v 720P / 2 秒片段耗时 739.5s（约 12.3 分钟，
    # 且并发争抢会进一步拉长），官方口径"通常 1~5 分钟"在本环境不成立。取 45 分钟上限
    # （15s × 180），既留足余量又不必等官方允许的 3 小时。嫌久可直接调小这个常量。
    POLL_TIMEOUT_SECONDS = 2700
    POLL_ERROR_BODY_CHARS = 300  # 非 2xx 轮询回显响应体的截断长度
    # 鉴权/权限类错误（401/403 + 以下语义）重试无意义：立即失败，别空转到超时。
    # 实测「current user api does not support asynchronous calls」= 该 API-KEY 无异步调用权限。
    FATAL_POLL_MARKERS = ('AccessDenied', 'InvalidApiKey', 'does not support asynchronous')

    # ── URL resolution ────────────────────────────────────────────

    def _api_root(self) -> str:
        """Return the API root (host + /api/v1).

        Tolerates both ``https://dashscope.aliyuncs.com/api/v1`` and the full
        synthesis endpoint being pasted into api_base.
        """
        base = self.api_base.rstrip('/')
        marker = '/' + self.SYNTHESIS_PATH
        if base.endswith(marker):
            return base[:-len(marker)]
        return base

    def _resolve_url(self, tool: str) -> str:
        return self._api_root() + '/' + self.SYNTHESIS_PATH

    # ── Headers ───────────────────────────────────────────────────

    def _build_headers(self) -> Dict[str, str]:
        headers = super()._build_headers()
        headers['X-DashScope-Async'] = 'enable'
        return headers

    # ── Payload construction ──────────────────────────────────────

    def _build_payload(self, prompt: str, **params) -> Dict[str, Any]:
        """按百炼三套协议分别构造 payload（见文件头部能力表）。

        分辨率一律由 resolution 档位表达（前端传统一小写档位 480p/720p/1080p，
        kling 额外接受官方 mode 取值 std/pro）；宽高比由 ratio 表达，不再从
        width/height 反推像素尺寸。所有越界/不配套取值直接抛 ValueError，
        由任务日志回显给用户，绝不静默替换（旧实现对 duration 做 5/10 snap 已移除）。
        """
        tool = params.get('tool', 'txt2video')
        if tool != 'txt2video':
            raise ValueError(f'DashScope adapter only supports txt2video, got tool={tool!r}')

        model = self.model_name
        family = _dashscope_video_family(model)

        # ── 分辨率档位 ────────────────────────────────────────────
        tier_raw = str(params.get('resolution') or '').strip().lower()
        if not tier_raw:
            tier = family['default_tier']
        elif family['protocol'] == 'mode':
            # kling 官方以 mode 表达分辨率，兼容调用方直接传 std/pro
            tier = _KLING_TIER_BY_MODE.get(tier_raw, tier_raw)
        else:
            tier = tier_raw
        if tier not in family['tiers']:
            tiers_text = '/'.join(family['tiers'])
            raise ValueError(f'{model} 不支持分辨率 {tier_raw or tier}（可选：{tiers_text}）')

        # ── 宽高比 ────────────────────────────────────────────────
        ratios = _dashscope_ratios(family, tier)
        ratio = str(params.get('ratio') or '').strip() or '16:9'  # 官方默认值
        if ratio not in ratios:
            ratios_text = '/'.join(ratios)
            raise ValueError(f'{model} 在 {tier} 档位下不支持画幅比 {ratio}（可选：{ratios_text}）')

        # ── 时长 ──────────────────────────────────────────────────
        raw_duration = params.get('duration', 5)
        try:
            duration = int(raw_duration if raw_duration not in (None, '') else 5)
        except (TypeError, ValueError):
            raise ValueError(f'时长需为整数秒，收到 {raw_duration!r}')
        duration = _dashscope_duration(family, model, duration)

        # ── 可选参数 ──────────────────────────────────────────────
        input_body: Dict[str, Any] = {'prompt': prompt}
        negative_prompt = str(params.get('negative_prompt') or '').strip()
        if negative_prompt:
            # 反向提示词：wan 全系与 kling-v3 / kling-v3-omni 支持；kling-v3-turbo 不支持
            input_body['negative_prompt'] = negative_prompt

        seed = params.get('seed')
        seed_value = None
        if seed not in (None, ''):
            try:
                seed_value = int(seed)
            except (TypeError, ValueError):
                raise ValueError(f'随机种子需为整数，收到 {seed!r}')
            if not 0 <= seed_value <= 2147483647:
                raise ValueError(f'随机种子需在 0～2147483647 之间，收到 {seed_value}')
            if not family['seed']:
                logger.warning('%s 不支持 seed 参数，已忽略 seed=%s', model, seed_value)
                seed_value = None

        # ── 组装 parameters ──────────────────────────────────────
        parameters: Dict[str, Any] = {}
        if family['protocol'] == 'resolution':  # wan2.7：resolution + ratio
            parameters['resolution'] = tier.upper()
            parameters['ratio'] = ratio
            parameters['duration'] = duration
        elif family['protocol'] == 'size':  # wan2.6~2.1：必须给具体像素值
            parameters['size'] = _DASHSCOPE_LEGACY_SIZE_TABLE[tier][ratio]
            parameters['duration'] = duration
        else:  # kling：mode 表达分辨率
            parameters['mode'] = _KLING_MODE_BY_TIER[tier]
            parameters['aspect_ratio'] = ratio
            parameters['duration'] = duration
            parameters['watermark'] = False  # 官方默认值，评测输出不需要水印

        if seed_value is not None:
            parameters['seed'] = seed_value

        return {
            'model': model,
            'input': input_body,
            'parameters': parameters,
        }

    # ── Async detection / polling ─────────────────────────────────

    def _is_async_response(self, data: Dict[str, Any]) -> bool:
        out = data.get('output') or {}
        return bool(out.get('task_id')) and bool(out.get('task_status'))

    def _resolve_poll_url(self, post_url: str, task_id: str) -> str:
        # DashScope polls at {api_root}/tasks/{id}, unrelated to the create path
        return self._api_root() + '/tasks/' + task_id

    def _poll_async(self, data: Dict[str, Any], post_url: str, headers: Dict[str, str]) -> Dict[str, Any]:
        out = data.get('output') or {}
        task_id = out.get('task_id', '')
        status = out.get('task_status', '')
        logger.info('DashScope task %s: status=%s', task_id, status)

        if status in self.SUCCESS_STATUSES:
            return data

        poll_url = self._resolve_poll_url(post_url, task_id)
        max_attempts = max(1, self.POLL_TIMEOUT_SECONDS // self.POLL_INTERVAL_SECONDS)
        last_status = status
        # 查询任务是同步 GET：不能带 X-DashScope-Async，否则被网关拒绝
        # （403 AccessDenied「current user api does not support asynchronous calls」；
        #  官方 error-code 文档要求移除该头或设为 disable）。
        poll_headers = {k: v for k, v in headers.items() if k.lower() != 'x-dashscope-async'}
        logger.info(
            'Polling %s every %ds, up to %d min',
            task_id,
            self.POLL_INTERVAL_SECONDS,
            self.POLL_TIMEOUT_SECONDS // 60,
        )

        for attempt in range(max_attempts):
            _time.sleep(self.POLL_INTERVAL_SECONDS)
            resp = requests.get(poll_url, headers=poll_headers, timeout=30)
            if not resp.ok:
                logger.warning(
                    'Poll %d: HTTP %d body=%s',
                    attempt + 1,
                    resp.status_code,
                    self._redact_secrets(resp.text)[:self.POLL_ERROR_BODY_CHARS],
                )
                # 鉴权/权限类错误重试无用：直接失败并把 message/request_id 交给用户
                if resp.status_code in (401, 403) and any(m in resp.text for m in self.FATAL_POLL_MARKERS):
                    raise RuntimeError(
                        f'DashScope 异步轮询被拒（不可重试，task_id={task_id}）：'
                        f'HTTP {resp.status_code} {self._poll_error_detail(resp.text)}。'
                        '请为该 API-KEY 开通异步调用权限（百炼控制台 → API-KEY / 业务空间）后重试。'
                    )
                continue
            data = resp.json()
            out = data.get('output') or {}
            status = out.get('task_status', '')
            last_status = status or last_status
            logger.info('DashScope task %s: status=%s', task_id, status)

            if status in self.SUCCESS_STATUSES:
                logger.info('DashScope task %s completed', task_id)
                return data
            if status in self.FAILURE_STATUSES:
                out = data.get('output') or {}
                fail_msg = out.get('message') or out.get('reason') or data.get('message') or ''
                logger.error('DashScope task %s failed: %s', task_id, fail_msg or data)
                raise RuntimeError(f'DashScope task {task_id} failed: {fail_msg or data}')

        raise TimeoutError(
            f'DashScope task {task_id} did not complete within {self.POLL_TIMEOUT_SECONDS}s '
            f'(last status={last_status}). task_id 有效 24 小时，可用 "GET {poll_url}" '
            f'手工复查结果，请勿重复创建任务。'
        )

    # ── Response extraction ───────────────────────────────────────

    def _extract_bytes(self, data: Dict[str, Any]) -> bytes:
        out = data.get('output') or {}
        video_url = out.get('video_url') or out.get('url')
        if not video_url:
            raise ValueError(
                f'Cannot extract video URL from DashScope response: keys={list(data.keys())}. '
                f'First 200 chars: {str(data)[:200]}'
            )
        resp = requests.get(video_url, timeout=120)
        resp.raise_for_status()
        return resp.content

    def _handle_retry(self, response, url: str, headers: dict, payload: dict, **params) -> requests.Response:
        # DashScope uses aspect_ratio, not size — the OpenAI size-format retry is irrelevant.
        return response


# ═══════════════════════════════════════════════════════════════════════
# Volcengine video generation adapter (火山方舟 Seedance)
# ═══════════════════════════════════════════════════════════════════════

# 火山方舟 Seedance 能力表：分辨率 / 时长 / 画幅比**都按模型前缀分档**，严格照官方文档
# （模型列表 + 创建视频生成任务），前缀顺序敏感 —— 更具体的前缀必须排在前面。
#   https://docs.volcengine.com/docs/ark/model-list
#   https://docs.volcengine.com/docs/ark/create-video-generation-task-api
# 官方 ratio 可选值（文生视频）：16:9、4:3、1:1、3:4、9:16、21:9、adaptive；
# 其中 adaptive 只有 2.5 / 2.0 系列支持（1.0 系列文生视频不支持），默认 16:9。
_VOLCENGINE_RATIOS = ('16:9', '4:3', '1:1', '3:4', '9:16', '21:9')

_VOLCENGINE_VIDEO_FAMILIES = (
    # Seedance 2.5：480p/720p/1080p（无 4k）；时长 4~30，或 -1；支持 adaptive
    {
        'name': 'doubao-seedance-2-5',
        'prefixes': ('doubao-seedance-2-5', ),
        'tiers': ('480p', '720p', '1080p'),
        'duration': (4, 30),
        'smart_duration': True,
        'ratios': _VOLCENGINE_RATIOS,
        'adaptive': True
    },
    # Seedance 2.0 fast / mini：仅 480p、720p（无 1080p/4k）；时长 4~15；支持 adaptive
    {
        'name': 'doubao-seedance-2-0-fast',
        'prefixes': ('doubao-seedance-2-0-fast', ),
        'tiers': ('480p', '720p'),
        'duration': (4, 15),
        'smart_duration': True,
        'ratios': _VOLCENGINE_RATIOS,
        'adaptive': True
    },
    {
        'name': 'doubao-seedance-2-0-mini',
        'prefixes': ('doubao-seedance-2-0-mini', ),
        'tiers': ('480p', '720p'),
        'duration': (4, 15),
        'smart_duration': True,
        'ratios': _VOLCENGINE_RATIOS,
        'adaptive': True
    },
    # Seedance 2.0：480p/720p/1080p/4k；时长 4~15；支持 adaptive
    {
        'name': 'doubao-seedance-2-0',
        'prefixes': ('doubao-seedance-2-0', ),
        'tiers': ('480p', '720p', '1080p', '4k'),
        'duration': (4, 15),
        'smart_duration': True,
        'ratios': _VOLCENGINE_RATIOS,
        'adaptive': True
    },
    # Seedance 1.0 pro / pro fast（官方标注即将下线）：480p/720p/1080p；时长 2~12；
    # 文生视频默认 16:9，不支持 adaptive，也不支持 -1
    {
        'name': 'doubao-seedance-1-0',
        'prefixes': ('doubao-seedance-1-0', ),
        'tiers': ('480p', '720p', '1080p'),
        'duration': (2, 12),
        'smart_duration': False,
        'ratios': _VOLCENGINE_RATIOS,
        'adaptive': False
    },
)

VOLCENGINE_DEFAULT_TIER = '720p'  # 官方：各模型默认分辨率 720p
VOLCENGINE_DEFAULT_RATIO = '16:9'  # 官方：文生视频默认 16:9


def _volcengine_video_family(model: str) -> Dict[str, Any]:
    """按模型 id 前缀匹配火山视频模型族；无法识别时报错而不是猜参数形态。"""
    name = (model or '').strip().lower()
    for family in _VOLCENGINE_VIDEO_FAMILIES:
        if any(name.startswith(prefix) for prefix in family['prefixes']):
            return family
    raise ValueError(
        f'无法识别火山引擎视频模型 {model!r}：'
        '目前支持 doubao-seedance-2-5 / doubao-seedance-2-0 系列 / doubao-seedance-1-0 系列'
    )


class VolcengineVideoAdapter(AIGCApiAdapter):
    """Native Volcengine Ark video-task API for Seedance text-to-video."""

    TASK_PATH = 'contents/generations/tasks'
    SUCCESS_STATUSES = ('succeeded', 'success', 'completed', 'done')
    FAILURE_STATUSES = ('failed', 'error', 'cancelled', 'canceled')

    # 异步轮询预算。官方：任务创建后需轮询查询结果，个人账号非 4k 并发仅 3
    # （企业 10），1080p / 长时长排队会显著拉长总耗时 —— 旧的 60×5s=300s 硬上限
    # 会把排队中的成功任务误判为失败。取与百炼同源的 15s × 180 = 45 分钟。
    POLL_INTERVAL_SECONDS = 15
    POLL_TIMEOUT_SECONDS = 2700
    POLL_ERROR_BODY_CHARS = 300  # 非 2xx 轮询回显响应体的截断长度

    def _api_root(self) -> str:
        base = self.api_base.rstrip('/')
        marker = '/' + self.TASK_PATH
        return base[:-len(marker)] if base.endswith(marker) else base

    def _resolve_url(self, tool: str) -> str:
        if tool != 'txt2video':
            raise ValueError(f'Volcengine adapter only supports txt2video, got tool={tool!r}')
        return self._api_root() + '/' + self.TASK_PATH

    def _build_payload(self, prompt: str, **params) -> Dict[str, Any]:
        if params.get('tool', 'txt2video') != 'txt2video':
            raise ValueError('Volcengine adapter only supports txt2video')
        family = _volcengine_video_family(self.model_name)
        width = int(params.get('width', 1280))
        height = int(params.get('height', 720))
        resolution = str(params.get('resolution') or _w_h_to_resolution(width, height)).strip().lower()
        if resolution not in family['tiers']:
            message = f"{self.model_name} 分辨率需为 {'、'.join(family['tiers'])}，收到 {resolution}"
            logger.error(message)
            raise ValueError(message)
        duration = int(params.get('duration', 5))
        low, high = family['duration']
        if duration == -1 and family['smart_duration']:
            pass  # 官方允许 -1：由模型在有效范围内智能选择时长（仅 2.0 / 2.5 系列）
        elif not low <= duration <= high:
            message = f'{self.model_name} 视频时长需为 {low}～{high} 秒整数，收到 {duration}'
            logger.error(message)
            raise ValueError(message)
        ratio = str(params.get('ratio') or '').strip() or VOLCENGINE_DEFAULT_RATIO
        if ratio == 'adaptive':
            # 官方：adaptive（自动适配宽高比）仅 2.5 / 2.0 系列支持；1.0 系列文生视频不支持
            if not family['adaptive']:
                message = (f"{self.model_name} 文生视频不支持画幅比 adaptive"
                           f"（可选：{'、'.join(family['ratios'])}）")
                logger.error(message)
                raise ValueError(message)
        elif ratio not in family['ratios']:
            message = (
                f"{self.model_name} 不支持画幅比 {ratio}"
                f"（可选：{'、'.join(family['ratios'])}{'、adaptive' if family['adaptive'] else ''}）"
            )
            logger.error(message)
            raise ValueError(message)
        return {
            'model': self.model_name,
            'content': [{
                'type': 'text',
                'text': prompt
            }],
            'resolution': resolution,
            'ratio': ratio,
            'duration': duration,
            'seed': int(params.get('seed', -1)),
        }

    def _is_async_response(self, data: Dict[str, Any]) -> bool:
        return bool(data.get('id'))

    def _resolve_poll_url(self, post_url: str, task_id: str) -> str:
        return self._api_root() + '/' + self.TASK_PATH + '/' + task_id

    def _poll_async(self, data: Dict[str, Any], post_url: str, headers: Dict[str, str]) -> Dict[str, Any]:
        task_id = data.get('id', '')
        status = str(data.get('status', '')).lower()
        logger.info('Volcengine task %s: status=%s', task_id, status)
        if status in self.SUCCESS_STATUSES:
            return data

        poll_url = self._resolve_poll_url(post_url, task_id)
        max_attempts = max(1, self.POLL_TIMEOUT_SECONDS // self.POLL_INTERVAL_SECONDS)
        last_status = status
        logger.info(
            'Polling %s every %ds, up to %d min',
            task_id,
            self.POLL_INTERVAL_SECONDS,
            self.POLL_TIMEOUT_SECONDS // 60,
        )

        for attempt in range(max_attempts):
            _time.sleep(self.POLL_INTERVAL_SECONDS)
            resp = requests.get(poll_url, headers=headers, timeout=30)
            if not resp.ok:
                logger.warning(
                    'Volcengine poll %d: HTTP %d body=%s',
                    attempt + 1,
                    resp.status_code,
                    self._redact_secrets(resp.text)[:self.POLL_ERROR_BODY_CHARS],
                )
                # 创建任务已用同一 API-KEY 成功，轮询再遇 401/403 属鉴权/权限问题，
                # 不会自愈：立即失败并把 message/request_id 交给用户，别空转到超时。
                if resp.status_code in (401, 403):
                    raise RuntimeError(
                        f'Volcengine 异步轮询被拒（不可重试，task_id={task_id}）：'
                        f'HTTP {resp.status_code} {self._poll_error_detail(resp.text)}。'
                        '请确认该 API-KEY 有效、且已开通对应视频模型的任务查询权限后重试。'
                    )
                continue
            result = resp.json()
            status = str(result.get('status', '')).lower()
            last_status = status or last_status
            logger.info('Volcengine task %s: status=%s', task_id, status)

            if status in self.SUCCESS_STATUSES:
                logger.info('Volcengine task %s completed', task_id)
                return result
            if status in self.FAILURE_STATUSES:
                error = result.get('error')
                if error is None:
                    detail = json.dumps(result, ensure_ascii=False)
                elif isinstance(error, str):
                    detail = error
                else:
                    detail = json.dumps(error, ensure_ascii=False)
                message = f'Volcengine task {task_id} failed: {str(detail)[:self.POLL_ERROR_BODY_CHARS]}'
                logger.error(message)
                raise RuntimeError(message)

        raise TimeoutError(
            f'Volcengine task {task_id} did not complete within {self.POLL_TIMEOUT_SECONDS}s '
            f'(last status={last_status}). 可用 "GET {poll_url}" 手工复查结果，'
            f'请勿重复创建任务。'
        )

    def _extract_bytes(self, data: Dict[str, Any]) -> bytes:
        content = data.get('content') or {}
        video_url = content.get('video_url') if isinstance(content, dict) else None
        if not video_url:
            raise ValueError(f'Cannot extract video URL from Volcengine response: {data}')
        resp = requests.get(video_url, timeout=120)
        resp.raise_for_status()
        return resp.content

    def _handle_retry(self, response, url: str, headers: dict, payload: dict, **params) -> requests.Response:
        return response


# ═══════════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════════


class DashScopeImageAdapter(AIGCApiAdapter):
    """百炼文生图：千问 2.0/万相 2.6 同步，早期万相异步。"""

    LEGACY_MODELS = {
        'wan2.5-t2i-preview',
        'wan2.2-t2i-flash',
        'wan2.2-t2i-plus',
        'wanx2.1-t2i-turbo',
        'wanx2.1-t2i-plus',
        'wanx2.0-t2i-turbo',
    }
    QWEN_MODELS = {
        'qwen-image-2.0',
        'qwen-image-2.0-2026-03-03',
        'qwen-image-2.0-pro',
        'qwen-image-2.0-pro-2026-03-03',
        'qwen-image-2.0-pro-2026-04-22',
        'qwen-image-2.0-pro-2026-06-22',
    }

    def __init__(self, config):
        super().__init__(config)
        if self.model_name not in self.LEGACY_MODELS | self.QWEN_MODELS | {'wan2.6-t2i'}:
            raise ValueError('未识别的百炼文生图模型；支持千问 2.0、万相 2.6/2.5/2.2/2.1/2.0 文生图模型')
        self.legacy = self.model_name in self.LEGACY_MODELS
        self.last_image_request: Dict[str, Any] = {}

    def _resolve_url(self, tool):
        if tool != 'txt2img':
            raise ValueError('百炼图片适配器仅支持文生图')
        base = self.api_base.rstrip('/')
        if 'compatible-mode' in base:
            raise ValueError('百炼文生图不能使用 compatible-mode，请填写域名/api/v1')
        if not base.endswith('/api/v1'):
            raise ValueError('百炼文生图 API 地址须以 /api/v1 结尾')
        endpoint = ('text2image/image-synthesis' if self.legacy else 'multimodal-generation/generation')
        return f'{base}/services/aigc/{endpoint}'

    def _build_headers(self):
        headers = super()._build_headers()
        if self.legacy:
            headers['X-DashScope-Async'] = 'enable'
        return headers

    def _build_payload(self, prompt, **params):
        for name in ('num_inference_steps', 'guidance_scale'):
            if params.get(name) is not None:
                raise ValueError(f'百炼文生图不支持 {name}，请留空')
        width, height = params.get('width', 1024), params.get('height', 1024)
        if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in (width, height)):
            raise ValueError('文生图尺寸须为正整数像素')
        if self.model_name in ('wan2.6-t2i', 'wan2.5-t2i-preview'):
            if not (1280 * 1280 <= width * height <= 1440 * 1440 and 0.25 <= width / height <= 4):
                raise ValueError('万相 2.5/2.6 总像素须在 1280×1280～1440×1440，宽高比须在 1:4～4:1')
        elif self.legacy:
            if not all(512 <= v <= 1440 for v in (width, height)):
                raise ValueError('万相早期模型宽高须在 512～1440')
        elif not 512 * 512 <= width * height <= 2048 * 2048:
            raise ValueError('千问 2.0 总像素须在 512×512～2048×2048')
        parameters = {'size': f'{width}*{height}', 'n': 1}
        seed = params.get('seed')
        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2147483647:
                raise ValueError('百炼图片种子须为 0～2147483647 的整数')
            parameters['seed'] = seed
        input_data = ({
            'prompt': prompt
        } if self.legacy else {
            'messages': [{
                'role': 'user',
                'content': [{
                    'text': prompt
                }]
            }],
        })
        if params.get('negative_prompt'):
            (input_data if self.legacy else parameters)['negative_prompt'] = params['negative_prompt']
        self.last_image_request = {'parameters': {'model': self.model_name, **parameters}}
        return {'model': self.model_name, 'input': input_data, 'parameters': parameters}

    def _handle_retry(self, response, url, headers, payload, **params):
        raise ValueError(f'百炼文生图 HTTP {response.status_code}: {self._redact_secrets(response.text)[:300]}')

    def _is_async_response(self, data):
        return bool((data.get('output') or {}).get('task_id'))

    def _poll_async(self, data, post_url, headers):
        task_id = data['output']['task_id']
        poll_url = f'{self.api_base.rstrip("/")}/tasks/{task_id}'
        poll_headers = {k: v for k, v in headers.items() if k.lower() != 'x-dashscope-async'}
        deadline = _time.monotonic() + 2700
        while True:
            output = data.get('output') or {}
            status = output.get('task_status')
            if status == 'SUCCEEDED':
                return data
            if status in ('FAILED', 'CANCELED', 'UNKNOWN') or data.get('code'):
                raise RuntimeError(f'百炼图片任务 {task_id} 失败: {self._redact_secrets(str(data))[:300]}')
            if _time.monotonic() >= deadline:
                raise TimeoutError(f'百炼图片任务 {task_id} 超时；请查询 {poll_url}，勿重复创建任务')
            _time.sleep(10)
            response = requests.get(poll_url, headers=poll_headers, timeout=30)
            if not response.ok:
                if response.status_code in (401, 403) or response.status_code < 429:
                    raise RuntimeError(
                        f'图片查询 HTTP {response.status_code}: '
                        f'{self._redact_secrets(response.text)[:300]}'
                    )
                continue
            data = response.json()

    def _extract_bytes(self, data):
        if data.get('code'):
            raise ValueError(f'百炼图片错误: {self._redact_secrets(str(data))[:300]}')
        output = data.get('output') or {}
        urls = [item.get('url') for item in output.get('results', [])]
        for choice in output.get('choices', []):
            urls.extend(item.get('image') for item in choice.get('message', {}).get('content', []))
        url = next((value for value in urls if value), None)
        if not url:
            raise ValueError('百炼图片响应缺少图片 URL')
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        return response.content


class DashScopeImageEditAdapter(DashScopeImageAdapter):
    """Qwen 2.0 synchronous single-reference editing protocol."""

    def _resolve_url(self, tool):
        if tool != 'img2img':
            raise ValueError('图像编辑适配器仅支持图生图')
        base = self.api_base.rstrip('/')
        if not base.endswith('/api/v1') or 'compatible-mode' in base:
            raise ValueError('百炼图生图 API 地址须以 /api/v1 结尾')
        return f'{base}/services/aigc/multimodal-generation/generation'

    def __init__(self, config):
        AIGCApiAdapter.__init__(self, config)
        if not re.fullmatch(
            r'qwen-image-2\.0(?:-pro)?(?:-\d{4}-\d{2}-\d{2})?', self.model_name
        ):
            raise ValueError('图生图百炼首期仅支持 qwen-image-2.0 / qwen-image-2.0-pro')
        if 'compatible-mode' in self.api_base:
            raise ValueError('百炼图生图请使用 /api/v1 原生入口')
        self.legacy = False
        self.last_image_request = {}

    def _build_payload(self, prompt, **params):
        if not prompt.strip():
            raise ValueError('图生图编辑指令不能为空')
        reference = params.get('reference_image_b64')
        if not reference:
            raise ValueError('图生图需要有效参考图')
        raw = base64.b64decode(reference, validate=True)
        if len(raw) > 10 * 1024 * 1024:
            raise ValueError('参考图不得超过 10 MB')
        width, height = params.get('width', 1024), params.get('height', 1024)
        if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in (width, height)):
            raise ValueError('图生图输出尺寸须为正整数')
        if not 512 * 512 <= width * height <= 2048 * 2048:
            raise ValueError('千问图像编辑总像素须在 512×512～2048×2048')
        payload = super()._build_payload(prompt, **params)
        payload['input']['messages'][0]['content'].insert(0, {'image': f'data:image/png;base64,{reference}'})
        for name in ('prompt_extend', 'watermark'):
            if name in params and params[name] is not None:
                if not isinstance(params[name], bool):
                    raise ValueError(f'{name} 须为布尔值')
                payload['parameters'][name] = params[name]
        self.last_image_request = {'parameters': {'model': self.model_name, **payload['parameters']}}
        return payload

    def _is_async_response(self, data):
        return False

    def _handle_retry(self, response, url, headers, payload, **params):
        raise ValueError(f'百炼图生图 HTTP {response.status_code}: {self._redact_secrets(response.text)[:300]}')


def create_adapter(config: Dict[str, Any]) -> AIGCApiAdapter:
    """Factory: create the appropriate API adapter from model config.

    Args:
        config: Model config dict with keys: provider, api_base, api_key, etc.

    Returns:
        AIGCApiAdapter subclass instance.
    """
    provider = config.get('provider', 'openai') or 'openai'

    if config.get('tool', config.get('model_type')) == 'txt2img':
        if provider == 'dashscope':
            return DashScopeImageAdapter(config)
        if provider not in ('openai', 'custom'):
            raise ValueError('文生图当前支持 OpenAI 兼容和百炼提供商')

    if config.get('tool', config.get('model_type')) == 'img2img':
        if provider == 'dashscope':
            return DashScopeImageEditAdapter(config)
        if provider not in ('openai', 'custom'):
            raise ValueError('图生图当前支持 OpenAI 兼容和百炼千问')

    if provider in ('openai', 'custom'):
        return OpenAICompatibleAdapter(config)

    if provider == 'dashscope':
        return DashScopeVideoAdapter(config)

    if provider == 'volcengine':
        if config.get('model_type', 'txt2video') != 'txt2video':
            raise ValueError('Volcengine adapter currently supports txt2video only')
        return VolcengineVideoAdapter(config)

    raise ValueError(f'Unsupported AIGC API provider: {provider}')


def _w_h_to_resolution(width: int, height: int) -> str:
    """Convert WxH to resolution string for size fallback."""
    if height <= 480:
        return '480p'
    if height <= 720:
        return '720p'
    return '1080p'


_ASPECT_RATIO_PRESETS = {
    (16, 9): '16:9',
    (9, 16): '9:16',
    (4, 3): '4:3',
    (3, 4): '3:4',
    (1, 1): '1:1',
}


def _wh_to_aspect_ratio(width: int, height: int) -> str:
    """Convert WxH to the nearest standard aspect ratio string (16:9 / 9:16 / 4:3 / 3:4 / 1:1).

    Used when the frontend did not pass an explicit ``ratio`` (e.g. 1280x720 → 16:9).
    """
    if width <= 0 or height <= 0:
        return '16:9'
    import math
    g = math.gcd(width, height)
    w, h = width // g, height // g
    if (w, h) in _ASPECT_RATIO_PRESETS:
        return _ASPECT_RATIO_PRESETS[(w, h)]
    # Fallback: nearest preset by decimal ratio
    ratio = width / height
    best, best_diff = '16:9', float('inf')
    for (pw, ph), name in _ASPECT_RATIO_PRESETS.items():
        diff = abs(pw / ph - ratio)
        if diff < best_diff:
            best, best_diff = name, diff
    return best
