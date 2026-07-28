"""TTS model adapter — DashScope WebSocket (Sambert / streaming TTS models).

DashScope TTS WebSocket protocol:
  1. Connect to wss://{host}/api-ws/v1/inference
  2. Send run-task message with model/input/parameters
  3. Wait for task-started event
  4. Receive audio as BINARY frames
  5. Receive task-finished event
"""
import json
import logging
import uuid
from typing import Any, Dict

import websockets
from websockets.sync.client import connect as ws_connect

from .asr_dashscope import _base_url
from .base import AudioModelBase

logger = logging.getLogger(__name__.replace('evalscope', 'evalperf'))

# ── protocol constants (mirrors dashscope.protocol.websocket) ──────────────

HEADER = "header"
PAYLOAD = "payload"
TASK_ID = "task_id"
ACTION = "action"
EVENT = "event"
ERROR_NAME = "error_code"
ERROR_MESSAGE = "error_message"


class WsEvent:
    STARTED = "task-started"
    GENERATED = "result-generated"
    FINISHED = "task-finished"
    FAILED = "task-failed"


class WsAction:
    START = "run-task"
    FINISH = "finish-task"


# ── adapter ────────────────────────────────────────────────────────────────


class TtsModelDashScopeWS(AudioModelBase):
    """TTS via DashScope WebSocket — Sambert / streaming models."""

    TASK_GROUP = "audio"
    TASK = "tts"
    FUNCTION = "SpeechSynthesizer"

    @property
    def _ws_url(self) -> str:
        base = _base_url(self.api_base)
        # Replace https:// with wss://
        ws_base = base.replace("https://", "wss://", 1)
        if not ws_base.startswith("wss://"):
            ws_base = f"wss://{ws_base}"
        return f"{ws_base}/api-ws/v1/inference"

    def _generate_api(self, prompt: str, **kwargs) -> bytes:
        """Call TTS via WebSocket, collect audio bytes."""
        url = self._ws_url
        voice = kwargs.get("voice", self.config.get("voice", "longxiaochun"))
        fmt = kwargs.get(
            "response_format", self.config.get("response_format", "mp3")
        )
        speed = float(kwargs.get("speed", self.config.get("speed", 1.0)))

        task_id = uuid.uuid4().hex

        payload = {
            "model": self.model_name,
            "task_group": self.TASK_GROUP,
            "task": self.TASK,
            "function": self.FUNCTION,
            "input": {"text": prompt},
            "parameters": {
                "voice": voice,
                "format": fmt,
            },
        }
        if speed != 1.0:
            payload["parameters"]["rate"] = speed

        start_msg = {
            HEADER: {
                TASK_ID: task_id,
                "streaming": "out",
                ACTION: WsAction.START,
            },
            PAYLOAD: payload,
        }

        logger.info(
            f"DashScope WS TTS: url={url}, model={self.model_name}, "
            f"voice={voice}, fmt={fmt}, speed={speed}"
        )

        audio_chunks: list[bytes] = []
        headers = {"Authorization": f"bearer {self.api_key}"}

        try:
            with ws_connect(url, additional_headers=headers, open_timeout=30) as ws:
                # 1) Send start task
                ws.send(json.dumps(start_msg, ensure_ascii=False))
                logger.debug(f"WS start task sent: {task_id}")

                # 2) Wait for task-started
                started = False
                while not started:
                    raw = ws.recv(timeout=30)
                    if isinstance(raw, bytes):
                        # Ignore binary during startup
                        continue
                    msg = json.loads(raw)
                    hdr = msg.get(HEADER, {})
                    evt = hdr.get(EVENT, "")
                    if evt == WsEvent.STARTED:
                        started = True
                        logger.debug("WS task-started received")
                    elif evt == WsEvent.FAILED:
                        err = hdr.get(ERROR_NAME, "Unknown")
                        err_msg = hdr.get(ERROR_MESSAGE, "")
                        raise RuntimeError(
                            f"DashScope WS TTS task failed: {err} - {err_msg}"
                        )
                    else:
                        logger.warning(f"Unexpected WS event during start: {evt}")

                # 3) Receive audio chunks until FINISHED
                while True:
                    raw = ws.recv(timeout=120)
                    if isinstance(raw, bytes):
                        # Binary = audio data
                        audio_chunks.append(raw)
                        logger.debug(f"WS audio chunk: {len(raw)} bytes")
                        continue

                    msg = json.loads(raw)
                    hdr = msg.get(HEADER, {})
                    evt = hdr.get(EVENT, "")

                    if evt == WsEvent.GENERATED:
                        # Text payload (e.g. sentence timestamps) — skip
                        continue
                    elif evt == WsEvent.FINISHED:
                        logger.debug("WS task-finished received")
                        break
                    elif evt == WsEvent.FAILED:
                        err = hdr.get(ERROR_NAME, "Unknown")
                        err_msg = hdr.get(ERROR_MESSAGE, "")
                        raise RuntimeError(
                            f"DashScope WS TTS failed: {err} - {err_msg}"
                        )
                    else:
                        logger.warning(f"Unexpected WS event: {evt}")

        except websockets.exceptions.ConnectionClosed as e:
            raise RuntimeError(
                f"DashScope WS TTS connection closed unexpectedly: {e}"
            ) from e

        if not audio_chunks:
            raise RuntimeError(
                "DashScope WS TTS: no audio data received"
            )

        audio_bytes = b"".join(audio_chunks)
        logger.info(f"DashScope WS TTS generated {len(audio_bytes)} bytes")
        return audio_bytes

    def _generate_local(self, prompt: str, **kwargs) -> bytes:
        raise NotImplementedError(
            "Local TTS model loading not yet implemented"
        )

    def generate(self, prompt: str, **kwargs) -> bytes:
        return self._generate_api(prompt, **kwargs)
