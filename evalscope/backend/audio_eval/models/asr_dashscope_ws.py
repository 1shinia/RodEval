"""ASR model adapter — DashScope WebSocket (paraformer-realtime / streaming ASR).

DashScope ASR WebSocket protocol:
  1. Connect to wss://{host}/api-ws/v1/inference
  2. Send run-task message (streaming='in')
  3. Wait for task-started event
  4. Send audio bytes
  5. Send finish-task message
  6. Receive result-generated events with sentence.text
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

HEADER = "header"
PAYLOAD = "payload"
TASK_ID = "task_id"
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


class AsrModelDashScopeWS(AudioModelBase):
    """ASR via DashScope WebSocket — paraformer-realtime / streaming models."""

    TASK_GROUP = "audio"
    TASK = "asr"
    FUNCTION = "Transcription"

    @property
    def _ws_url(self) -> str:
        base = _base_url(self.api_base)
        ws_base = base.replace("https://", "wss://", 1)
        if not ws_base.startswith("wss://"):
            ws_base = f"wss://{ws_base}"
        return f"{ws_base}/api-ws/v1/inference"

    def _generate_api(self, audio_path: str, **kwargs) -> Dict[str, Any]:
        """Read audio file, send via WebSocket, return transcription."""
        import os as _os
        language = kwargs.get("language", self.language)

        # Read audio file
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

        # Detect format from file extension
        ext = _os.path.splitext(audio_path)[1].lower().lstrip(".")
        audio_format = ext if ext in ("wav", "mp3", "pcm", "flac", "ogg") else "wav"

        url = self._ws_url
        task_id = uuid.uuid4().hex

        payload = {
            "model": self.model_name,
            "task_group": self.TASK_GROUP,
            "task": self.TASK,
            "function": self.FUNCTION,
            "input": {},
            "parameters": {
                "format": audio_format,
            },
        }
        # Detect actual sample rate for correct ASR decoding
        try:
            from mutagen.mp3 import MP3
            audio_info = MP3(audio_path)
            if audio_info.info.sample_rate:
                payload["parameters"]["sample_rate"] = audio_info.info.sample_rate
        except Exception:
            pass
        if language and language != "auto":
            payload["parameters"]["language_hints"] = [language]

        start_msg = {
            HEADER: {
                TASK_ID: task_id,
                "streaming": "in",
                "action": WsAction.START,
            },
            PAYLOAD: payload,
        }

        finish_msg = {
            HEADER: {
                TASK_ID: task_id,
                "streaming": "in",
                "action": WsAction.FINISH,
            },
            PAYLOAD: {"input": {}},
        }

        logger.info(
            f"DashScope WS ASR: url={url}, model={self.model_name}, "
            f"lang={language}, audio={len(audio_bytes)} bytes"
        )

        headers = {"Authorization": f"bearer {self.api_key}"}

        try:
            with ws_connect(
                url, additional_headers=headers, open_timeout=30
            ) as ws:
                # 1) Send start
                ws.send(json.dumps(start_msg, ensure_ascii=False))
                logger.debug(f"WS start task sent: {task_id}")

                # 2) Wait for task-started
                started = False
                while not started:
                    raw = ws.recv(timeout=30)
                    if isinstance(raw, bytes):
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
                            f"DashScope WS ASR start failed: {err} - {err_msg}"
                        )

                # 3) Send audio
                ws.send(audio_bytes)
                logger.debug(f"WS audio sent: {len(audio_bytes)} bytes")

                # 4) Send finish
                ws.send(json.dumps(finish_msg, ensure_ascii=False))
                logger.debug("WS finish-task sent")

                # 5) Receive results — last sentence event has full text
                last_text = ""
                while True:
                    raw = ws.recv(timeout=120)
                    if isinstance(raw, bytes):
                        continue
                    msg = json.loads(raw)
                    hdr = msg.get(HEADER, {})
                    evt = hdr.get(EVENT, "")

                    if evt == WsEvent.GENERATED:
                        pl = msg.get(PAYLOAD, {})
                        output = pl.get("output", pl)
                        st = output.get("sentence", {})
                        text = st.get("text", "")
                        if text:
                            last_text = text
                            logger.debug(
                                f"WS ASR sentence: {text[:60]}..."
                            )
                    elif evt == WsEvent.FINISHED:
                        logger.debug("WS task-finished received")
                        break
                    elif evt == WsEvent.FAILED:
                        err = hdr.get(ERROR_NAME, "Unknown")
                        err_msg = hdr.get(ERROR_MESSAGE, "")
                        raise RuntimeError(
                            f"DashScope WS ASR failed: {err} - {err_msg}"
                        )

        except websockets.exceptions.ConnectionClosed as e:
            raise RuntimeError(
                f"DashScope WS ASR connection closed: {e}"
            ) from e

        transcription = last_text
        logger.info(
            f"DashScope WS ASR result: {transcription[:100]}..."
        )
        return {"text": transcription}

    def _generate_local(self, audio_path: str, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError(
            "Local ASR model loading not yet implemented"
        )

    def generate(self, audio_path: str, **kwargs) -> Dict[str, Any]:
        return self._generate_api(audio_path, **kwargs)
