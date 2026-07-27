"""ASR model adapter — local (faster-whisper: CPU-friendly, no GPU needed).

Uses faster-whisper (CTranslate2 backend, torch-independent) for local ASR.
Supports whisper-tiny/base/small/medium/large models.
"""
import logging
from pathlib import Path
from typing import Any, Dict

from .base import AudioModelBase

logger = logging.getLogger(__name__)

# Default model cache dir
_DEFAULT_MODEL_DIR = "/root/.cache/whisper"


class AsrModelLocal(AudioModelBase):
    """Local ASR via faster-whisper — CPU, no GPU, no API key."""

    def load(self):
        """Lazy-load the whisper model on first use."""
        if self._model is not None:
            return

        model_path = self.config.get("model_name_or_path", "")
        model_name = "small"  # default

        if model_path:
            p = Path(model_path)
            # If path is a directory, try to detect model size from dir name
            # e.g. /root/.cache/whisper → use "small"
            # Or if it contains "models--Systran--faster-whisper-small" → "small"
            dir_name = p.name.lower()
            for size in ("tiny", "base", "small", "medium", "large"):
                if size in dir_name or size in str(p):
                    model_name = size
                    break

        try:
            from faster_whisper import WhisperModel

            logger.info(f"Loading faster-whisper model: {model_name} (CPU)")
            self._model = WhisperModel(
                model_name,
                device="cpu",
                compute_type="int8",
                download_root=_DEFAULT_MODEL_DIR,
            )
            logger.info(f"Whisper model {model_name} loaded")
        except Exception as e:
            logger.error(f"Failed to load whisper model: {e}")
            raise RuntimeError(
                f"Whisper model load failed: {e}. "
                f"Ensure faster-whisper is installed and model is downloaded."
            ) from e

    def _generate_api(self, audio_path: str, **kwargs) -> Dict[str, Any]:
        """Transcribe audio file using local whisper model."""
        self.load()

        language = kwargs.get("language", self.language)
        lang_code = None if language == "auto" else language

        logger.info(
            f"Local ASR: model={self._model}, file={audio_path}, lang={lang_code}"
        )

        try:
            segments, info = self._model.transcribe(
                audio_path,
                language=lang_code,
                beam_size=5,
                vad_filter=True,
            )
            text = "".join(s.text for s in segments)
            logger.info(
                f"Local ASR result: {text[:100]}... "
                f"(lang={info.language}, prob={info.language_probability:.2f})"
            )
            return {"text": text}
        except Exception as e:
            logger.error(f"Local ASR transcription failed: {e}")
            raise RuntimeError(f"Local ASR failed: {e}") from e

    def _generate_local(self, audio_path: str, **kwargs) -> Dict[str, Any]:
        return self._generate_api(audio_path, **kwargs)

    def generate(self, audio_path: str, **kwargs) -> Dict[str, Any]:
        return self._generate_api(audio_path, **kwargs)
