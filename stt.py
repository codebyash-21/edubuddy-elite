"""
Speech-to-text: faster-whisper wrapper.

Transcribes either a numpy float32 array (terminal mode) or an audio file /
bytes blob (web mode, where the browser sends WebM/Opus). Whisper's own
language identification is used, then clamped to en/hi/ta.
"""
from __future__ import annotations

import io
import threading

import numpy as np

import config

_model = None
_lock = threading.Lock()


def load(verbose: bool = True):
    """Load the Whisper model once. Safe to call from several threads."""
    global _model
    if _model is not None:
        return _model
    with _lock:
        if _model is not None:
            return _model
        from faster_whisper import WhisperModel

        if verbose:
            print(f"[stt] loading faster-whisper '{config.WHISPER_MODEL}' "
                  f"({config.WHISPER_COMPUTE_TYPE}) ...")
        _model = WhisperModel(
            config.WHISPER_MODEL,
            device=config.WHISPER_DEVICE,
            compute_type=config.WHISPER_COMPUTE_TYPE,
            download_root=str(config.WHISPER_DIR),
        )
        if verbose:
            print("[stt] ready")
    return _model


def is_ready() -> bool:
    return _model is not None


def transcribe(audio, language=None):
    """
    Transcribe audio.

    audio     : np.ndarray (float32, 16 kHz mono) | bytes | str path | file object
    language  : force a language ("en"/"hi"/"ta"), or None to auto-detect

    Returns (text, detected_language).
    """
    model = load()

    if isinstance(audio, bytes):
        audio = io.BytesIO(audio)
    if isinstance(audio, np.ndarray):
        audio = audio.astype(np.float32, copy=False)

    lang = config.normalise_lang(language) if language else None

    segments, info = model.transcribe(
        audio,
        language=lang,
        beam_size=config.WHISPER_BEAM_SIZE,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 400},
        condition_on_previous_text=False,
        task="transcribe",
    )

    text = "".join(seg.text for seg in segments).strip()
    detected = config.normalise_lang(lang or getattr(info, "language", "en"))
    return text, detected


def transcribe_file(path, language=None):
    return transcribe(path, language)
