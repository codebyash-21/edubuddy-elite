"""
Text-to-speech: Meta MMS-TTS (VITS), one small CPU voice per language.

facebook/mms-tts-eng / -hin / -tam all have is_uroman=false, so Devanagari and
Tamil script are fed in directly with no romanisation step. Long replies are
split on sentence boundaries and concatenated, which keeps VITS stable and
avoids a very long single forward pass.
"""
from __future__ import annotations

import io
import re
import threading
import wave

import numpy as np

import config

_models = {}          # lang -> (model, tokenizer)
_lock = threading.Lock()

# Fewer surviving tokens than this and the chunk is not worth synthesising.
_MIN_TOKENS = 4


def load(lang: str, verbose: bool = True):
    """Load and cache the VITS voice for one language."""
    lang = config.normalise_lang(lang)
    if lang in _models:
        return _models[lang]

    with _lock:
        if lang in _models:
            return _models[lang]

        import torch
        from transformers import AutoTokenizer, VitsModel

        repo = config.TTS_REPOS[lang]
        if verbose:
            print(f"[tts] loading {repo} ...")

        tokenizer = AutoTokenizer.from_pretrained(repo, cache_dir=str(config.TTS_DIR))
        model = VitsModel.from_pretrained(repo, cache_dir=str(config.TTS_DIR))
        model.eval()
        torch.set_num_threads(max(1, config.LLM_THREADS))

        _models[lang] = (model, tokenizer)
        if verbose:
            print(f"[tts] {lang} ready")
    return _models[lang]


def is_ready(lang: str = None) -> bool:
    if lang is None:
        return len(_models) == len(config.LANGUAGES)
    return config.normalise_lang(lang) in _models


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?。।!?])\s+|\n+")


def _chunk(text: str, limit: int):
    """Split into speakable pieces no longer than `limit` characters."""
    parts, buf = [], ""
    for sentence in _SENTENCE_SPLIT.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(buf) + len(sentence) + 1 <= limit:
            buf = f"{buf} {sentence}".strip()
        else:
            if buf:
                parts.append(buf)
            # A single sentence longer than the limit gets hard-split.
            while len(sentence) > limit:
                parts.append(sentence[:limit])
                sentence = sentence[limit:]
            buf = sentence
    if buf:
        parts.append(buf)
    return parts or [text[:limit]]


def synth(text: str, lang: str = "en"):
    """
    Synthesise speech.

    Returns (waveform float32 in [-1, 1], sample_rate).
    """
    import torch

    lang = config.normalise_lang(lang)
    text = (text or "").strip()
    if not text:
        return np.zeros(0, dtype=np.float32), 16000

    model, tokenizer = load(lang)
    sample_rate = int(model.config.sampling_rate)

    pieces = []
    skipped = 0
    for part in _chunk(text, config.TTS_MAX_CHARS):
        inputs = tokenizer(part, return_tensors="pt")

        # MMS-TTS normalises against a fixed per-language vocabulary and silently
        # drops everything outside it. A chunk that is mostly Latin letters or
        # digits can therefore reduce to (almost) nothing, and VITS then emits a
        # burst of noise instead of speech. Skip such chunks rather than speak
        # garbage.
        ids = inputs.get("input_ids")
        if ids is None or ids.shape[-1] < _MIN_TOKENS:
            skipped += 1
            continue

        with torch.no_grad():
            out = model(**inputs).waveform
        pieces.append(out.squeeze().cpu().numpy().astype(np.float32))
        # Short pause between sentences.
        pieces.append(np.zeros(int(sample_rate * 0.15), dtype=np.float32))

    if skipped:
        print(f"[tts] skipped {skipped} chunk(s) with no speakable {lang} "
              f"characters (likely Latin text or digits)")
    if not pieces:
        print(f"[tts] nothing speakable in {lang}; returning silence")
        return np.zeros(0, dtype=np.float32), sample_rate

    wav = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)

    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if peak > 0:
        wav = (wav / peak) * 0.95

    return wav, sample_rate


def to_wav_bytes(wav: np.ndarray, sample_rate: int) -> bytes:
    """Pack a float32 waveform into a 16-bit PCM WAV blob for the browser."""
    pcm = np.clip(wav, -1.0, 1.0)
    pcm = (pcm * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(pcm.tobytes())
    return buf.getvalue()


def speak_bytes(text: str, lang: str = "en") -> bytes:
    wav, sr = synth(text, lang)
    return to_wav_bytes(wav, sr)
